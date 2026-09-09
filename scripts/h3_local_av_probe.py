"""Short local modality diagnostic, confined to a previously rated clip."""
import argparse
import hashlib
import json
import os
import time

try:
    from .h3_local_av_evaluator import (ROOT, REPO, POLICY, MANIFEST, read_json, digest, now,
        save_new, isolate, inventory, audio_control, decode_audio, effective_video_fps, check_generation_queue, stopping_ids)
except ImportError:
    from h3_local_av_evaluator import (ROOT, REPO, POLICY, MANIFEST, read_json, digest, now,
        save_new, isolate, inventory, audio_control, decode_audio, effective_video_fps, check_generation_queue, stopping_ids)

PROTOCOL = REPO / "docs/research/data/2026-09-09-h3-local-av-evaluator-amendment-03.json"
PROMPT = "Describe only the sounds you actually hear in at most three short sentences. Mention whether speech or music is audible. If you hear no sound, say so. Do not infer sounds from visible actions."


def run(name):
    if not name or name in (".", "..") or "/" in name:
        raise ValueError("A simple new run name is required")
    isolate(offline=True)
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from qwen_omni_utils import fetch_video
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

    protocol = read_json(PROTOCOL)
    if protocol["parent_policy_sha256"] != digest(POLICY):
        raise ValueError("Unexpected policy")
    manifest, receipt = read_json(MANIFEST), read_json(ROOT / "setup-receipt.json")
    if manifest["policy_sha256"] != digest(POLICY) or receipt["policy_sha256"] != digest(POLICY):
        raise ValueError("Source policy mismatch")
    case = next(c for c in manifest["cases"] if c["source_job"] ==
                "av2-boxing-2026090803-combat-cinema-combat")
    if digest(case["path"]) != case["sha256"]:
        raise ValueError("Previously rated source changed")
    model_dir = ROOT / "model"
    for asset in receipt["files"]:
        st = (model_dir / asset["name"]).stat()
        if (st.st_size, st.st_mtime_ns) != (asset["bytes"], asset["mtime_ns"]):
            raise ValueError("Hash-verified model asset changed")
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise RuntimeError("GPU unavailable or insufficient free memory")
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, exist_ok=False)
    save_new(directory / "start.json", {"started_at_utc": now(), "protocol_sha256": digest(PROTOCOL),
        "helper_sha256": digest(__file__), "input_helper_sha256": digest(REPO / "scripts/h3_local_av_evaluator.py"),
        "manifest_sha256": digest(MANIFEST), "receipt_sha256": digest(ROOT / "setup-receipt.json"),
        "source_sha256": case["sha256"], "queue_before": check_generation_queue(),
        "prompt": PROMPT, "packages": inventory(), "machine_diagnostic_only": True})
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_dir,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if (processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id) != (151645, 151643):
        raise ValueError("Tokenizer generation IDs differ from verified config")
    print(json.dumps({"event": "model_loaded", "file_generation_eos": model.generation_config.eos_token_id,
                      "explicit_eos": processor.tokenizer.eos_token_id}), flush=True)
    audio, audio_metadata = decode_audio(case["path"])
    element = {"type": "video", "video": case["path"], "fps": 4.0,
               "min_pixels": 100352, "max_pixels": 200704}
    (video, metadata), _ = fetch_video(element, return_video_sample_fps=True, return_video_metadata=True)
    effective_fps = effective_video_fps(metadata["frames_indices"].tolist(), metadata["fps"])
    jobs = [(m, c, 0) for m in ("audio_only", "audiovisual") for c in ("original", "muted")]
    jobs += [("audiovisual", c, 1) for c in ("original", "muted")]
    for modality, condition, repeat in jobs:
        queue = check_generation_queue()
        waveform = audio_control(audio, condition)
        media = element if modality == "audiovisual" else {"type": "audio", "audio": waveform}
        conversation = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                        {"role": "user", "content": [media, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        if case["path"] in text or case["source_job"] in text:
            raise ValueError("Source identity leaked")
        kwargs = {} if modality == "audio_only" else {"videos": [video], "videos_kwargs": {
            "fps": effective_fps, "use_audio_in_video": True, "do_sample_frames": False,
            "min_pixels": 100352, "max_pixels": 200704}}
        inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True, **kwargs)
        if "input_features" not in inputs or not inputs["feature_attention_mask"].sum():
            raise ValueError("Audio features missing")
        result = {"modality": modality, "condition": condition, "repeat": repeat,
                  "source_sha256": case["sha256"], "audio_metadata": audio_metadata,
                  "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                  "waveform_peak": float(np.abs(waveform).max()), "queue_before": queue,
                  "feature_frames": int(inputs["feature_attention_mask"].sum()),
                  "input_shapes": {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")},
                  "eos_token_id": processor.tokenizer.eos_token_id,
                  "generation": protocol["diagnostic_generation"]}
        inputs = inputs.to(model.device).to(model.dtype)
        model.rope_deltas = None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, use_audio_in_video=(modality == "audiovisual"),
                **stopping_ids(processor.tokenizer),
                **protocol["diagnostic_generation"])
        torch.cuda.synchronize()
        ids = output[0, inputs["input_ids"].shape[1]:].tolist()
        result.update(raw_response=processor.tokenizer.decode(ids, skip_special_tokens=True),
            generated_token_ids=ids, stopped_on_eos=bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
            inference_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        ident = f"{modality}-{condition}-r{repeat}"
        save_new(directory / f"{ident}.json", result)
        print(json.dumps({"event": "probe_complete", "case": ident, "stopped_on_eos": result["stopped_on_eos"],
                          "text": result["raw_response"]}), flush=True)
        del inputs, output
    save_new(directory / "complete.json", {"completed_at_utc": now(), "cases": len(jobs),
                                          "rating_or_heldout_qualification": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    run(parser.parse_args().run_name)
