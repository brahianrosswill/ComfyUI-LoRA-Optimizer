"""Trace joint AV input routing on one already-rated calibration source."""
import argparse
import ast
import hashlib
import inspect
import os
import time
import traceback

try:
    from .h3_local_av_probe import PROMPT
    from .h3_local_av_evaluator import (REPO, ROOT, POLICY, MANIFEST, read_json, save_new,
        digest, isolate, inventory, now, audio_control, decode_audio, effective_video_fps,
        stopping_ids, check_generation_queue)
except ImportError:
    from h3_local_av_probe import PROMPT
    from h3_local_av_evaluator import (REPO, ROOT, POLICY, MANIFEST, read_json, save_new,
        digest, isolate, inventory, now, audio_control, decode_audio, effective_video_fps,
        stopping_ids, check_generation_queue)

PROTOCOL = REPO / "docs/research/data/2026-09-09-h3-local-av-evaluator-amendment-04.json"


def token_runs(ids, audio_id, video_id):
    runs = []
    for token in ids:
        kind = "audio" if token == audio_id else "video" if token == video_id else "other"
        if runs and runs[-1]["kind"] == kind:
            runs[-1]["length"] += 1
        else:
            runs.append({"kind": kind, "length": 1})
    return runs


def default_system_prompt(module):
    """Read the installed processor's documented default as data, not instructions."""
    constants = [node.value for node in ast.walk(ast.parse(inspect.getsource(module)))
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)
                 and node.value.startswith("You are Qwen, a virtual human")]
    unique = set(constants)
    if len(unique) != 1:
        raise ValueError("Cannot unambiguously identify the pinned default system message")
    return unique.pop()


def run(name, protocol_number=4):
    if not name or name in (".", "..") or "/" in name:
        raise ValueError("A simple new run name is required")
    isolate(offline=True)
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from qwen_omni_utils import fetch_video
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

    protocol_path = (PROTOCOL if protocol_number == 4 else REPO /
                     "docs/research/data/2026-09-09-h3-local-av-evaluator-amendment-05.json")
    protocol, manifest, receipt = read_json(protocol_path), read_json(MANIFEST), read_json(ROOT / "setup-receipt.json")
    if any(p["parent_policy_sha256" if p is protocol else "policy_sha256"] != digest(POLICY)
           for p in (protocol, manifest, receipt)):
        raise ValueError("Frozen policy mismatch")
    case = next(c for c in manifest["cases"] if c["source_job"] == protocol["source_job"])
    if digest(case["path"]) != case["sha256"]:
        raise ValueError("Calibration source changed")
    for asset in receipt["files"]:
        stat = (ROOT / "model" / asset["name"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (asset["bytes"], asset["mtime_ns"]):
            raise ValueError("Verified model asset changed")
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise RuntimeError("GPU unavailable or less than 24 GiB free")
    directory = ROOT / "runs" / name
    directory.mkdir(parents=True, exist_ok=False)
    start = {"started_at_utc": now(), "protocol_sha256": digest(protocol_path),
        "source_sha256": case["sha256"], "receipt_sha256": digest(ROOT / "setup-receipt.json"),
        "manifest_sha256": digest(MANIFEST), "helper_sha256": digest(__file__),
        "dependency_sha256": {str(p): digest(p) for p in (
            REPO / "scripts/h3_local_av_evaluator.py", REPO / "scripts/h3_local_av_probe.py",
            REPO / "scripts/h3_local_av_setup.py")},
        "packages": inventory(), "prompt": PROMPT, "queue_before": check_generation_queue(),
        "cases": [{"route": r, "condition": c} for r in protocol["routes_in_order"]
                  for c in protocol["conditions_per_route"]]}
    save_new(directory / "start.json", start)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(ROOT / "model",
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(ROOT / "model", local_files_only=True, trust_remote_code=False)
    audio, audio_metadata = decode_audio(case["path"])
    element = {"type": "video", "video": case["path"], "fps": 4.0,
               "min_pixels": 100352, "max_pixels": 200704}
    (video, metadata), reader_fps = fetch_video(element, return_video_sample_fps=True, return_video_metadata=True)
    indices = metadata["frames_indices"].tolist()
    endpoint_fps = effective_video_fps(indices, metadata["fps"])
    audio_id, video_id = model.config.audio_token_index, model.config.video_token_index

    def tensor_info(tensor):
        cpu = tensor.detach().float().contiguous().cpu()
        return {"shape": list(cpu.shape), "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                "finite": bool(torch.isfinite(cpu).all()), "max_abs": float(cpu.abs().max())}

    for job in start["cases"]:
        route, condition = job["route"], job["condition"]
        result = dict(job, source_sha256=case["sha256"], queue_before=check_generation_queue())
        ident = f"{route}-{condition}"
        try:
            waveform = audio_control(audio, condition)
            audio_item = {"type": "audio", "audio": waveform}
            use_video = route != "audio_only"
            interleave = route.startswith("interleaved")
            contents = [element] if interleave else [element, audio_item] if use_video else [audio_item]
            question = protocol["question"] if route in ("interleaved_joint_question", "interleaved_official_joint") else PROMPT
            system = "You are a helpful assistant."
            if route in ("interleaved_official_system", "interleaved_official_joint"):
                from transformers.models.qwen2_5_omni import processing_qwen2_5_omni
                system = default_system_prompt(processing_qwen2_5_omni)
            contents = contents + [{"type": "text", "text": question}]
            conversation = [{"role": "system", "content": [{"type": "text", "text": system}]},
                            {"role": "user", "content": contents}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            if case["path"] in text or case["source_job"] in text:
                raise ValueError("Source identity leaked")
            fps = reader_fps if route == "interleaved_reader" else endpoint_fps
            kwargs = {} if not use_video else {"videos": [video], "videos_kwargs": {
                "fps": fps, "use_audio_in_video": interleave, "do_sample_frames": False,
                "min_pixels": 100352, "max_pixels": 200704}}
            inputs = processor(text=text, audio=[waveform], return_tensors="pt", padding=True, **kwargs)
            ids = inputs["input_ids"][0]
            count_a, count_v = int((ids == audio_id).sum()), int((ids == video_id).sum())
            if count_a != 129 or (count_v != 2400 if use_video else count_v != 0):
                raise ValueError("Unexpected fixed-source token coverage")
            result.update(audio_input=tensor_info(inputs["input_features"]),
                system_prompt=system, question=question,
                feature_frames=int(inputs["feature_attention_mask"].sum()),
                waveform_sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
                waveform_peak=float(np.abs(waveform).max()), video_used=use_video,
                frame_indices=indices if use_video else [], fps=fps if use_video else None,
                token_counts={"audio": count_a, "video": count_v, "all": len(ids)},
                token_runs=token_runs(ids.tolist(), audio_id, video_id),
                audio_decode=audio_metadata, generation=protocol["generation"],
                stopping_ids=stopping_ids(processor.tokenizer))
            inputs = inputs.to(model.device).to(model.dtype)
            audio_mask = inputs["input_ids"][0] == audio_id
            video_mask = inputs["input_ids"][0] == video_id
            prefill = {}

            def inspect_prefill(module, args, kwargs):
                embeds = kwargs.get("inputs_embeds")
                if prefill or embeds is None or embeds.shape[1] == 1:
                    return
                prefill["audio_embeddings"] = tensor_info(embeds[0, audio_mask])
                prefill["audio_position_ids"] = kwargs["position_ids"][:, 0, audio_mask].detach().cpu().tolist()
                if use_video:
                    prefill["video_embeddings"] = tensor_info(embeds[0, video_mask])
                    prefill["video_time_positions"] = kwargs["position_ids"][0, 0, video_mask].unique().detach().cpu().tolist()

            hook = model.model.register_forward_pre_hook(inspect_prefill, with_kwargs=True)
            model.rope_deltas = None
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = model.generate(**inputs, use_audio_in_video=interleave,
                                            **stopping_ids(processor.tokenizer), **protocol["generation"])
            finally:
                hook.remove()
            torch.cuda.synchronize()
            new_ids = output[0, inputs["input_ids"].shape[1]:].tolist()
            result.update(prefill=prefill, generated_token_ids=new_ids,
                raw_response=processor.tokenizer.decode(new_ids, skip_special_tokens=True),
                stopped_on_eos=bool(new_ids and new_ids[-1] == processor.tokenizer.eos_token_id),
                inference_seconds=time.perf_counter()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                status="complete")
            if not prefill:
                raise ValueError("Missing actual decoder-prefill trace")
            save_new(directory / f"{ident}.json", result)
            print({"case": ident, "text": result["raw_response"], "audio_embed_sha256": prefill["audio_embeddings"]["sha256"]}, flush=True)
            del inputs, output
        except Exception:
            result.update(status="failed", error=traceback.format_exc())
            if not (directory / f"{ident}.json").exists():
                save_new(directory / f"{ident}.json", result)
            raise
    save_new(directory / "complete.json", {"completed_at_utc": now(), "cases": len(start["cases"]),
                                          "quality_qualification": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", type=int, choices=(4, 5), default=4)
    args = parser.parse_args()
    run(args.run_name, args.protocol)
