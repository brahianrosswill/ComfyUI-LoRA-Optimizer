"""Strict loading and alignment guards for isolated FATE research, not nodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from .h3_fate_setup import ROOT, REPO, SOURCE_REV, assets, digest, inventory, isolate, now, read_json, save_new
except ImportError:
    from h3_fate_setup import ROOT, REPO, SOURCE_REV, assets, digest, inventory, isolate, now, read_json, save_new


PREFIXES = {"audio_model.audio_encoder.": "audio_encoder.",
            "video_model.video_encoder.": "video_encoder.",
            "audio_model.audio_head.": "audio_head.",
            "video_model.video_head.": "video_head."}


def base_mapping(header, expected):
    """One-to-one, shape-exact mapping; never guess suffix matches or skip missing."""
    mapped = {}
    for key, entry in header.items():
        for source, target in PREFIXES.items():
            if key.startswith(source):
                destination = target + key[len(source):]
                if destination in mapped:
                    raise ValueError("Duplicate base destination")
                if destination not in expected or list(entry["shape"]) != list(expected[destination]):
                    raise ValueError(f"Unexpected or mismatched required tensor: {key}")
                mapped[destination] = key
    if set(mapped) != set(expected):
        raise ValueError(f"Missing required tensors: {sorted(set(expected) - set(mapped))}")
    if not mapped:
        raise ValueError("Empty model is not a valid evaluator")
    return mapped


def validate_alignment(video, audio, mask_video=None, mask_audio=None):
    """This harness supports one unpadded window only; reject upstream edge cases."""
    import torch
    if video.ndim != 3 or audio.ndim != 3 or video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("Only single-window N,T,D inputs are supported")
    if min(video.shape) < 1 or min(audio.shape) < 1 or video.shape[-1] != audio.shape[-1]:
        raise ValueError("Empty or mismatched hidden states")
    for mask, value in ((mask_video, video), (mask_audio, audio)):
        if mask is not None and (tuple(mask.shape) != tuple(value.shape[:2])
                                 or not bool(torch.all(mask == 1))):
            raise ValueError("Padded or inconsistent masks are unsupported, not silently realigned")


def import_source():
    isolate()
    source = ROOT / "source"
    rev = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if rev != SOURCE_REV or dirty:
        raise ValueError("Audited upstream source changed")
    sys.path.insert(0, str(source))
    from models.pe_av import PeAudioVideoConfig, PeAudioVideoModel, PeAudioVideoProcessor

    class GuardedFate(PeAudioVideoModel):
        def _align_video_hidden_state(self, video_hidden_state, audio_hidden_state,
                                     padding_mask_videos=None, padding_mask=None):
            validate_alignment(video_hidden_state, audio_hidden_state, padding_mask_videos, padding_mask)
            value = super()._align_video_hidden_state(video_hidden_state, audio_hidden_state,
                                                     padding_mask_videos, padding_mask)
            if value.shape != audio_hidden_state.shape:
                raise ValueError("Upstream temporal alignment returned an invalid shape")
            self.last_alignment = {
                "video_shape": list(video_hidden_state.shape), "audio_shape": list(audio_hidden_state.shape),
                "all_masks_valid": True, "unchanged_upstream_alignment": True,
                "video_index_for_audio_feature": [i * video_hidden_state.shape[1] // audio_hidden_state.shape[1]
                                                   for i in range(audio_hidden_state.shape[1])]}
            return value

    return PeAudioVideoConfig, GuardedFate, PeAudioVideoProcessor


def verify_download():
    receipt = read_json(ROOT / "download-receipt.json")
    for entry in assets():
        folder = ROOT / ("base" if entry["repo"] == "facebook/pe-av-small" else "adapter")
        for item in entry["files"]:
            if item["path"].endswith(".safetensors"):
                path = folder / item["path"]
                if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                    raise ValueError("Pinned checkpoint changed")
    metadata = ROOT / "metadata-receipt.json"
    if digest(metadata) != receipt["metadata_receipt_sha256"]:
        raise ValueError("Metadata receipt changed")
    for item in read_json(metadata)["files"]:
        if digest(REPO / item["file"]) != item["sha256"]:
            raise ValueError("Pinned model metadata changed")


def load_model():
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from peft import PeftModel, get_peft_model_state_dict
    verify_download()
    torch.set_num_threads(4)
    Config, Model, Processor = import_source()
    cfg = Config.from_pretrained(ROOT / "base", local_files_only=True)
    # Keep deterministic nonpersistent rotary buffers on CPU, not uninitialized meta.
    with init_empty_weights(include_buffers=False):
        model = Model(cfg)
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    header = read_json(ROOT / "base/model.safetensors.header.json")
    mapping = base_mapping(header, expected)
    with safe_open(ROOT / "base/model.safetensors", framework="pt", device="cpu") as reader:
        weights = {target: reader.get_tensor(source) for target, source in mapping.items()}
        if any(not bool(torch.isfinite(t).all()) for t in weights.values()):
            raise ValueError("Nonfinite base weights")
        model.load_state_dict(weights, strict=True, assign=True)
        if any(not torch.equal(model.state_dict()[key], tensor) for key, tensor in weights.items()):
            raise ValueError("Base weight copy not exact")
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Uninitialized meta parameter/buffer remains")
    report = {"required_base_tensors": len(mapping), "base_keys_total": len(header) - 1,
              "unused_base_keys": sorted(set(header) - {"__metadata__"} - set(mapping.values())),
              "base_mapping": mapping, "all_required_base_values_exact": True,
              "direct_unmapped_key_overlap": len(set(expected) & set(header)),
              "missing_required_base_tensors": [], "random_required_parameters": False}
    model = PeftModel.from_pretrained(model, ROOT / "adapter", is_trainable=False,
                                      local_files_only=True, use_safetensors=True)
    actual = get_peft_model_state_dict(model)
    with safe_open(ROOT / "adapter/adapter_model.safetensors", framework="pt", device="cpu") as reader:
        if set(reader.keys()) != set(actual):
            raise ValueError("Adapter key coverage mismatch")
        for key in reader.keys():
            wanted = reader.get_tensor(key)
            if (actual[key].shape != wanted.shape or not bool(torch.isfinite(wanted).all())
                    or not torch.equal(actual[key].float(), wanted.float())):
                raise ValueError(f"Adapter value not loaded exactly: {key}")
    report.update(adapter_tensors=len(actual), all_adapter_values_exact_after_dtype_promotion=True,
                  adapter_head_tensors=sum("_head." in k for k in actual),
                  adapter_target_modules=sorted(k.rsplit(".lora_", 1)[0] for k in actual if ".lora_A." in k),
                  attention_implementations=sorted({str(m.config._attn_implementation)
                      for m in model.modules() if hasattr(m, "config")}))
    model.requires_grad_(False).eval()
    processor = Processor.from_pretrained(ROOT / "base", local_files_only=True)
    return model, processor, report


def audit():
    import numpy as np
    import torch
    isolate()
    target = ROOT / "loading-audit.json"
    if target.exists():
        raise FileExistsError(target)
    model, processor, report = load_model()
    # Synthetic processor-only check: no media and no model forward pass.
    inputs = processor(videos=torch.zeros(48, 3, 384, 640, dtype=torch.uint8),
                       audio=np.zeros(96000, dtype=np.float32), sampling_rate=48000,
                       padding=True, return_tensors="pt")
    shapes = {k: list(v.shape) for k, v in inputs.items()}
    if (shapes.get("input_values") != [1, 1, 96000]
            or shapes.get("pixel_values_videos") != [1, 48, 3, 336, 336]
            or not bool(torch.all(inputs["padding_mask"] == 1))):
        raise ValueError(f"Unexpected processor shapes/masks: {shapes}")
    report.update(utc=now(), packages=inventory(), script_sha256=digest(__file__),
                  download_receipt_sha256=digest(ROOT / "download-receipt.json"),
                  synthetic_processor_shapes=shapes, source_revision=SOURCE_REV,
                  source_files={str(p.relative_to(REPO)): digest(p)
                                for p in sorted((ROOT / "source/models/pe_av").glob("*.py"))},
                  gpu_used=False, model_forward=False, media_loaded=False,
                  production_changed=False, qualified_for_quality=False)
    save_new(target, report)
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ("base_mapping", "unused_base_keys", "packages", "adapter_target_modules", "source_files")}), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
