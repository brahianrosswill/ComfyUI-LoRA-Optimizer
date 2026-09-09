"""Headless, local H3 merge experiments; does not start or restart ComfyUI.

Uses real adapter tensors and real ComfyUI patch mapping against a checkpoint's
shape-only module tree. This is valid only for base-independent additive LoRAs:
no DoRA, magnitude taming, or generation in this process. Render the exports
against the actual matching checkpoint separately. All artifacts stay in --out.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import resource
import struct
import sys
import time
import types


PAIRS = {
    "vbvr_cinema": ("concept/VBVR_H3_attn_only", "style/Cinema-MH3-V02_000010000"),
    "combat_cinema": ("concept/H3_Combat_V2", "style/Cinema-MH3-V02_000010000"),
    "combat_repair": ("concept/H3_Combat_V2", "concept/Motion_Repair"),
}
for _character, _file in {
    "sully": "h3sully_000002500",
    "series30": "MiniMax H3  AI Girl Fictional Women Series30",
}.items():
    for _effect, _path in {"cinema": PAIRS["combat_cinema"][1],
                           "combat": PAIRS["combat_cinema"][0]}.items():
        PAIRS[f"{_character}_{_effect}"] = (
            f"character/h3-identity-study-20260908/{_file}", _path)


def merge_options(mode, patch_compression, np_strength=.5, ct_scale=1.):
    """Explicit global stable arms; retain the pilot's existing defaults."""
    cfg = None
    if mode == "np_lora":
        cfg = dict(version=1, subject_slot=1, style_slot=2, strength=np_strength, rank=0, energy=1.)
    elif mode == "ct_merge":
        cfg = dict(version=1, common_rank=4, residual_rank=16, scale=ct_scale)
    elif mode not in ("additive", "per_prefix", "slerp", "ties"):
        raise ValueError("Unsupported study mode")
    if patch_compression not in ("smart", "aggressive", "disabled"):
        raise ValueError("Unsupported compression setting")
    overridden = cfg is not None or mode in ("slerp", "ties")
    return dict(optimization_mode="global" if overridden else mode,
                merge_strategy_override=mode if overridden else "",
                _experimental_config=cfg, patch_compression=patch_compression)


def header(path):
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 32 * 1024**2:
            raise ValueError("Unreasonable safetensors header size")
        return json.loads(stream.read(length))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_gpu_headroom(action, free_bytes):
    # Conservative guards from this study's full-size H3 measurements, not a
    # general memory estimator. An idle server may still retain 25+ GiB.
    required_gib = 20 if action == "tune" else 6
    if free_bytes < required_gib * 1024**3:
        raise RuntimeError(f"H3 study {action} needs at least {required_gib} GiB free GPU headroom; "
                           f"only {free_bytes / 1024**3:.2f} GiB is free. Check running jobs and "
                           "idle model caches first; this script never unloads or interrupts them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "merge", "tune", "replay"))
    parser.add_argument("--pair", choices=PAIRS, default="vbvr_cinema")
    parser.add_argument("--comfy", type=Path, default=Path("/media/p5/Comfyui"))
    parser.add_argument("--loras", type=Path, default=Path("/media/p5/ComfyUI-Model-CIFS-Cache/loras/MiniMax H3"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("additive", "per_prefix", "np_lora", "ct_merge", "slerp", "ties"), default="additive")
    parser.add_argument("--patch-compression", choices=("smart", "aggressive", "disabled"), default="aggressive")
    parser.add_argument("--strengths", type=float, nargs=2, default=(0.8, 0.8))
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--np-strength", type=float, default=0.5)
    parser.add_argument("--ct-scale", type=float, default=1.0)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--mapped-export", action="store_true",
                        help="Research-only full-target TIES export to bounded file-backed storage")
    parser.add_argument("--tuner-data", type=Path)
    parser.add_argument("--merge-seed", type=int, default=2026090800,
                        help="Research RNG seed for repeatable randomized compression; not a video seed")
    args = parser.parse_args()
    if args.mapped_export and (args.action != "merge" or args.mode != "ties" or args.patch_compression == "aggressive"):
        parser.error("Mapped export requires an explicit TIES merge without aggressive compression")
    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = args.comfy / "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    base_header = header(checkpoint)
    files = [args.loras / (name + ".safetensors") for name in PAIRS[args.pair]]
    manifest = {"pair": args.pair, "action": args.action, "mode": args.mode,
                "strengths": args.strengths, "checkpoint": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
                "shape_only_base": True, "tame_layers": 0, "adapters": [],
                "merge_seed": args.merge_seed,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_sha256": {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                  for name in ("lora_optimizer.py", "experimental_merge.py")}}
    for path in files:
        h = header(path)
        meta = h.pop("__metadata__", {})
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sidecar = json.loads(path.with_suffix(".metadata.json").read_text())
        if digest != sidecar["sha256"]:
            raise ValueError(f"Installed file differs from community identity: {path}")
        manifest["adapters"].append({"path": str(path), "sha256": digest,
            "metadata": meta, "model_name": sidecar["model_name"],
            "civitai_version_id": sidecar.get("civitai", {}).get("id"),
            "tensor_count": len(h), "shapes": {k: v["shape"] for k, v in h.items()}})
    write_json(args.out / "manifest.json", manifest)
    if args.action == "inspect":
        print(json.dumps({"manifest": str(args.out / "manifest.json"),
                          "models": [a["model_name"] for a in manifest["adapters"]]}))
        return

    # No server, checkpoint payload loading, or global Comfy configuration edits.
    sys.path.insert(0, str(args.comfy))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    from comfy.model_patcher import ModelPatcher
    from safetensors.torch import load_file
    folder_paths.set_user_directory(str(args.out / "user"))
    folder_paths.set_temp_directory(str(args.out / "temp"))
    # AutoTuner analysis caches are independent of its memory_mode setting.
    # Isolate models_dir before importing the optimizer's cache constants.
    folder_paths.models_dir = str(args.out / "models")
    folder_paths.add_model_folder_path("loras", str(args.loras))
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("h3_study_optimizer", root / "lora_optimizer.py")
    opt_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = opt_module
    spec.loader.exec_module(opt_module)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-size study; use the approved environment outside the sandbox")
    check_gpu_headroom(args.action, torch.cuda.mem_get_info()[0])
    torch.set_num_threads(4)
    torch.manual_seed(args.merge_seed)
    opt_module._LoRAMergeBase._get_compute_device = staticmethod(lambda: torch.device("cuda"))
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.FileHandler(args.out / "merge.log"), logging.StreamHandler()])

    class ShapeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_config = types.SimpleNamespace(unet_config={})
            self.lora_optimizer_h3_profile = {"partition": "fl2va", "basis": "pruned"}
            for key, info in base_header.items():
                if key == "__metadata__" or not key.endswith((".weight", ".bias")):
                    continue
                # The selected INT8 checkpoint stores unpacked matrix shapes.
                if info["dtype"] not in ("I8", "F16", "BF16", "F32"):
                    raise ValueError(f"Unsupported shape-only checkpoint storage: {key}")
                parent = self
                parts = ("diffusion_model." + key).split(".")
                for part in parts[:-1]:
                    if part not in parent._modules:
                        parent.add_module(part, torch.nn.Module())
                    parent = parent._modules[part]
                parent.register_parameter(parts[-1], torch.nn.Parameter(
                    torch.empty(info["shape"], device="meta", dtype=torch.bfloat16), requires_grad=False))

    model = ModelPatcher(ShapeModel(), torch.device("cpu"), torch.device("cpu"))
    stack = []
    for path, strength, info in zip(files, args.strengths, manifest["adapters"]):
        tensors = load_file(str(path))
        if any("dora" in k.lower() or "adaln" in k.lower() for k in tensors):
            raise ValueError("This pilot requires base-independent non-AdaLN adapters")
        stack.append({"name": str(path.relative_to(args.loras)), "lora": tensors,
                      "strength": strength, "metadata": info["metadata"], "h3_layout": "comfy"})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if args.action == "tune":
        node = opt_module.LoRAAutoTuner()
        result = node.auto_tune(model, stack, 1., top_n=args.top_n,
            normalize_keys="enabled", scoring_svd="disabled", scoring_device="gpu",
            scoring_speed="full", memory_mode="disabled", cache_patches="disabled",
            community_cache="disabled", diff_cache_mode="disabled", record_dataset="disabled",
            output_mode="tuning_only", vram_budget=0.35,
            experimental_options=({"np_lora": True, "ct_merge": True} if args.experimental else None))
        write_json(args.out / "tuner_data.json", result[4])
        (args.out / "ranking.txt").write_text(result[2])
    elif args.action == "replay":
        if args.tuner_data is None:
            raise ValueError("replay requires --tuner-data")
        tuner_data = json.loads(args.tuner_data.read_text())
        expected = [{"name": item["name"], "strength": item["strength"]} for item in stack]
        if tuner_data["source_loras"] != expected:
            raise ValueError("Replay stack differs from the recorded tuner stack")
        class ObservedSelector(opt_module.LoRAMergeSelector):
            def _compute_auto_strengths(self, *a, **kw):
                info = super()._compute_auto_strengths(*a, **kw)
                self.observed_auto_strength = info
                return info
        node = ObservedSelector()
        node.observed_auto_strength = None
        result = node.select_merge(model, stack, tuner_data, 1, 1.)
        if result[3] is None:
            raise ValueError("Selector did not produce a merge")
        decisions = result[3]["per_prefix_decisions"]
        selected = tuner_data["top_n"][0]
        if (selected["config"]["optimization_mode"] == "per_prefix"
                and (not selected.get("per_prefix_decisions") or decisions != selected["per_prefix_decisions"])):
            raise ValueError("Replayed per-prefix decisions differ from the recorded winner")
        (args.out / "merge_report.txt").write_text(result[2])
        export = opt_module.SaveMergedLoRA().save_lora(result[3], str(args.out), "merged")[0]
        manifest.update(export=export, export_size=Path(export).stat().st_size,
                        tuner_data=str(args.tuner_data), tuner_data_sha256=hashlib.sha256(args.tuner_data.read_bytes()).hexdigest(),
                        selected=selected, replay_per_prefix_decisions=decisions,
                        replay_auto_strength=node.observed_auto_strength,
                        patch_compression="smart")
    else:
        # Explicit compression is a study setting, NOT an invisible alteration
        # of normal AutoTuner behavior. Its numerical error needs separate checks.
        store = None
        if args.mapped_export:
            import comfy.lora
            try:
                from .h3_mapped_export import MappedPatchStore, layout_from_native
            except ImportError:
                from h3_mapped_export import MappedPatchStore, layout_from_native
            native_map = comfy.lora.model_lora_keys_unet(model.model, {})
            native = [comfy.lora.load_lora(item["lora"], native_map) for item in stack]
            for item, patches in zip(stack, native):
                if set(item["lora"]) != set().union(*(p.loaded_keys for p in patches.values())):
                    raise ValueError("Native source coverage is incomplete")
            layout = layout_from_native(native, {k: v.shape for k, v in model.model.state_dict().items()})
            store = MappedPatchStore(args.out / "merged.safetensors", layout)

            class OfflineMappedOptimizer(opt_module.LoRAOptimizer):
                def _new_patch_store(self, is_clip=False):
                    return {} if is_clip else store

                def _merge_diffs(self, diffs_with_weights, mode, **kwargs):
                    if mode == "ties":
                        self.tie_parameters.add((kwargs.get("density", .5), kwargs.get("majority_sign_method", "total")))
                    return super()._merge_diffs(diffs_with_weights, mode, **kwargs)

            node = OfflineMappedOptimizer()
            node.tie_parameters = set()
        else:
            node = opt_module.LoRAOptimizer()
        options = merge_options(args.mode, args.patch_compression, args.np_strength, args.ct_scale)
        result = node.optimize_merge(model, stack, 1.,
            **options, normalize_keys="enabled", cache_patches="disabled",
            svd_device="gpu", vram_budget=0., tame_layers=0.,
            _skip_model_apply=args.mapped_export, _skip_qkv_refusion=args.mapped_export)
        (args.out / "merge_report.txt").write_text(result[2])
        if store is not None:
            if len(node.tie_parameters) != 1:
                raise ValueError("Global TIES parameters were not consistent")
            density, sign = next(iter(node.tie_parameters))
            export = store.finish({"tool": "H3 offline mapped export study",
                "merge_mode": "ties", "merge_optimization_mode": "global",
                "merge_output_strength": "1.0", "merge_bake_strength": "True",
                "lora_optimizer_h3_layout": "comfy", "merge_ties_density": str(density),
                "merge_ties_sign_method": sign,
                "merge_metadata": json.dumps(result[4]["merge_metadata"], sort_keys=True),
                "description": "Lossless native FP32 serialization of normal TIES patches; no rank truncation"})
            manifest.update(mapped_export=True, native_targets=len(layout),
                streamed_components=store.components, ties_density=density, ties_sign_method=sign,
                mapped_writer_sha256=hashlib.sha256((root / "scripts/h3_mapped_export.py").read_bytes()).hexdigest())
            store.close()
        else:
            export = opt_module.SaveMergedLoRA().save_lora(result[4], str(args.out), "merged")[0]
        manifest["export"] = export
        manifest["export_size"] = Path(export).stat().st_size
        manifest["patch_compression"] = args.patch_compression
        manifest["merge_options"] = options
    manifest.update(elapsed_seconds=time.monotonic() - started,
                    torch_version=torch.__version__, cuda_device=torch.cuda.get_device_name(0),
                    peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                    max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("adapters", "shapes", "selected")}))
    # Release real patchers before interpreter teardown.
    del result, model, node


if __name__ == "__main__":
    main()
