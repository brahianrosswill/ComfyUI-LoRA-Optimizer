"""Stream a full native H3 TIES export through real ComfyUI, one component at a time.

Reconstruct raw source deltas with native mapping (not optimizer normalization).
The TIES formula is reused from the pinned optimizer: this validates storage and
loader fidelity, not the algorithm's perceptual merits. Report normal storage
rounding separately from serialization discrepancy. Never rewrite old reports.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

try:
    from .h3_benchmark import digest, save_new
    from .h3_merge_study import header, check_gpu_headroom
    from .h3_native_export_check import component_rows, load_native, require_coverage
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_merge_study import header, check_gpu_headroom
    from h3_native_export_check import component_rows, load_native, require_coverage


def export_groups(tensor_header, source_targets):
    """Only exact native .diff or complete plain factors; no ignored keys."""
    groups, consumed = {}, set()
    for target in source_targets:
        if not isinstance(target, str) or not target.endswith(".weight"):
            raise ValueError("Expected complete native weight targets")
        prefix = target.removesuffix(".weight")
        dense = {prefix + ".diff"}
        factors = {prefix + s for s in (".lora_up.weight", ".lora_down.weight", ".alpha")}
        if dense <= tensor_header.keys():
            names = dense
        elif factors <= tensor_header.keys():
            names = factors
        else:
            raise ValueError(f"Incomplete exported target: {target}")
        groups[target] = sorted(names)
        consumed.update(names)
    if consumed != set(tensor_header) - {"__metadata__"}:
        raise ValueError("Unexpected or competing export tensor keys")
    return groups


def audit(run, comfy):
    output = run / "dense_export_check.json"
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    root = Path(__file__).resolve().parents[1]
    if not manifest.get("mapped_export") or manifest["mode"] != "ties":
        raise ValueError("This checker requires a completed mapped TIES study run")
    for name, sha in manifest["source_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError("Production source differs from exported run")
    if digest(root / "scripts/h3_mapped_export.py") != manifest["mapped_writer_sha256"]:
        raise ValueError("Mapped writer differs from exported run")
    checkpoint_stat = Path(manifest["checkpoint"]).stat()
    if (checkpoint_stat.st_size != manifest["checkpoint_size"]
            or checkpoint_stat.st_mtime_ns != manifest["checkpoint_mtime_ns"]):
        raise ValueError("Shape-only checkpoint identity changed")
    for source in manifest["adapters"]:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Input adapter identity changed")
        if "alpha" in source["metadata"]:
            raise ValueError("File-level alpha requires separate policy validation")
    exported_path = Path(manifest["export"])
    if exported_path.stat().st_size != manifest["export_size"]:
        raise ValueError("Export size changed")
    tensor_header = header(exported_path)
    meta = tensor_header["__metadata__"]
    density, sign = manifest["ties_density"], manifest["ties_sign_method"]
    if (meta["merge_mode"] != "ties" or meta["merge_optimization_mode"] != "global"
            or float(meta["merge_ties_density"]) != density or meta["merge_ties_sign_method"] != sign
            or float(meta["merge_output_strength"]) != 1. or meta["merge_bake_strength"] != "True"
            or meta["lora_optimizer_h3_layout"] != "comfy"):
        raise ValueError("Export metadata differs from recorded merge")

    sys.path.insert(0, str(comfy))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.lora
    from safetensors import safe_open
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized GPU environment")
    check_gpu_headroom("merge", torch.cuda.mem_get_info()[0])
    torch.cuda.reset_peak_memory_stats()
    spec = importlib.util.spec_from_file_location("h3_dense_reference", root / "lora_optimizer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    node = module.LoRAOptimizer()
    base = header(manifest["checkpoint"])
    state = {"diffusion_model." + k: torch.empty(v["shape"], device="meta")
             for k, v in base.items() if k != "__metadata__" and k.endswith(".weight")}
    shape_model = types.SimpleNamespace(state_dict=lambda: state, model_config=types.SimpleNamespace(unet_config={}))
    mapping = comfy.lora.model_lora_keys_unet(shape_model, {})
    sources = [load_native(a["path"], mapping, torch, comfy.lora) for a in manifest["adapters"]]
    targets = set().union(*(set(s) for s in sources))
    groups = export_groups(tensor_header, targets)
    require_coverage(sources, groups)
    results = []
    for target in sorted(targets):
        names = groups[target]
        is_dense = len(names) == 1 and names[0].endswith(".diff")
        contributors = [i for i, source in enumerate(sources) if target in source]
        if is_dense != (len(contributors) > 1):
            raise ValueError("Unexpected dense/factor representation for contributor count")
        for component, rows in enumerate(component_rows(target, state[target].shape)):
            # Scope the real reader to this component: do not retain 72+ GiB
            # worth of previously accessed mapped pages in this process.
            with safe_open(str(exported_path), framework="pt", device="cpu") as reader:
                sd = {name: reader.get_slice(name)[rows, :].to("cuda")
                      if name.endswith((".diff", ".lora_up.weight")) else reader.get_tensor(name).to("cuda")
                      for name in names}
            if any(t.dtype != torch.float32 for t in sd.values()):
                raise ValueError("Mapped export storage is not FP32")
            patches = comfy.lora.load_lora(sd, mapping)
            if set(patches) != {target}:
                raise ValueError("Native loader mapped component to a different target")
            if not is_dense and set(patches[target].loaded_keys) != set(names):
                raise ValueError("Native loader did not consume every factor key")
            shape = (rows.stop - rows.start, state[target].shape[1])
            actual = comfy.lora.calculate_weight([(1., patches[target], 1., None, None)],
                                                 torch.zeros(shape, device="cuda"), target)
            del sd, patches
            diffs, storage_dtype = [], None
            for i in contributors:
                up, down, alpha, mid, dora, reshape = sources[i][target].weights
                if mid is not None or dora is not None or reshape is not None:
                    raise ValueError("Unsupported source adapter")
                item_dtype = torch.promote_types(up.dtype, down.dtype)
                storage_dtype = item_dtype if storage_dtype is None else torch.promote_types(storage_dtype, item_dtype)
                # Match ordinary dense reconstruction's operation order. Do
                # not pre-scale a factor, which could shift a TIES trim vote.
                scale = float(alpha) / down.shape[0] if alpha is not None else 1.
                delta = (up[rows].to("cuda").float() @ down.to("cuda").float()) * scale
                diffs.append((delta, manifest["strengths"][i]))
            additive = sum(d * w for d, w in diffs)
            expected = node._merge_diffs(list(diffs), "ties", density=density,
                majority_sign_method=sign, compute_device=torch.device("cuda"), keep_on_gpu=True)
            # Unique targets retain native FP32 factors, not rounded dense
            # matrices. Shared TIES targets follow ordinary storage rounding.
            rounded = expected.to(storage_dtype).float() if is_dense else expected
            if not torch.isfinite(rounded).all():
                rounded = expected
            norm = expected.norm().item()
            error = (actual - rounded).norm().item()
            results.append(dict(prefix=target, component=component, source_indices=contributors,
                representation="diff" if is_dense else "lora", normal_storage_dtype=str(storage_dtype) if is_dense else "native FP32 factors",
                relative_export_error=error / max(rounded.norm().item(), 1e-30),
                relative_storage_rounding_error=(rounded - expected).norm().item() / max(norm, 1e-30),
                relative_fp32_reference_error=(actual - expected).norm().item() / max(norm, 1e-30),
                relative_method_change=(expected - additive).norm().item() / max(additive.norm().item(), 1e-30),
                exact_stored_values=torch.equal(actual, rounded),
                finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())))
            del actual, expected, rounded, additive, diffs, delta
        print(f"Checked native {target}", flush=True)
    report = dict(mode="ties", ties_density=density, ties_sign_method=sign, all_targets=True,
        native_loader="comfy.lora; no optimizer key normalization; per-component safetensors reader",
        reference="Pinned ordinary TIES formula; normal storage rounding separately measured",
        groups_checked=len(results), native_targets=len(targets), source_native_targets=[len(s) for s in sources],
        all_source_and_export_keys_consumed=True, manifest_sha256=digest(manifest_path),
        export_sha256=digest(exported_path), checker_sha256=digest(__file__),
        comfy_lora_sha256=digest(comfy.lora.__file__), torch_version=torch.__version__,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-started,
        results=results)
    save_new(output, report)
    print(json.dumps(dict(output=str(output), groups=len(results),
        exact_stored_components=sum(r["exact_stored_values"] for r in results),
        max_relative_export_error=max(r["relative_export_error"] for r in results),
        max_relative_storage_rounding_error=max(r["relative_storage_rounding_error"] for r in results),
        finite=all(r["finite"] for r in results))))
    if not all(r["finite"] for r in results):
        raise ValueError("Non-finite export/reference; inspect preserved report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, default=Path("/media/p5/Comfyui"))
    args = parser.parse_args()
    audit(args.run, args.comfy)
