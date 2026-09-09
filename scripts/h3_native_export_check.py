"""Full-target export fidelity through real ComfyUI's native key mapping.

Unlike the original pilot checker, this accepts Musubi and AI-Toolkit sources
without borrowing the optimizer's key normalizer. Only base-independent 2D
LoRA factors are supported. This checks serialization fidelity, not AV quality.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

try:
    from .h3_benchmark import digest, save_new
    from .h3_merge_study import header
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_merge_study import header


def factor_pair(patch, rows, device):
    """Preserve raw rank alpha and signed up/down values in FP32."""
    up, down, alpha, mid, dora, reshape = patch.weights
    if mid is not None or dora is not None or reshape is not None:
        raise ValueError("Native checker supports plain linear LoRA only")
    if up.ndim != 2 or down.ndim != 2 or up.shape[1] != down.shape[0]:
        raise ValueError("Invalid linear factor shapes")
    scale = float(alpha) / down.shape[0] if alpha is not None else 1.
    return up[rows].to(device=device).float() * scale, down.to(device=device).float()


def component_rows(target, shape):
    if len(shape) != 2:
        raise ValueError("Only linear target matrices are supported")
    splits = 3 if target.endswith(".qkv_proj.weight") else 1
    if shape[0] % splits:
        raise ValueError("Invalid native QKV output shape")
    return [slice(i * shape[0] // splits, (i + 1) * shape[0] // splits) for i in range(splits)]


def require_coverage(sources, exported):
    expected = set().union(*(set(source) for source in sources))
    if not expected or set(exported) != expected:
        raise ValueError(f"Native target coverage mismatch: missing={sorted(expected-set(exported))}, extra={sorted(set(exported)-expected)}")


def effective_mode(metadata):
    # The existing export stores the auto-detected strategy in merge_mode,
    # alongside the explicit optimization_mode that can override it.
    # Do not misinterpret an additive export as a weighted average.
    if metadata.get("merge_optimization_mode") == "additive":
        if json.loads(metadata.get("merge_experimental", "null")) is not None:
            raise ValueError("Additive export carries an experimental config")
        return "weighted_sum"
    if metadata.get("merge_optimization_mode") == "per_prefix":
        return "per_prefix"
    return metadata["merge_mode"]


def component_label(target, component):
    prefix = target.removesuffix(".weight")
    if prefix.endswith(".qkv_proj"):
        return prefix.removesuffix("qkv_proj") + ("to_q", "to_k", "to_v")[component]
    if component != 0:
        raise ValueError("Non-QKV target cannot have multiple components")
    return prefix


def replay_policy(manifest, metadata):
    """Refuse unsupported winner semantics rather than audit an easier blend."""
    selected = manifest["selected"]
    config = selected["config"]
    decisions = manifest["replay_per_prefix_decisions"]
    if (manifest["action"] != "replay" or config["optimization_mode"] != "per_prefix"
            or not decisions or decisions != selected["per_prefix_decisions"]
            or set(decisions.values()) - {"weighted_sum", "weighted_average", "normalize", "slerp"}
            or config["sparsification"] != "disabled" or config["merge_refinement"] != "none"
            or config.get("experimental") is not None):
        raise ValueError("Winner needs a different numerical reference")
    for key in ("sparsification", "merge_refinement", "auto_strength", "optimization_mode"):
        if metadata["merge_" + key] != config[key]:
            raise ValueError("Replay metadata/config mismatch")
    scale = 1.
    if config["auto_strength"] == "enabled":
        info = manifest["replay_auto_strength"]
        scale = info["model_scale"]
        if (not math.isfinite(scale) or not 0 <= scale <= 1
                or info["original_model_strengths"] != manifest["strengths"]
                or info["model_strengths"] != [w * scale for w in manifest["strengths"]]):
            raise ValueError("Invalid captured automatic strength")
    elif config["auto_strength"] != "disabled" or manifest["replay_auto_strength"] is not None:
        raise ValueError("Unexpected automatic strength policy")
    return decisions, scale


def load_native(path, mapping, torch, comfy_lora):
    from safetensors.torch import load_file
    sd = load_file(str(path), device="cpu")
    if any(not torch.isfinite(t).all().item() for t in sd.values()):
        raise ValueError(f"Non-finite adapter tensors: {path}")
    patches = comfy_lora.load_lora(sd, mapping)
    # The selected study exports and sources contain only plain LoRA factors.
    # Refuse dense/other adapters explicitly rather than silently omitting them.
    if any(not hasattr(p, "weights") for p in patches.values()):
        raise ValueError("This checker requires native low-rank export factors")
    consumed = set().union(*(p.loaded_keys for p in patches.values()))
    if consumed != set(sd):
        raise ValueError(f"Unconsumed native loader keys: {sorted(set(sd)-consumed)}")
    return patches


def audit(run, comfy):
    output = run / "export_check.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    root = Path(__file__).resolve().parents[1]
    for name, sha in manifest["source_sha256"].items():
        if digest(root / name) != sha:
            raise ValueError("Production source differs from exported run")
    for source in manifest["adapters"]:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError("Input adapter identity changed")
        if "alpha" in source["metadata"]:
            raise ValueError("File-level alpha requires separate native-loader policy validation")
    exported_path = Path(manifest["export"])
    if exported_path.stat().st_size != manifest["export_size"]:
        raise ValueError("Export size changed")
    meta = header(exported_path)["__metadata__"]
    mode = effective_mode(meta)
    cfg = json.loads(meta.get("merge_experimental", "null"))
    if mode not in ("weighted_sum", "slerp", "per_prefix", "np_lora", "ct_merge"):
        raise ValueError("This reference supports additive/SLERP/plain recorded per-prefix/NP/CT only")
    if (mode in ("weighted_sum", "slerp", "per_prefix")) != (cfg is None):
        raise ValueError("Export mode/config mismatch")
    decisions, model_scale = {}, 1.
    if mode == "per_prefix":
        if digest(manifest["tuner_data"]) != manifest["tuner_data_sha256"]:
            raise ValueError("Tuner evidence changed")
        decisions, model_scale = replay_policy(manifest, meta)

    sys.path.insert(0, str(comfy))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import comfy.lora
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized GPU environment for the full-size check")
    spec = importlib.util.spec_from_file_location("h3_native_reference", root / "experimental_merge.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    stable_node = None
    if mode in ("slerp", "per_prefix"):
        spec = importlib.util.spec_from_file_location("h3_slerp_reference", root / "lora_optimizer.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        stable_node = module.LoRAOptimizer()
    base = header(manifest["checkpoint"])
    state = {"diffusion_model." + k: torch.empty(v["shape"], device="meta")
             for k, v in base.items() if k != "__metadata__" and k.endswith(".weight")}
    shape_model = types.SimpleNamespace(state_dict=lambda: state, model_config=types.SimpleNamespace(unet_config={}))
    mapping = comfy.lora.model_lora_keys_unet(shape_model, {})
    sources = [load_native(a["path"], mapping, torch, comfy.lora) for a in manifest["adapters"]]
    exported = load_native(exported_path, mapping, torch, comfy.lora)
    require_coverage(sources, exported)
    if decisions and set(decisions) != {
            component_label(target, c) for target in exported
            for c, _ in enumerate(component_rows(target, state[target].shape))}:
        raise ValueError("Recorded decisions do not cover exactly the native component union")
    results = []
    started = time.monotonic()
    for target in sorted(exported):
        if not isinstance(target, str) or target not in state:
            raise ValueError("Unexpected non-native or sliced target")
        for component, rows in enumerate(component_rows(target, state[target].shape)):
            local_mode = decisions[component_label(target, component)] if decisions else mode
            up, down = factor_pair(exported[target], rows, "cuda")
            actual = up @ down
            if actual.shape != (rows.stop - rows.start, state[target].shape[1]):
                raise ValueError("Export factor output differs from checkpoint target shape")
            diffs, factors, indices = [], {}, []
            for i, (source, weight) in enumerate(zip(sources, manifest["strengths"])):
                if target not in source:
                    continue
                b, a = factor_pair(source[target], rows, "cuda")
                diffs.append((b @ a, weight * model_scale))
                factors[i] = (b * weight, a)
                indices.append(i)
            additive = sum(d * w for d, w in diffs)
            if stable_node is not None:
                # Native mapping is independent; the pinned normal SLERP
                # formula is shared with production, not a second algorithm.
                expected = stable_node._merge_diffs(list(diffs), local_mode,
                    compute_device=torch.device("cuda"), keep_on_gpu=True)
                storage_dtype = None
                for i in indices:
                    weights = sources[i][target].weights
                    item_dtype = torch.promote_types(weights[0].dtype, weights[1].dtype)
                    storage_dtype = item_dtype if storage_dtype is None else torch.promote_types(storage_dtype, item_dtype)
                rounded = expected.to(storage_dtype).float() if local_mode == "slerp" and len(indices) > 1 else expected
                if not torch.isfinite(rounded).all():
                    rounded = expected
            else:
                expected = additive if cfg is None else exp.merge(diffs, mode, cfg,
                    source_indices=indices, role_indices=[0, 1], svd_factors=factors)
                rounded = expected
            error = (actual - expected).norm().item()
            change = (expected - additive).norm().item()
            results.append(dict(prefix=target, component=component,
                local_mode=local_mode, model_scale=model_scale,
                source_indices=indices, norm=expected.norm().item(),
                relative_export_error=error / max(expected.norm().item(), 1e-30),
                relative_normal_storage_rounding_error=(rounded - expected).norm().item() / max(expected.norm().item(), 1e-30),
                relative_error_after_storage_rounding=(actual - rounded).norm().item() / max(rounded.norm().item(), 1e-30),
                relative_method_change=change / max(additive.norm().item(), 1e-30),
                error_to_method_change=error / change if change else None,
                finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())))
            del actual, expected, rounded, additive, diffs, factors, up, down, b, a
        print(f"Checked native {target}", flush=True)
    report = dict(mode=mode, export_metadata_mode=meta["merge_mode"], all_targets=True, native_loader="comfy.lora; no optimizer key normalization",
        groups_checked=len(results), native_targets=len(exported),
        source_native_targets=[len(s) for s in sources],
        all_source_and_export_keys_consumed=True,
        manifest_sha256=digest(manifest_path), export_sha256=digest(exported_path),
        checker_sha256=digest(__file__), comfy_lora_sha256=digest(comfy.lora.__file__),
        torch_version=torch.__version__, elapsed_seconds=time.monotonic()-started,
        reference=("Pinned ordinary stable formulas and captured replay policy; FP32 discrepancy includes normal rounding/compression, reported separately"
                   if stable_node is not None else "FP32 additive or pinned experimental formula"),
        results=results)
    save_new(output, report)
    print(json.dumps(dict(output=str(output), groups=len(results),
        max_relative_export_error=max(r["relative_export_error"] for r in results),
        finite=all(r["finite"] for r in results))))
    if not all(r["finite"] for r in results):
        raise ValueError("Non-finite export/reference; inspect preserved report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, default=Path("/media/p5/Comfyui"))
    args = parser.parse_args()
    audit(args.run, args.comfy)
