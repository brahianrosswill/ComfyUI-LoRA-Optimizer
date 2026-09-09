"""Read-only numerical audit of a pilot export against uncompressed FP32 deltas.

This measures export/compression fidelity, not audiovisual quality. QKV is
checked component-by-component, matching optimizer normalization semantics.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--all-targets", action="store_true")
    args = parser.parse_args()
    output = args.run / "export_check.json"
    if output.exists():
        raise FileExistsError(output)
    import torch
    from safetensors import safe_open
    if not torch.cuda.is_available():
        raise RuntimeError("Run using the approved local GPU environment")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("experimental", root / "experimental_merge.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    manifest = json.loads((args.run / "manifest.json").read_text())
    sources = [safe_open(a["path"], framework="pt", device="cpu") for a in manifest["adapters"]]
    exported = safe_open(manifest["export"], framework="pt", device="cpu")
    mode = exported.metadata()["merge_mode"]
    cfg = json.loads(exported.metadata().get("merge_experimental", "null"))
    prefixes = sorted(k.removesuffix(".lora_down.weight") for k in exported.keys() if k.endswith(".lora_down.weight"))
    if not args.all_targets:
        prefixes = [p for p in prefixes if any(p.startswith("diffusion_model." + b + ".") for b in (
            "blocks.0", "blocks.25", "blocks.49", "token_refiner.blocks.0"))]
    results = []
    started = time.monotonic()
    for prefix in prefixes:
        up = exported.get_tensor(prefix + ".lora_up.weight").cuda().float()
        down = exported.get_tensor(prefix + ".lora_down.weight").cuda().float()
        scale = float(exported.get_tensor(prefix + ".alpha")) / down.shape[0]
        splits = 3 if prefix.endswith("qkv_proj") else 1
        if up.shape[0] % splits:
            raise ValueError("Unexpected QKV shape")
        for component in range(splits):
            rows = slice(component * up.shape[0] // splits, (component + 1) * up.shape[0] // splits)
            actual = (up[rows] @ down) * scale
            factors, diffs, ids = {}, [], []
            for index, (source, weight) in enumerate(zip(sources, manifest["strengths"])):
                if prefix + ".lora_A.weight" not in source.keys():
                    continue
                a = source.get_tensor(prefix + ".lora_A.weight").cuda().float()
                b = source.get_tensor(prefix + ".lora_B.weight")[rows].cuda().float()
                alpha_key = prefix + ".alpha"
                alpha = float(source.get_tensor(alpha_key)) if alpha_key in source.keys() else a.shape[0]
                b = b * (alpha / a.shape[0])
                diffs.append((b @ a, weight))
                factors[index] = (b * weight, a)
                ids.append(index)
            additive = sum(d * w for d, w in diffs)
            if cfg is None:
                if mode != "weighted_sum":
                    raise ValueError("This checker currently supports additive/NP/CT exports only")
                expected = additive
            else:
                expected = exp.merge(diffs, mode, cfg, source_indices=ids,
                                     role_indices=[0, 1], svd_factors=factors)
            norm = expected.norm().item()
            error = (actual - expected).norm().item()
            change = (expected - additive).norm().item()
            row = dict(prefix=prefix, component=component, norm=norm,
                       relative_export_error=error / max(norm, 1e-30),
                       relative_method_change=change / max(additive.norm().item(), 1e-30),
                       error_to_method_change=error / change if change else None,
                       finite=bool(torch.isfinite(actual).all()))
            results.append(row)
            del actual, additive, expected, diffs, factors, a, b
        del up, down
        print(f"Checked {prefix}", flush=True)
    report = dict(mode=mode, all_targets=args.all_targets, groups_checked=len(results),
                  elapsed_seconds=time.monotonic() - started, results=results)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "groups_checked": len(results),
        "max_relative_export_error": max(r["relative_export_error"] for r in results),
        "max_error_to_method_change": max((r["error_to_method_change"] or 0) for r in results)}))


if __name__ == "__main__":
    main()
