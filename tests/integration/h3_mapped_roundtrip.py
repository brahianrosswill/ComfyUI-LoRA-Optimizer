"""Real CPU ComfyUI: normal versus mapped TIES serialization, both dtypes.

Run separately with the ComfyUI path; no checkpoints or unit-test stubs.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.lora
import comfy.sd
from comfy.model_patcher import ModelPatcher
from safetensors.torch import load_file
from scripts.h3_mapped_export import MappedPatchStore, layout_from_native

spec = importlib.util.spec_from_file_location("mapped_test_optimizer", root / "lora_optimizer.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m._LoRAMergeBase._get_compute_device = staticmethod(lambda: torch.device("cpu"))
torch.set_num_threads(2)


class TinyH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = types.SimpleNamespace(unet_config={})
        self.lora_optimizer_h3_profile = {"partition": "fl2va", "basis": "pruned"}
        self.diffusion_model = torch.nn.Module()
        for path, shape in (("blocks.0.attn.qkv_proj", (768, 48)),
                            ("blocks.0.mlp.fc1", (96, 48)),
                            ("token_refiner.blocks.0.attn.qkv_proj", (768, 48))):
            parent = self.diffusion_model
            pieces = path.split(".")
            for name in pieces[:-1]:
                if name not in parent._modules:
                    parent.add_module(name, torch.nn.Module())
                parent = parent._modules[name]
            parent.add_module(pieces[-1], torch.nn.Linear(shape[1], shape[0], bias=False))


class TrackingOptimizer(m.LoRAOptimizer):
    def __init__(self):
        super().__init__()
        self.parameters_seen = set()
    def _merge_diffs(self, diffs_with_weights, mode, **kwargs):
        if mode == "ties":
            self.parameters_seen.add((kwargs["density"], kwargs["majority_sign_method"]))
        return super()._merge_diffs(diffs_with_weights, mode, **kwargs)


for dtype in (torch.float32, torch.bfloat16):
    torch.manual_seed(4812)
    base = ModelPatcher(TinyH3(), torch.device("cpu"), torch.device("cpu"))
    stack = []
    for i, (rank, alpha, strength) in enumerate(((2, -3., -.7), (3, 2., .4))):
        tensors = {}
        paths = [("blocks.0.attn.qkv_proj", 768), ("blocks.0.mlp.fc1", 96)]
        if i == 1:
            paths.append(("token_refiner.blocks.0.attn.qkv_proj", 768))
        for name, rows in paths:
            prefix = "lora_unet_" + name.replace(".", "_") if i == 0 else "diffusion_model." + name
            suffixes = (".lora_down.weight", ".lora_up.weight") if i == 0 else (".lora_A.weight", ".lora_B.weight")
            tensors[prefix + suffixes[0]] = torch.randn(rank, 48).to(dtype)
            tensors[prefix + suffixes[1]] = (torch.randn(rows, rank) * .02).to(dtype)
            tensors[prefix + ".alpha"] = torch.tensor(alpha)
        stack.append(dict(name=f"fixture-{i}", lora=tensors, strength=strength, h3_layout="comfy"))
    kwargs = dict(optimization_mode="global", merge_strategy_override="ties",
                  patch_compression="smart", normalize_keys="enabled",
                  cache_patches="disabled", vram_budget=0., _skip_model_apply=True)
    baseline = TrackingOptimizer()
    torch.manual_seed(808)
    ordinary = baseline.optimize_merge(base, stack, 1., **kwargs)
    mapping = comfy.lora.model_lora_keys_unet(base.model, {})
    native = [comfy.lora.load_lora(item["lora"], mapping) for item in stack]
    layout = layout_from_native(native, {k: v.shape for k, v in base.model.state_dict().items()})
    with tempfile.TemporaryDirectory() as directory:
        regular_path = m.SaveMergedLoRA().save_lora(ordinary[4], directory, "regular")[0]
        store = MappedPatchStore(Path(directory) / "mapped.safetensors", layout, disk_margin=0)
        class MappedOptimizer(TrackingOptimizer):
            def _new_patch_store(self, is_clip=False):
                return {} if is_clip else store
        mapped = MappedOptimizer()
        torch.manual_seed(808)
        streamed = mapped.optimize_merge(base, stack, 1., **kwargs, _skip_qkv_refusion=True)
        assert mapped.parameters_seen == baseline.parameters_seen
        assert streamed[0] is base and not base.patches
        mapped_path = store.finish({"merge_mode": "ties", "lora_optimizer_h3_layout": "comfy"})
        regular_sd, mapped_sd = load_file(regular_path), load_file(mapped_path)
        assert set(regular_sd) == set(mapped_sd)
        for key in regular_sd:
            assert regular_sd[key].dtype == mapped_sd[key].dtype, (dtype, key, "storage dtype")
            assert torch.equal(regular_sd[key], mapped_sd[key]), (dtype, key)
        reloaded, _ = comfy.sd.load_lora_for_models(base, None, mapped_sd, 1., 1.)
        assert set(reloaded.patches) == set(ordinary[4]["model_patches"])
        for key, patch in ordinary[4]["model_patches"].items():
            weight = base.model.state_dict()[key].float()
            expected = comfy.lora.calculate_weight([(1., patch, 1., None, None)], weight.clone(), key)
            actual = comfy.lora.calculate_weight(reloaded.patches[key], weight.clone(), key)
            assert torch.equal(actual, expected), (dtype, key)
        store.close()
    print(f"PASS {dtype}: identical tensor payloads and real ComfyUI application; shared dense QKV, mixed Musubi/native keys, unique factor QKV and signed alpha/strength")
    del ordinary, streamed, reloaded, base, mapped, baseline
