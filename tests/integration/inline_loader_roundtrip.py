"""Real stock LoRA loaders -> Inline -> save/reload, on tiny CPU models.

Run separately: python tests/integration/inline_loader_roundtrip.py /path/to/ComfyUI
No checkpoints, custom-node initialization, server or render jobs are needed.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.lora
import comfy.sd
import folder_paths
import nodes
from comfy.model_patcher import ModelPatcher
from safetensors.torch import save_file, load_file

spec = importlib.util.spec_from_file_location("inline_loader_integration", root / "lora_optimizer.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
m._LoRAMergeBase._get_compute_device = staticmethod(lambda: torch.device("cpu"))
torch.set_num_threads(2)
m._install_lora_name_stamp()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = types.SimpleNamespace(unet_config={})
        self.diffusion_model = torch.nn.Module()
        for name in ("a", "b", "c", "shared"):
            self.diffusion_model.add_module(name, torch.nn.Linear(16, 16, bias=False))


class TinyClip:
    def __init__(self, patcher=None):
        if patcher is None:
            module = torch.nn.Module()
            parent = module
            for name in "clip_l.transformer.text_model.encoder.layers.0.self_attn".split("."):
                child = torch.nn.Module()
                parent.add_module(name, child)
                parent = child
            parent.q_proj = torch.nn.Linear(16, 16, bias=True)
            patcher = ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))
        self.patcher = patcher
        self.cond_stage_model = patcher.model

    def clone(self):
        return TinyClip(self.patcher.clone())

    def add_patches(self, patches, strength):
        return self.patcher.add_patches(patches, strength)


def factors(prefix, seed):
    generator = torch.Generator().manual_seed(seed)
    return {prefix + ".lora_up.weight": torch.randn(16, 2, generator=generator) * .02,
            prefix + ".lora_down.weight": torch.randn(2, 16, generator=generator),
            prefix + ".alpha": torch.tensor(-3. if seed % 2 else 2.)}


def effective(patcher):
    return {key: comfy.lora.calculate_weight(patcher.patches.get(key, []), weight.clone(), key)
            for key, weight in patcher.model.state_dict().items()}


def equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=2e-5, msg=key)


def main():
    torch.manual_seed(2391)
    base = ModelPatcher(TinyModel(), torch.device("cpu"), torch.device("cpu"))
    clip = TinyClip()
    sources = {}
    for i, name in enumerate(("a", "b", "c")):
        sd = {**factors("diffusion_model." + name, 10 + i),
              **factors("diffusion_model.shared", 20 + i)}
        if name != "b":
            sd.update(factors("lora_te1_text_model_encoder_layers_0_self_attn_q_proj", 30 + i))
            sd["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.diff_b"] = torch.linspace(-.03, .04, 16) * (i + 1)
        sources[name] = sd

    with tempfile.TemporaryDirectory(prefix="inline-loader-") as directory:
        folder_paths.folder_names_and_paths["loras"] = ([directory], {".safetensors"})
        for name, sd in sources.items():
            save_file(sd, str(Path(directory) / (name + ".safetensors")))
        loader = nodes.LoraLoader()
        chain, chain_clip = loader.load_lora(base, clip, "a.safetensors", 1, 1)
        chain = nodes.LoraLoaderModelOnly().load_lora_model_only(chain, "b.safetensors", 1)[0]
        chain, chain_clip = loader.load_lora(chain, chain_clip, "c.safetensors", 1, 1)
        original_model, original_clip = effective(chain), effective(chain_clip.patcher)
        assert not base.patches and not clip.patcher.patches

        for mode in ("weighted_sum", "weighted_average", "slerp", "ties", "per_prefix"):
            for disable_middle in (False, True):
                settings = dict(m.LoRAOptimizerSimple._SIMPLE_DEFAULTS,
                    mode="advanced", auto_strength="disabled", architecture_preset="dit",
                    optimization_mode="per_prefix" if mode == "per_prefix" else "global",
                    merge_strategy_override="" if mode == "per_prefix" else mode,
                    patch_compression="disabled", svd_device="cpu", cache_patches="disabled")
                options = dict(visibility="simple", slots=[{}, {"enabled": not disable_middle}, {}])
                result = m.LoRAOptimizerInline().execute_inline(chain, 1., clip=chain_clip,
                                                              settings=settings, chain_options=options)
                result = result["result"] if isinstance(result, dict) else result
                stack = [dict(name=name + ".safetensors", lora=sd, strength=1.,
                              clip_strength=0. if name == "b" else 1.)
                         for name, sd in sources.items() if name != "b" or not disable_middle]
                reference = m.LoRAOptimizerSimple().execute_simple(base, stack, 1., clip=clip, settings=settings)
                reference = reference["result"] if isinstance(reference, dict) else reference
                equal(effective(result[0]), effective(reference[0]))
                equal(effective(result[1].patcher), effective(reference[1].patcher))
                if mode == "weighted_sum" and not disable_middle:
                    equal(effective(result[0]), original_model)
                    equal(effective(result[1].patcher), original_clip)
                equal(effective(chain), original_model)
                equal(effective(chain_clip.patcher), original_clip)
                assert not result[0].get_attachment(m.LORAOPT_CHAIN_RECORDS_ATTACH)
                assert not result[1].patcher.get_attachment(m.LORAOPT_CHAIN_RECORDS_ATTACH)
                path = m.SaveMergedLoRA().save_lora(result[4], directory, f"{mode}-{disable_middle}")[0]
                reloaded_model, reloaded_clip = comfy.sd.load_lora_for_models(base, clip, load_file(path), 1., 1.)
                equal(effective(reloaded_model), effective(result[0]))
                equal(effective(reloaded_clip.patcher), effective(result[1].patcher))
                print(f"PASS {mode}, disable_middle={disable_middle}: stock loaders, file-stack parity, CLIP, export/reload, upstream intact")


if __name__ == "__main__":
    main()
    # Break patcher reference cycles while ComfyUI globals still exist.
    import gc
    gc.collect()
