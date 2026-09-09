"""Real local ComfyUI's full-file loader probe, CPU only, no generation.

Usage: python tests/integration/h3_full_mapped_loader.py COMFYUI RUN_DIRECTORY
Requires the completed full numerical audit. Reuses its recorded payload hash;
checks stable inode/size/mtime during this read-only probe, not a second hash.
"""
import json
from pathlib import Path
import resource
import sys
import time
import types

root = Path(__file__).resolve().parents[2]
comfy_path, run = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(comfy_path))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy_aimdo.control
# Load native mmap symbols before Comfy imports capture the library handle.
# No init_device/init_devices: this probe does not allocate a GPU context.
if not comfy_aimdo.control.init():
    raise RuntimeError("Cannot load Aimdo's native mmap library")
import comfy.lora
import comfy.utils
from scripts.h3_benchmark import digest, save_new
from scripts.h3_merge_study import header
from scripts.h3_dense_export_check import export_groups

output = run / "full_loader_probe.json"
if output.exists():
    raise FileExistsError(output)
manifest = json.loads((run / "manifest.json").read_text())
audit_path = run / "dense_export_check.json"
audit = json.loads(audit_path.read_text())
if not audit["all_targets"] or not all(r["finite"] for r in audit["results"]):
    raise ValueError("Full numerical audit is not successful")
if audit["manifest_sha256"] != digest(run / "manifest.json"):
    raise ValueError("Manifest differs from numerical audit")
path = Path(manifest["export"]).resolve()
before = path.stat()
if before.st_size != manifest["export_size"]:
    raise ValueError("Export size changed")
torch.set_num_threads(2)
base = header(manifest["checkpoint"])
state = {"diffusion_model." + k: torch.empty(v["shape"], device="meta")
         for k, v in base.items() if k != "__metadata__" and k.endswith(".weight")}
model = types.SimpleNamespace(state_dict=lambda: state, model_config=types.SimpleNamespace(unet_config={}))
targets = {r["prefix"] for r in audit["results"]}
expected_keys = set().union(*(set(v) for v in export_groups(header(path), targets).values()))
started = time.monotonic()
# This is the branch used by the existing server's enabled DynamicVRAM/Aimdo.
# Call it explicitly: --cpu in this isolated probe does not enable Aimdo GPU.
sd, metadata = comfy.utils.load_safetensors(str(path))
assert set(sd) == expected_keys
assert all(hasattr(t.untyped_storage(), "_comfy_tensor_mmap_refs") and
           hasattr(t.untyped_storage(), "_comfy_tensor_file_slice") for t in sd.values())
patches = comfy.lora.load_lora(sd, comfy.lora.model_lora_keys_unet(model, {}))
assert set(patches) == targets
for target, patch in patches.items():
    actual_shape = patch[1][0].shape if isinstance(patch, tuple) else (patch.weights[0].shape[0], patch.weights[1].shape[1])
    assert tuple(actual_shape) == tuple(state[target].shape)
after = path.stat()
assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
record = dict(purpose="Full native mapped loader only; no INT8 forward pass, video or quality claim",
    export=str(path), export_size=before.st_size, export_sha256_from_prior_audit=audit["export_sha256"],
    full_payload_rehashed=False, stable_file_identity_during_probe=True,
    prior_audit_sha256=digest(audit_path), probe_sha256=digest(__file__),
    comfy_utils_sha256=digest(comfy.utils.__file__), comfy_lora_sha256=digest(comfy.lora.__file__),
    loader="comfy.utils.load_safetensors (Aimdo branch)", tensor_count=len(sd), native_targets=len(patches),
    aimdo_gpu_device_initialized=False,
    all_tensors_retain_mmap_and_file_slice_references=True,
    elapsed_seconds=time.monotonic()-started,
    max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save_new(output, record)
print(json.dumps(record))
