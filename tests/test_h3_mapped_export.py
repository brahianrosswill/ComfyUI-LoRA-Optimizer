"""Bounded writer safety; real-reader/optimizer parity is a separate integration."""
import json
import struct
import sys
import types

import pytest
import torch

from scripts.h3_mapped_export import MappedPatchStore


def test_normal_optimizer_storage_is_fresh_plain_dict_not_mapped():
    from tests.test_lora_optimizer import lora_optimizer
    node = lora_optimizer.LoRAOptimizer()
    model, clip = node._new_patch_store(), node._new_patch_store(is_clip=True)
    assert type(model) is dict and type(clip) is dict and model is not clip
    model["sentinel"] = 1
    assert not clip and not node._new_patch_store()


@pytest.fixture
def reader(monkeypatch):
    class Reader:
        def __init__(self, path, **kwargs):
            with open(path, "rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                self.header = json.loads(stream.read(size))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def keys(self):
            return set(self.header) - {"__metadata__"}
    # Other unit suites intentionally stub safetensors; do not imply this is
    # the actual format validation done by the integration subprocess.
    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace(safe_open=Reader))


def test_dense_qkv_writes_exact_order_and_promotes_bf16_losslessly(tmp_path, reader):
    target = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    store = MappedPatchStore(tmp_path / "out.safetensors", {target: dict(shape=[12, 8], rank=None)}, disk_margin=0)
    expected = torch.arange(96).reshape(12, 8).bfloat16()
    for start in (8, 0, 4):
        store[(target, (0, start, 4))] = ("diff", (expected[start:start + 4],))
    assert torch.equal(store.tensor(target.removesuffix(".weight") + ".diff"), expected.float())
    assert store.finish({"mode": "test"}) == str(tmp_path / "out.safetensors")
    assert not store.partial.exists()
    with pytest.raises(ValueError, match="Sealed"):
        store[target] = ("diff", (expected,))
    store.close()


def test_incomplete_duplicate_invalid_or_nonfinite_is_not_published(tmp_path, reader):
    target = "x.attn.qkv_proj.weight"
    store = MappedPatchStore(tmp_path / "out.safetensors", {target: dict(shape=[12, 8], rank=None)}, disk_margin=0)
    key = (target, (0, 0, 4))
    store[key] = ("diff", (torch.ones(4, 8),))
    with pytest.raises(ValueError, match="duplicate"):
        store[key] = ("diff", (torch.ones(4, 8),))
    with pytest.raises(ValueError, match="Incomplete"):
        store.finish({})
    with pytest.raises(ValueError, match="QKV"):
        store[(target, (0, 2, 4))] = ("diff", (torch.ones(4, 8),))
    with pytest.raises(ValueError, match="Non-finite"):
        store[(target, (0, 4, 4))] = ("diff", (torch.full((4, 8), float("inf")),))
    with pytest.raises(ValueError, match="lossy"):
        store[(target, (0, 4, 4))] = ("diff", (torch.ones(4, 8, dtype=torch.float64),))
    assert not store.destination.exists() and store.partial.exists()
    store.close()


def test_publish_never_replaces_existing_user_file(tmp_path, reader):
    target = "x.weight"
    dest = tmp_path / "out.safetensors"
    store = MappedPatchStore(dest, {target: dict(shape=[2, 2], rank=None)}, disk_margin=0)
    store[target] = ("diff", (torch.ones(2, 2),))
    dest.write_bytes(b"existing user content")
    with pytest.raises(FileExistsError):
        store.finish({})
    assert dest.read_bytes() == b"existing user content"
    assert store.partial.exists()
    store.close()


def test_unique_qkv_requires_shared_down_and_alpha(tmp_path, reader):
    target = "x.attn.qkv_proj.weight"
    store = MappedPatchStore(tmp_path / "out.safetensors", {target: dict(shape=[12, 8], rank=2)}, disk_margin=0)
    down = torch.arange(16).reshape(2, 8).float()
    for start in (0, 4, 8):
        patch = types.SimpleNamespace(weights=(torch.ones(4, 2) * start, down, -3., None, None, None))
        store[(target, (0, start, 4))] = patch
    assert store.tensor("x.attn.qkv_proj.alpha").item() == -3.
    store.finish({})
    store.close()


def test_mismatched_qkv_factor_sharing_stops(tmp_path, reader):
    target = "x.attn.qkv_proj.weight"
    store = MappedPatchStore(tmp_path / "out.safetensors", {target: dict(shape=[12, 8], rank=2)}, disk_margin=0)
    for start, alpha in ((0, 2.), (4, 3.)):
        patch = types.SimpleNamespace(weights=(torch.ones(4, 2), torch.ones(2, 8), alpha, None, None, None))
        if start == 0:
            store[(target, (0, start, 4))] = patch
        else:
            with pytest.raises(ValueError, match="sharing"):
                store[(target, (0, start, 4))] = patch
    store.close()
