"""Research-only, lossless FP32 safetensors sink for native H3 TIES exports.

Only complete linear native targets, exact Q/K/V row slices and unique-source
plain LoRA factors are supported. Writes are bounded by one merged target.
Partial files are never published; existing destinations are never replaced.
Normal ComfyUI nodes do not enable this storage path.
"""
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import struct
import sys


def layout_from_native(sources, shapes):
    layout = {}
    for target in sorted(set().union(*(set(s) for s in sources))):
        if not isinstance(target, str) or not target.endswith(".weight"):
            raise ValueError("Expected native linear weight target")
        shape = tuple(shapes[target])
        if len(shape) != 2 or any(n <= 0 for n in shape):
            raise ValueError("Expected nonempty matrix")
        contributors = [s[target] for s in sources if target in s]
        rank = None
        if len(contributors) == 1:
            up, down, alpha, mid, dora, reshape = contributors[0].weights
            if mid is not None or dora is not None or reshape is not None:
                raise ValueError("Unique contributor is not a plain LoRA")
            if up.ndim != 2 or down.ndim != 2 or (up.shape[0], down.shape[1]) != shape or up.shape[1] != down.shape[0]:
                raise ValueError("Unique factor dimensions differ from target")
            rank = down.shape[0]
        layout[target] = dict(shape=list(shape), rank=rank)
    if not layout:
        raise ValueError("Empty export layout")
    return layout


class MappedPatchStore(dict):
    HEADER_BYTES = 128 * 1024

    def __init__(self, destination, layout, *, disk_margin=2 * 1024**3):
        import torch
        super().__init__()
        if sys.byteorder != "little":
            raise RuntimeError("Research writer requires little-endian storage")
        self.torch = torch
        self.destination = Path(destination)
        self.partial = self.destination.with_name(self.destination.name + ".partial")
        if self.destination.exists() or self.partial.exists():
            raise FileExistsError("Existing export/partial; inspect, never overwrite")
        self.layout, self.header, self.intervals, self.shared = layout, {}, {}, {}
        self.sealed = False
        offset = 0
        for target, item in sorted(layout.items()):
            rows, cols = item["shape"]
            prefix = target.removesuffix(".weight")
            entries = {prefix + ".diff": [rows, cols]} if item["rank"] is None else {
                prefix + ".lora_up.weight": [rows, item["rank"]],
                prefix + ".lora_down.weight": [item["rank"], cols], prefix + ".alpha": []}
            for name, shape in entries.items():
                size = math.prod(shape) * 4
                self.header[name] = dict(dtype="F32", shape=shape, data_offsets=[offset, offset + size])
                offset += size
        self.total_bytes = 8 + self.HEADER_BYTES + offset
        if shutil.disk_usage(self.destination.parent).free < self.total_bytes + disk_margin:
            raise RuntimeError("Insufficient disk for lossless dense export and safety margin")
        initial = self._header_bytes({"research_status": "incomplete; not published"})
        self.fd = os.open(self.partial, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(self.fd, self.total_bytes)
        os.pwrite(self.fd, struct.pack("<Q", self.HEADER_BYTES) + initial, 0)
        self.mapping = mmap.mmap(self.fd, self.total_bytes, access=mmap.ACCESS_WRITE)
        self.flat = torch.frombuffer(self.mapping, dtype=torch.float32)
        self.components = 0

    def _header_bytes(self, metadata):
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise ValueError("Safetensors metadata must be strings")
        payload = json.dumps({**self.header, "__metadata__": metadata}, separators=(",", ":"), allow_nan=False).encode()
        if len(payload) > self.HEADER_BYTES:
            raise ValueError("Reserved header exhausted")
        return payload + b" " * (self.HEADER_BYTES - len(payload))

    def tensor(self, name):
        item = self.header[name]
        start, end = item["data_offsets"]
        base = (8 + self.HEADER_BYTES) // 4
        return self.flat[base + start // 4:base + end // 4].view(item["shape"])

    def _flush_region(self, name):
        # Commit and release mapped pages, retaining tensors as file-backed
        # views. This avoids keeping every already-written matrix resident.
        start, end = self.header[name]["data_offsets"]
        start += 8 + self.HEADER_BYTES
        end += 8 + self.HEADER_BYTES
        aligned = start // mmap.PAGESIZE * mmap.PAGESIZE
        self.mapping.flush(aligned, end - aligned)
        if hasattr(self.mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            self.mapping.madvise(mmap.MADV_DONTNEED, aligned, end - aligned)

    def _write(self, name, value, rows=None):
        torch = self.torch
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Writer refuses lossy conversion from unsupported dtype")
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite result; partial export remains unpublished")
        dest = self.tensor(name)
        if rows is not None:
            dest = dest[rows]
        if dest.shape != value.shape:
            raise ValueError("Patch shape differs from reserved export region")
        dest.copy_(value.detach().to(device="cpu", dtype=torch.float32))
        self._flush_region(name)
        return dest

    def __setitem__(self, key, patch):
        torch = self.torch
        if self.sealed or key in self:
            raise ValueError("Sealed/duplicate patch; export store is append-only")
        target = key[0] if isinstance(key, tuple) else key
        if target not in self.layout:
            raise ValueError("Unplanned native target")
        item = self.layout[target]
        count = item["shape"][0]
        start, length = 0, count
        if isinstance(key, tuple):
            if len(key) != 2 or not target.endswith(".qkv_proj.weight"):
                raise ValueError("Only native QKV slices are supported")
            axis, start, length = key[1]
            if count % 3 or axis != 0 or length != count // 3 or start not in (0, length, 2 * length):
                raise ValueError("Invalid QKV row offset")
        end = start + length
        intervals = self.intervals.setdefault(target, [])
        if any(start < b and a < end for a, b in intervals):
            raise ValueError("Overlapping target rows")
        rows = slice(start, end)
        prefix = target.removesuffix(".weight")
        if item["rank"] is None:
            if not isinstance(patch, tuple) or patch[0] != "diff":
                raise ValueError("Expected uncompressed dense TIES patch")
            stored = ("diff", (self._write(prefix + ".diff", patch[1][0], rows),))
        else:
            up, down, alpha, mid, dora, reshape = patch.weights
            if mid is not None or dora is not None or reshape is not None:
                raise ValueError("Unsupported unique LoRA patch")
            alpha = float(down.shape[0]) if alpha is None else float(alpha)
            if not math.isfinite(alpha):
                raise ValueError("Non-finite alpha")
            if target in self.shared:
                old_down, old_alpha = self.shared[target]
                if old_alpha != alpha or not torch.equal(old_down, down.float().cpu()):
                    raise ValueError("QKV factor sharing differs; refusing incorrect fusion")
            else:
                self.shared[target] = (down.detach().float().cpu().clone(), alpha)
                self._write(prefix + ".lora_down.weight", down)
                self._write(prefix + ".alpha", torch.tensor(alpha, dtype=torch.float32))
            self._write(prefix + ".lora_up.weight", up, rows)
            stored = patch  # Only unique low-rank factors stay in RAM.
        intervals.append((start, end))
        dict.__setitem__(self, key, stored)
        self.components += 1

    def finish(self, metadata):
        if self.sealed:
            raise ValueError("Already published")
        for target, item in self.layout.items():
            intervals = sorted(self.intervals.get(target, []))
            if not intervals or intervals[0][0] != 0 or intervals[-1][1] != item["shape"][0] or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
                raise ValueError(f"Incomplete target coverage: {target}")
        payload = self._header_bytes(metadata)
        os.pwrite(self.fd, payload, 8)
        self.mapping.flush()
        os.fsync(self.fd)
        # Validate format with the real reader without materializing payloads.
        from safetensors import safe_open
        with safe_open(str(self.partial), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(self.header):
                raise ValueError("Reader rejected export tensor coverage")
        os.link(self.partial, self.destination)  # Atomic, fails if destination exists.
        self.partial.unlink()  # Same data remains in the published hard link.
        self.sealed = True
        return str(self.destination)

    def close(self):
        self.clear()
        self.shared.clear()
        self.flat = None
        try:
            self.mapping.close()
        except BufferError:
            pass  # Live returned tensor views retain their valid mmap owner.
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
