"""Bounded, safetensors-only setup for an isolated FATE research evaluator.

No production imports, media uploads, credentials, pickle or training state.
Metadata and download receipts are new files, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import urllib.request

try:
    from .h3_local_av_setup import REPO, digest, inventory, now, read_json, save_new
except ImportError:
    from h3_local_av_setup import REPO, digest, inventory, now, read_json, save_new

ROOT = REPO / ".h3-study-artifacts/20260909/fate"
PREFLIGHT = REPO / "docs/research/data/2026-09-09-h3-fate-preflight.json"
PREFLIGHT_SHA = "8e8441bad172b55228750705bbb070693b871fccce3459cdafe3dc8e4dedc88a"
SOURCE_REV = "beae95aeb6f72cf1751d06d1428931016a7a1867"


def isolate(offline=True):
    for name, folder in (("HF_HOME", "hf-home"), ("XDG_CACHE_HOME", "xdg-cache"),
                         ("NUMBA_CACHE_DIR", "numba-cache")):
        os.environ[name] = str(ROOT / folder)
    for name in ("HF_HUB_DISABLE_IMPLICIT_TOKEN", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_XET"):
        os.environ[name] = "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def assets():
    if digest(PREFLIGHT) != PREFLIGHT_SHA:
        raise ValueError("Frozen preflight changed")
    return read_json(PREFLIGHT)["public_models"]


def metadata():
    target = ROOT / "metadata-receipt.json"
    if target.exists():
        raise FileExistsError(target)
    rows = []
    for entry in assets():
        folder = ROOT / ("base" if entry["repo"] == "facebook/pe-av-small" else "adapter")
        folder.mkdir(parents=True, exist_ok=True)
        for asset in entry["files"]:
            name = asset["path"]
            if not name.endswith((".json", ".safetensors")):
                continue
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{name}"
            is_header = name.endswith(".safetensors")
            path = folder / (name + ".header.json" if is_header else name)
            if path.exists():
                raise FileExistsError(path)
            req = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"} if is_header else {})
            with urllib.request.urlopen(req, timeout=45) as response:
                if is_header:
                    n = struct.unpack("<Q", response.read(8))[0]
                    if n < 2 or n > 1048500:
                        raise ValueError("Unexpected safetensors header size")
                    data = response.read(n)
                    if len(data) != n:
                        raise ValueError("Truncated header")
                else:
                    if asset["bytes"] > 4 * 1024**2:
                        raise ValueError("Metadata size exceeds bounded allowance")
                    data = response.read(asset["bytes"] + 1)
                    if len(data) != asset["bytes"]:
                        raise ValueError("Metadata file size changed")
            parsed = json.loads(data)
            with path.open("xb") as stream:
                stream.write(data)
            rows.append({"repo": entry["repo"], "revision": entry["revision"], "url": url,
                         "file": str(path.relative_to(REPO)), "sha256": digest(path),
                         "bytes": len(data), "header_only": is_header,
                         "tensor_count": len(parsed) - ("__metadata__" in parsed) if is_header else None})
            print(json.dumps(rows[-1]), flush=True)
    save_new(target, {"utc": now(), "files": rows, "script_sha256": digest(__file__),
                      "preflight_sha256": PREFLIGHT_SHA, "weights_downloaded": False})


def download():
    isolate(offline=False)
    from huggingface_hub import hf_hub_download
    target = ROOT / "download-receipt.json"
    if target.exists():
        raise FileExistsError(target)
    if shutil.disk_usage(ROOT).free < 15 * 1024**3:
        raise RuntimeError("Require at least 15 GiB free")
    metadata_receipt = read_json(ROOT / "metadata-receipt.json")
    for item in metadata_receipt["files"]:
        if digest(REPO / item["file"]) != item["sha256"]:
            raise ValueError("Previously inspected metadata changed")
    verified = []
    for entry in assets():
        folder = ROOT / ("base" if entry["repo"] == "facebook/pe-av-small" else "adapter")
        for item in entry["files"]:
            if not item["path"].endswith(".safetensors"):
                continue
            path = Path(hf_hub_download(entry["repo"], item["path"], revision=entry["revision"],
                                        local_dir=folder, token=False))
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("Downloaded weights failed size/hash verification")
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = stream.read(size)
            if header != (folder / (item["path"] + ".header.json")).read_bytes():
                raise ValueError("Weights header differs from inspected header")
            verified.append({"file": str(path.relative_to(REPO)), **item})
            print(json.dumps({"event": "verified", **verified[-1]}), flush=True)
    save_new(target, {"utc": now(), "files": verified, "packages": inventory(),
                      "python": sys.version, "script_sha256": digest(__file__),
                      "requirements_sha256": digest(REPO / "docs/research/h3-fate-requirements.txt"),
                      "metadata_receipt_sha256": digest(ROOT / "metadata-receipt.json"),
                      "source_revision": SOURCE_REV, "model_inference": False,
                      "pickle": False, "remote_code": False, "media_uploaded": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["metadata", "download"])
    {"metadata": metadata, "download": download}[parser.parse_args().action]()
