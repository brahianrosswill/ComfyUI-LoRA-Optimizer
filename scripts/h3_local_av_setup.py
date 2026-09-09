"""Download the approved official AV evaluator into isolated research storage.

No ComfyUI imports, environment changes, credentials, remote model code or pickle
weights. Download receipts are immutable; interrupted HF downloads can resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "docs/research/data/2026-09-09-h3-local-av-evaluator-policy.json"
ROOT = REPO / ".h3-study-artifacts/20260909/audio-evaluator"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def isolate(offline=False):
    os.environ["HF_HOME"] = str(ROOT / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "xdg-cache")
    os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "numba-cache")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def inventory():
    return {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}


def download():
    isolate()
    from huggingface_hub import HfApi, snapshot_download

    policy = read_json(POLICY)
    receipt_path = ROOT / "setup-receipt.json"
    if receipt_path.exists():
        raise FileExistsError("Completed receipt already exists; verify it, do not overwrite it")
    if shutil.disk_usage(ROOT).free < 45 * 1024**3:
        raise RuntimeError("At least 45 GiB free required before the bounded model download")
    info = HfApi(token=False).model_info(
        policy["model"], revision=policy["model_revision"], files_metadata=True
    )
    if info.sha != policy["model_revision"]:
        raise ValueError("Official model revision changed")
    files = []
    for item in info.siblings:
        name = item.rfilename
        if name.endswith((".json", ".safetensors")) or name in ("merges.txt", "README.md", "LICENSE"):
            if Path(name).name != name:
                raise ValueError("Only expected root-level model assets are accepted")
            files.append({"name": name, "bytes": item.size,
                          "lfs_sha256": item.lfs.sha256 if item.lfs else None})
    if not files or not any(x["name"].endswith(".safetensors") for x in files):
        raise ValueError("Official safetensors missing")
    intent = {"model": policy["model"], "revision": info.sha,
              "policy_sha256": digest(POLICY), "files": files}
    intent_path = ROOT / "download-intent.json"
    if intent_path.exists():
        if read_json(intent_path) != intent:
            raise ValueError("Cannot resume a different download intent")
    else:
        save_new(intent_path, intent)
    print(json.dumps({"event": "download_start", "utc": now(),
                      "bytes": sum(x["bytes"] for x in files), "files": len(files)}), flush=True)
    model_dir = Path(snapshot_download(
        policy["model"], revision=info.sha, token=False,
        local_dir=ROOT / "model", allow_patterns=[x["name"] for x in files], max_workers=2
    ))
    verified = []
    for item in files:
        path = model_dir / item["name"]
        sha = digest(path)
        stat = path.stat()
        if stat.st_size != item["bytes"] or (item["lfs_sha256"] and sha != item["lfs_sha256"]):
            raise ValueError(f"Model asset verification failed: {item['name']}")
        verified.append(dict(item, sha256=sha, mtime_ns=stat.st_mtime_ns))
        print(json.dumps({"event": "verified", "file": item["name"], "sha256": sha}), flush=True)
    receipt = {"completed_at_utc": now(), "model": policy["model"], "revision": info.sha,
               "model_dir": str(model_dir), "python": sys.version, "executable": sys.executable,
               "packages": inventory(), "files": verified,
               "policy_sha256": digest(POLICY), "helper_sha256": digest(__file__),
               "requirements_sha256": digest(REPO / "docs/research/h3-audio-evaluator-requirements.txt"),
               "remote_code": False, "pickle_weights": False, "media_uploaded": False,
               "model_validated_as_judge": False}
    save_new(receipt_path, receipt)
    print(json.dumps({"event": "download_complete", "receipt": str(receipt_path),
                      "receipt_sha256": digest(receipt_path)}), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    download()
