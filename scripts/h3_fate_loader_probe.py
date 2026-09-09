"""One CPU-only probe of the upstream default loader; never runs this model."""
from __future__ import annotations

import contextlib
import json
import logging as standard_logging

try:
    from .h3_fate import import_source, verify_download
    from .h3_fate_setup import ROOT, digest, now, save_new
except ImportError:
    from h3_fate import import_source, verify_download
    from h3_fate_setup import ROOT, digest, now, save_new


def probe():
    import torch
    target = ROOT / "default-loader-probe-02.json"
    log = ROOT / "default-loader-probe-02.log"
    if target.exists() or log.exists():
        raise FileExistsError("Default-loader probe already attempted")
    torch.set_num_threads(4)
    verify_download()
    import_source()
    from models.pe_av import PeAudioVideoModel
    from transformers.utils import logging
    with log.open("x") as stream, contextlib.redirect_stderr(stream), contextlib.redirect_stdout(stream):
        handler = standard_logging.StreamHandler(stream)
        logging.disable_default_handler()
        logging.add_handler(handler)
        try:
            model, info = PeAudioVideoModel.from_pretrained(ROOT / "base", local_files_only=True,
                use_safetensors=True, output_loading_info=True)
            result = {"status": "returned_model", "loading_info": {k: sorted(v) if isinstance(v, set) else v for k,v in info.items()},
                      "model_tensor_count": len(model.state_dict())}
        except Exception as exc:
            result = {"status": "load_failed", "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            standard_logging.getLogger("transformers").removeHandler(handler)
            logging.enable_default_handler()
    result.update(utc=now(), script_sha256=digest(__file__), log_sha256=digest(log),
                  earlier_probe_log_sha256=digest(ROOT / "default-loader-probe.log"),
                  earlier_probe_runner_sha256=digest(ROOT / "default-loader-probe-01.py"),
                  earlier_failure="Transformers remove_handler assertion during cleanup; no model inference. Original log/runner preserved.",
                  model_forward=False, gpu_used=False, production_changed=False,
                  note="This model, if constructed, is discarded without evaluation. Strict mapped evaluator is separate.")
    save_new(target, result)
    print(json.dumps({"status": result["status"], "loading_info_counts": {k:len(v) for k,v in result.get("loading_info",{}).items()},
                      "receipt_sha256":digest(target)}))


if __name__ == "__main__":
    probe()
