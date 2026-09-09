"""One existing-session TIES loader/forward smoke test; not a quality comparison.

The old cup prompt/seed are reused. Preserve the character matrix and all its
held-out outputs. Refuse ambiguous submissions and never restart/cancel jobs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

try:
    from .h3_benchmark import digest, save_new
    from .h3_render_study import graph, request, PROMPTS
    from .h3_identity_study import completed
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_render_study import graph, request, PROMPTS
    from h3_identity_study import completed


def run_smoke(run):
    manifest = json.loads((run / "manifest.json").read_text())
    base = "http://127.0.0.1:8189"
    loras = [["h3-autotuner-study-20260908/series30-cinema-ties-01-mapped.safetensors", 1.]]
    # The unchanged preparation CLI stores its nargs values as strings.
    recorded_loras = [[name, float(strength)] for name, strength in manifest["loras"]]
    if (manifest["base"], manifest["profile"], manifest["prompt_name"], manifest["seed"], recorded_loras) != (
            base, "local", "cup", 2026090801, loras):
        raise ValueError("Prepared run differs from the pinned technical smoke test")
    expected = graph(PROMPTS["cup"], 2026090801, loras, manifest["output_prefix"], profile="local")
    if json.loads((run / "prompt_api.json").read_text()) != expected:
        raise ValueError("Prepared graph changed")
    if completed(run, expected):
        print("Already completed; nothing submitted")
        return
    if not (run / "submission.json").exists():
        if (run / "submission-intent.json").exists():
            raise RuntimeError("Ambiguous prior submission: reconcile client/graph with queue/history; never resubmit blindly")
        queue = request(base, "/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            print("Existing queued work: nothing submitted")
            return
        node = request(base, "/object_info/LoraLoaderModelOnly")
        available = node["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
        if loras[0][0] not in available:
            raise ValueError("Verified export is not visible in this server's loader")
        save_new(run / "queue_before.json", {k: [j[1] for j in v] for k, v in queue.items()})
        save_new(run / "system_stats.json", request(base, "/system_stats"))
        submitted_at = time.time()
        save_new(run / "submission-intent.json", dict(client_id=manifest["client_id"],
            graph_sha256=digest(run / "prompt_api.json"), submitted_at=submitted_at,
            runner_sha256=digest(__file__),
            purpose="Technical dense TIES loader/INT8 forward smoke; no identity ranking or held-out evidence"))
        result = request(base, "/prompt", dict(prompt=expected, client_id=manifest["client_id"]))
        save_new(run / "submission.json", dict(**result, submitted_at=submitted_at))
    print(json.dumps(json.loads((run / "submission.json").read_text())), flush=True)
    with (run / "watch.log").open("a") as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name("h3_render_study.py")),
                        "watch", "--out", str(run)], stdout=log, stderr=subprocess.STDOUT, check=True)
    if not completed(run, expected):
        raise RuntimeError("No successful terminal history; inspect same prompt handle")
    print("Completed dense TIES loader/forward smoke; no quality claim", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run_smoke(parser.parse_args().run)
