"""Audit completed frozen H3 jobs once, without queue polling or quality labels."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

try:
    from .h3_av_review import output_video
    from .h3_benchmark import digest, save_new
    from .h3_study_record import collect_render_run
except ImportError:
    from h3_av_review import output_video
    from h3_benchmark import digest, save_new
    from h3_study_record import collect_render_run


def audit(plan_path, stage, seed, out=None):
    plan = json.loads(plan_path.read_text())
    if out is not None and out.exists():
        raise FileExistsError(out)
    records, missing = [], []
    for job in plan["jobs"]:
        if job["stage"] != stage or job["seed"] != seed:
            continue
        run = Path(plan["root"]) / job["id"]
        history_path = run / "history.json"
        if not history_path.exists():
            missing.append(job["id"])
            continue
        history = json.loads(history_path.read_text())
        video = output_video(history)
        audit_dir = run / "media-audit"
        if not audit_dir.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("h3_video_metrics.py")),
                            "--video", str(video), "--out", str(audit_dir)], check=True,
                           stdout=subprocess.DEVNULL)
        # Existing incomplete/changed audits are rejected, not overwritten.
        record = collect_render_run(run)
        recorded_graph = history["prompt"][2]
        prepared_graph = json.loads((run / "prompt_api.json").read_text())
        if recorded_graph != prepared_graph:
            raise ValueError(f"Executed graph differs from prepared graph: {run}")
        records.append(record)
        print(json.dumps(dict(audited=job["id"], frames=record["metrics"]["decoded_frames"],
                              audio_clipped_fraction=record["metrics"]["audio"]["clipped_fraction"])), flush=True)
    if not records and not missing:
        raise ValueError("No matching frozen cases")
    record = dict(plan_sha256=digest(plan_path), stage=stage, seed=seed, renders=records,
                  incomplete_jobs=missing, quality_labels=[],
                  note="Whole-clip technical diagnostics only; no full-motion or listening judgment.")
    if out is not None:
        save_new(out, record)
    return dict(audited=len(records), incomplete=len(missing), record=str(out) if out else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "heldout"), default="calibration")
    parser.add_argument("--seed", type=int, default=2026090803)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.plan, args.stage, args.seed, args.out)))


if __name__ == "__main__":
    main()
