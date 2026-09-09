"""Preserve compact, hash-backed merge evidence; never overwrite an earlier record.

Keeps Python integer precision for nanosecond timestamps and excludes bulky
tensor-shape inventories (the original manifest is hashed and remains on disk).
No ComfyUI, GPU, network, model loading, or inference is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect_merge_run(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    record = {k: v for k, v in manifest.items() if k != "adapters"}
    record.update(run=run.name, artifact_directory=str(run),
                  manifest_sha256=digest(run / "manifest.json"))
    record["adapters"] = [{k: a[k] for k in ("path", "sha256", "model_name", "civitai_version_id", "tensor_count")
                           if k in a} for a in manifest.get("adapters", [])]
    if "export" in manifest:
        record["export_sha256"] = digest(manifest["export"])
        if Path(manifest["export"]).stat().st_size != manifest["export_size"]:
            raise ValueError("Export size differs from recorded manifest")
    if (run / "tuner_data.json").exists():
        tuner = json.loads((run / "tuner_data.json").read_text())
        record["tuner"] = {k: tuner[k] for k in ("algo_version", "analysis_summary", "top_n") if k in tuner}
        record["tuner_sha256"] = digest(run / "tuner_data.json")
    if (run / "export_check.json").exists():
        check = json.loads((run / "export_check.json").read_text())
        rows = check["results"]
        if not rows or check["groups_checked"] != len(rows):
            raise ValueError("Incomplete numerical check")
        record["precision"] = {k: v for k, v in check.items() if k != "results"}
        record["precision"].update(
            all_finite=all(r["finite"] for r in rows),
            max_relative_export_error=max(r["relative_export_error"] for r in rows),
            max_error_to_method_change=max((r["error_to_method_change"] or 0) for r in rows),
            worst_groups=sorted(rows, key=lambda r: r["relative_export_error"], reverse=True)[:3],
            source_sha256=digest(run / "export_check.json"))
        for field in ("relative_normal_storage_rounding_error", "relative_error_after_storage_rounding"):
            if all(field in r for r in rows):
                record["precision"]["max_" + field] = max(r[field] for r in rows)
    if manifest.get("mapped_export"):
        check_path = run / "dense_export_check.json"
        check = json.loads(check_path.read_text())
        rows = check["results"]
        if (not rows or check["groups_checked"] != len(rows) or not check["all_targets"]
                or check["manifest_sha256"] != record["manifest_sha256"]
                or check["export_sha256"] != record["export_sha256"]):
            raise ValueError("Incomplete or mismatched dense numerical audit")
        record["dense_precision"] = {k: v for k, v in check.items() if k != "results"}
        record["dense_precision"].update(
            all_finite=all(r["finite"] for r in rows),
            exact_stored_components=sum(r["exact_stored_values"] for r in rows),
            max_relative_export_error=max(r["relative_export_error"] for r in rows),
            max_relative_storage_rounding_error=max(r["relative_storage_rounding_error"] for r in rows),
            max_relative_fp32_reference_error=max(r["relative_fp32_reference_error"] for r in rows),
            worst_groups=sorted(rows, key=lambda r: r["relative_export_error"], reverse=True)[:3],
            source_sha256=digest(check_path))
        probe_path = run / "full_loader_probe.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text())
            if (probe["prior_audit_sha256"] != digest(check_path)
                    or probe["export_sha256_from_prior_audit"] != record["export_sha256"]):
                raise ValueError("Full loader probe belongs to a different audit/export")
            record["full_loader_probe"] = dict(probe, source_sha256=digest(probe_path))
    if (run / "merge.log").exists():
        log = (run / "merge.log").read_text()
        terms = ("Candidate #", "Pass 2:", "Model patches:", "SVD-compressed:",
                 "sparsification skipped", "Low-rank patches:")
        record["log_observations"] = [line for line in log.splitlines() if any(term in line for term in terms)]
        record["log_sha256"] = digest(run / "merge.log")
    return record


def collect_render_run(run):
    """A render record requires actual output plus a whole-clip decode audit."""
    try:
        from .h3_av_review import output_video
    except ImportError:
        from h3_av_review import output_video
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    history = json.loads((run / "history.json").read_text())
    video = output_video(history)
    metrics = json.loads((run / "media-audit" / "metrics.json").read_text())
    probe = json.loads((run / "media-audit" / "probe.json").read_text())
    video_hash = digest(video)
    if metrics["video_sha256"] != video_hash:
        raise ValueError("Metrics belong to different media")
    vs = [s for s in probe["streams"] if s["codec_type"] == "video"]
    aus = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    if len(vs) != 1 or len(aus) != 1:
        raise ValueError("Expected exactly one video and one audio stream")
    if (vs[0]["width"], vs[0]["height"], metrics["decoded_frames"], vs[0]["r_frame_rate"],
            int(aus[0]["sample_rate"]), aus[0]["channels"]) != (640, 384, 124, "24/1", 32000, 2):
        raise ValueError("Output does not match the fixed local AV profile")
    messages = history["status"]["messages"]
    times = {kind: data["timestamp"] for kind, data in messages
             if kind in ("execution_start", "execution_success")}
    return dict(run=run.name, manifest=manifest, video=str(video), video_sha256=video_hash,
                elapsed_seconds=(times["execution_success"] - times["execution_start"]) / 1000,
                prompt_api_sha256=digest(run / "prompt_api.json"),
                history_sha256=digest(run / "history.json"), metrics=metrics,
                streams=[{k: s[k] for k in ("codec_type", "codec_name", "width", "height", "r_frame_rate",
                                            "sample_rate", "channels", "duration", "nb_frames") if k in s}
                         for s in probe["streams"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, default=[])
    parser.add_argument("--render-run", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.run and not args.render_run:
        parser.error("At least one --run or --render-run is required")
    if args.out.exists():
        raise FileExistsError(args.out)
    data = {"study": "H3 autotuner correctness and native-factor follow-up",
            "scope": "Numerical/performance evidence only; no new audiovisual preference labels.",
            "merge_runs": [collect_merge_run(run) for run in args.run],
            "render_runs": [collect_render_run(run) for run in args.render_run]}
    with args.out.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"record": str(args.out), "runs": len(data["merge_runs"]), "renders": len(data["render_runs"])}))


if __name__ == "__main__":
    main()
