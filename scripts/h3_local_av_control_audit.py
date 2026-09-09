"""Audit calibration controls without treating valid JSON as AV quality evidence."""
import argparse
import hashlib

try:
    from .h3_local_av_evaluator import (ROOT, REPO, POLICY, read_json, digest, save_new, now,
        isolate, decode_audio, audio_control, validate_flat_response)
except ImportError:
    from h3_local_av_evaluator import (ROOT, REPO, POLICY, read_json, digest, save_new, now,
        isolate, decode_audio, audio_control, validate_flat_response)


def control_gates(records, best_case):
    by_key = {(r["case_id"], r["condition"], r["repeat"]): r for r in records}
    if len(by_key) != len(records):
        raise ValueError("Duplicated control record")
    cases = {r["case_id"] for r in records}
    expected = {(case, cond, 0) for case in cases for cond in ("original", "muted", "delay750")}
    expected |= {(best_case, cond, 1) for cond in ("original", "muted")}
    if len(cases) != 2 or set(by_key) != expected:
        raise ValueError("Full two-source control battery and repeats required")
    def valid(record):
        return record["status"] == "valid_response" and isinstance(record.get("response"), dict)
    def score(key):
        record = by_key[key]
        return record["response"]["synchronization_score"] if valid(record) else None
    muted = [by_key[(case, "muted", 0)] for case in cases]
    silence = all(valid(r) and all(r["response"][k] is False
                  for k in ("audible_sound", "speech_present", "music_present")) for r in muted)
    original_score, delayed_score = score((best_case, "original", 0)), score((best_case, "delay750", 0))
    sync = (type(original_score) is int and type(delayed_score) is int and delayed_score < original_score)
    categorical = ("audible_sound", "speech_present", "music_present")
    scoring = tuple(k + "_score" for k in ("action", "temporal", "appearance", "audio", "synchronization", "overall"))
    repeats = all(valid(by_key[(best_case, c, n)]) for c in ("original", "muted") for n in (0, 1))
    if repeats:
        repeats = all(by_key[(best_case, c, 0)]["response"][k] == by_key[(best_case, c, 1)]["response"][k]
                      for c in ("original", "muted") for k in categorical + scoring)
    return {"all_structured_responses_valid": all(valid(r) for r in records),
            "silence_boolean_gate": silence, "synchronization_direction_gate": sync,
            "best_source_original_sync_score": original_score,
            "best_source_delayed_sync_score": delayed_score,
            "categorical_and_score_repeatability_gate": repeats,
            "necessary_numeric_gates_pass": all(valid(r) for r in records) and silence and sync and repeats,
            "semantic_evidence_review_required": True,
            "qualified_for_quality_ranking": False}


def audit(directory):
    isolate(offline=True)
    directory = directory.resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Expected isolated evaluator run")
    start, end = read_json(directory / "start.json"), read_json(directory / "complete.json")
    manifest_path = ROOT / "calibration-inputs-flat-01.json"
    manifest, policy = read_json(manifest_path), read_json(POLICY)
    if start["profile"] != "joint_flat" or start["mode"] != "controls" or end["cases"] != 8:
        raise ValueError("Only the complete flat calibration control battery is supported")
    if start["manifest_sha256"] != digest(manifest_path) or start["policy_sha256"] != digest(POLICY):
        raise ValueError("Frozen input/policy mismatch")
    if start["setup_receipt_sha256"] != digest(ROOT / "setup-receipt.json"):
        raise ValueError("Model receipt changed")
    if start["helper_sha256"] != digest(REPO / "scripts/h3_local_av_evaluator.py"):
        raise ValueError("Freeze the runner source and explicitly audit that revision before proceeding")
    sources = {c["case_id"]: c for c in manifest["cases"] if c["source_job"] in policy["control_sources"]}
    best = next(k for k, v in sources.items() if v["source_job"].endswith("-combat"))
    waveforms = {}
    for key, case in sources.items():
        if digest(case["path"]) != case["sha256"]:
            raise ValueError("Frozen calibration source changed")
        waveforms[key], _ = decode_audio(case["path"])
    records, pins, video_hashes = [], {}, {}
    for job in start["jobs"]:
        ident = f"{job['case_id']}-{job['condition']}-r{job['repeat']}"
        path = directory / f"{ident}.json"
        pre = directory / f"{ident}.input.json"
        result = read_json(path)
        if any(result[k] != job[k] for k in ("case_id", "condition", "repeat")):
            raise ValueError("Job identity changed")
        case = sources[result["case_id"]]
        if result["source_sha256"] != case["sha256"]:
            raise ValueError("Source attribution mismatch")
        expected_audio = audio_control(waveforms[result["case_id"]], result["condition"])
        if hashlib.sha256(expected_audio.tobytes()).hexdigest() != result["inputs"]["waveform_sha256"]:
            raise ValueError("Actual model waveform differs from declared control")
        if read_json(pre)["inputs"] != result["inputs"]:
            raise ValueError("Pre-inference input evidence changed")
        video_hashes.setdefault(result["case_id"], set()).add(result["inputs"]["video_tensor_sha256"])
        if result["status"] == "valid_response" and validate_flat_response(result["raw_response"]) != result["response"]:
            raise ValueError("Parsed response differs from raw model output")
        if not result["generated_token_ids"] or result["generated_token_ids"][-1] != result["eos_token_id"]:
            raise ValueError("Response did not terminate at EOS")
        pins[str(path)] = digest(path)
        pins[str(pre)] = digest(pre)
        records.append(result)
    if any(len(hashes) != 1 for hashes in video_hashes.values()):
        raise ValueError("Video changed across audio controls")
    for path in (directory / "start.json", directory / "complete.json", manifest_path, POLICY,
                 REPO / "scripts/h3_local_av_evaluator.py", ROOT / "setup-receipt.json"):
        pins[str(path)] = digest(path)
    report = {"audited_at_utc": now(), "helper_sha256": digest(__file__), "control_cases": len(records),
              "best_case": best, "sources": sources, "input_integrity_pass": True,
              "gates": control_gates(records, best), "records": records, "artifact_sha256": pins,
              "heldout_evaluated": False, "human_ratings_created": False, "production_changed": False}
    save_new(directory / "control-audit.json", report)
    return report


if __name__ == "__main__":
    from pathlib import Path
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(audit(parser.parse_args().directory)["gates"], indent=2))
