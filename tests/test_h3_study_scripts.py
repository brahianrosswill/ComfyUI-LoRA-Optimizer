"""Offline checks for the research harness; no server or model files required."""
import json
import struct
import hashlib
import shutil
import subprocess

import pytest

from scripts.h3_merge_study import header, check_gpu_headroom
from scripts.h3_render_study import graph, PROMPTS
from scripts.h3_study_record import collect_merge_run
from scripts.h3_benchmark import case_matrix, EXPORT_RUNS, successful
from scripts.h3_av_review import validate_labels, output_video, DIMENSIONS
from scripts import h3_av_review
from scripts.h3_study_record import collect_render_run


def test_local_profile_uses_requested_quantization_and_limited_resolution():
    result = graph(PROMPTS["cup"], 123, [], "study", profile="local")
    assert result["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert result["2"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert result["2"]["inputs"]["type"] == "minimax"
    assert (result["5"]["inputs"]["width"], result["5"]["inputs"]["height"]) == (640, 384)
    assert result["5"]["inputs"]["length"] % 17 == 5
    assert result["6"]["inputs"]["noise_seed"] == 123
    assert result["8"]["inputs"]["steps"] == 20
    assert result["13"]["inputs"]["audio"] == ["12", 0]
    assert not any("Lora" in n["class_type"] for n in result.values())


def test_adapter_chain_controls_both_schedule_and_guider_without_prompt_changes():
    result = graph(PROMPTS["boxing"], 456, [("a.safetensors", .8), ("b.safetensors", -.3)], "study", "local")
    assert result["20"]["inputs"]["model"] == ["1", 0]
    assert result["21"]["inputs"]["model"] == ["20", 0]
    assert result["21"]["inputs"]["strength_model"] == -.3
    assert result["8"]["inputs"]["model"] == result["9"]["inputs"]["model"] == ["21", 0]
    assert result["5"]["inputs"]["prompt"] == PROMPTS["boxing"]
    assert result["6"]["inputs"]["noise_seed"] == 456


def test_checkpoint_header_reader_does_not_require_tensor_payload(tmp_path):
    payload = json.dumps({"layer.weight": {"shape": [8, 16], "dtype": "I8", "data_offsets": [0, 128]}}).encode()
    path = tmp_path / "shape-only.safetensors"
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)
    assert header(path)["layer.weight"]["shape"] == [8, 16]


def test_checkpoint_header_rejects_unreasonable_allocation(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(ValueError, match="header size"):
        header(path)


def test_headless_study_refuses_retained_gpu_cache_before_large_allocations():
    with pytest.raises(RuntimeError, match="idle model caches"):
        check_gpu_headroom("tune", 2 * 1024**3)
    with pytest.raises(RuntimeError, match="6 GiB"):
        check_gpu_headroom("merge", 5 * 1024**3)
    check_gpu_headroom("tune", 24 * 1024**3)
    check_gpu_headroom("merge", 8 * 1024**3)


def test_durable_record_preserves_timestamp_precision_and_export_identity(tmp_path):
    export = tmp_path / "merged.safetensors"
    export.write_bytes(b"synthetic fixture")
    manifest = {"checkpoint_mtime_ns": 1788468997480069696, "adapters": [],
                "export": str(export), "export_size": export.stat().st_size}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    record = collect_merge_run(tmp_path)
    assert json.loads(json.dumps(record))["checkpoint_mtime_ns"] == 1788468997480069696
    assert len(record["export_sha256"]) == 64
    export.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Export size"):
        collect_merge_run(tmp_path)


def test_mapped_record_separates_rounding_from_export_error_and_rejects_stale_audit(tmp_path):
    export = tmp_path / "merged.safetensors"
    export.write_bytes(b"fixture")
    manifest = {"adapters": [], "mapped_export": True, "export": str(export), "export_size": 7}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    check_path = tmp_path / "dense_export_check.json"
    check = dict(groups_checked=1, all_targets=True,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
        results=[dict(finite=True, exact_stored_values=True, relative_export_error=0.,
                      relative_storage_rounding_error=.0016, relative_fp32_reference_error=.0016)])
    check_path.write_text(json.dumps(check))
    report = collect_merge_run(tmp_path)["dense_precision"]
    assert report["exact_stored_components"] == 1
    assert report["max_relative_export_error"] == 0.
    assert report["max_relative_storage_rounding_error"] == .0016
    export.write_bytes(b"changed")  # Same length: only a fresh hash catches this.
    with pytest.raises(ValueError, match="mismatched dense"):
        collect_merge_run(tmp_path)


def test_frozen_av_matrix_has_matched_controls_and_disjoint_holdout():
    exports = {pair: {mode: {"api_name": f"{pair}-{mode}.safetensors"} for mode in modes}
               for pair, modes in EXPORT_RUNS.items()}
    jobs, cases = case_matrix(exports)
    assert len(jobs) == 48 and len(cases) == 56
    assert (jobs, cases) == case_matrix(exports)
    assert len({j["id"] for j in jobs}) == 48
    assert all(j["seed"] == 2026090803 for j in jobs[:12])
    calibration = [j for j in jobs if j["stage"] == "calibration"]
    heldout = [j for j in jobs if j["stage"] == "heldout"]
    assert {j["prompt"] for j in calibration}.isdisjoint(j["prompt"] for j in heldout)
    assert {j["seed"] for j in calibration}.isdisjoint(j["seed"] for j in heldout)
    for stage, prompt, seed in {(c["stage"], c["prompt"], c["seed"]) for c in cases}:
        group = [c for c in cases if (c["stage"], c["prompt"], c["seed"]) == (stage, prompt, seed)]
        for pair in EXPORT_RUNS:
            assert {c["variant"] for c in group if c["pair"] == pair} == {
                "base", "combat", "second", "additive", "winner", "np", "ct"}
        for control in ("base", "combat"):
            assert len({c["job_id"] for c in group if c["variant"] == control}) == 1


def test_benchmark_never_treats_failed_history_as_completed(tmp_path):
    assert not successful(tmp_path)
    (tmp_path / "history.json").write_text(json.dumps({"status": {"status_str": "error"}}))
    with pytest.raises(RuntimeError, match="do not auto-resubmit"):
        successful(tmp_path)


def test_heldout_h3_prompt_is_structured_and_keeps_sound_observable():
    text = PROMPTS["padwork"]
    assert text.startswith("integrated_multimodal_description: [Shot 1]")
    assert text.count("[Shot") == 1
    assert text.index("overall_soundscape:") < text.index("non_diegetic_music:")
    assert "No speech" in text and text.endswith("non_diegetic_music: N/A")


def test_av_labels_require_explicit_full_review_and_exact_media_identity():
    key = {"review_id": "r", "plan_sha256": "p", "cases": [{"blind_id": "C01", "media_sha256": "m"}]}
    row = {"blind_id": "C01", "media_sha256": "m", "full_video_reviewed": True,
           "audio_listened": True, "scores": {d: 3 for d in DIMENSIONS}}
    labels = {"review_id": "r", "plan_sha256": "p", "reviewer": "test-only fixture", "ratings": [row]}
    assert len(validate_labels(labels, key)) == 1
    for field, invalid, error in (("audio_listened", False, "both"), ("full_video_reviewed", False, "both"),
                                   ("media_sha256", "wrong", "hash")):
        with pytest.raises(ValueError, match=error):
            validate_labels({**labels, "ratings": [{**row, field: invalid}]}, key)
    for value in (float("nan"), True, 5, -1, 3.5):
        with pytest.raises(ValueError, match="integer"):
            validate_labels({**labels, "ratings": [{**row, "scores": {**row["scores"], "audio": value}}]}, key)
    with pytest.raises(ValueError, match="repeated"):
        validate_labels({**labels, "ratings": [row, row]}, key)
    with pytest.raises(ValueError, match="another"):
        validate_labels({**labels, "plan_sha256": "other"}, key)


def test_review_rejects_output_path_escape(tmp_path):
    history = {"status": {"status_str": "success"}, "outputs": {"14": {"images": [
        {"type": "output", "subfolder": "..", "filename": "unexpected.mp4"}]}}}
    with pytest.raises(ValueError, match="path"):
        output_video(history, tmp_path)


def test_render_record_requires_matching_media_hash_and_full_av_profile(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic stream fixture, not a real render")
    monkeypatch.setattr(h3_av_review, "output_video", lambda history: video)
    metrics = {"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "decoded_frames": 124}
    probe = {"streams": [{"codec_type": "video", "width": 640, "height": 384, "r_frame_rate": "24/1"},
                         {"codec_type": "audio", "sample_rate": "32000", "channels": 2}]}
    audit = tmp_path / "media-audit"
    audit.mkdir()
    (audit / "metrics.json").write_text(json.dumps(metrics))
    (audit / "probe.json").write_text(json.dumps(probe))
    (tmp_path / "manifest.json").write_text('{}')
    (tmp_path / "prompt_api.json").write_text('{}')
    (tmp_path / "history.json").write_text(json.dumps({"status": {"messages": [
        ["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 3000}]]}}))
    assert collect_render_run(tmp_path)["elapsed_seconds"] == 2
    (audit / "metrics.json").write_text(json.dumps({**metrics, "decoded_frames": 123}))
    with pytest.raises(ValueError, match="profile"):
        collect_render_run(tmp_path)
    (audit / "metrics.json").write_text(json.dumps({**metrics, "video_sha256": "different"}))
    with pytest.raises(ValueError, match="different media"):
        collect_render_run(tmp_path)


def test_review_page_does_not_embed_private_method_mapping(tmp_path, monkeypatch):
    run = tmp_path / "SECRET_METHOD_JOB"
    run.mkdir()
    (run / "history.json").write_text('{}')
    video = tmp_path / "source.mp4"
    video.write_bytes(b"synthetic bytes for page construction only")
    monkeypatch.setattr(h3_av_review, "output_video", lambda history: video)
    def fake_remux(command, **kwargs):
        Path(command[-1]).write_bytes(video.read_bytes())
    from pathlib import Path
    real_run = subprocess.run
    monkeypatch.setattr(h3_av_review.subprocess, "run", fake_remux)
    plan = {"root": str(tmp_path), "cases": [{"stage": "calibration", "seed": 3, "pair": "combat_cinema",
             "prompt": "boxing", "variant": "SECRET_METHOD", "job_id": run.name}],
            "prompt_text": {"boxing": "A harmless synthetic test prompt."}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / "review"
    h3_av_review.build(plan_path, out, "calibration", 3)
    page = (out / "review.html").read_text()
    assert "SECRET_METHOD" not in page and str(video) not in page
    assert "SECRET_METHOD" in (out / "private-key.json").read_text()
    assert "data:video/mp4;base64," in page
    if shutil.which("node"):
        script = page.split('<script>', 1)[1].split('</script>', 1)[0]
        real_run(['node', '--check'], input=script, text=True, check=True, capture_output=True)
