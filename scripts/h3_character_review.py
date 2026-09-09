"""Method-blinded character review with per-pair prompts and identity anchors.

Original video/audio samples are stream-copied, not normalized or retimed.
Character-only controls are explicitly available as anchors; their duplicate
blind candidates can therefore be recognizable. No ratings are prefilled.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

try:
    from .h3_av_review import PAGE, blinded_cases, output_video
    from .h3_character_benchmark import expected_graph
    from .h3_identity_study import completed
    from .h3_benchmark import digest, save_new
except ImportError:
    from h3_av_review import PAGE, blinded_cases, output_video
    from h3_character_benchmark import expected_graph
    from h3_identity_study import completed
    from h3_benchmark import digest, save_new

DIMENSIONS = ("identity", "effect", "temporal", "audio", "synchronization", "overall")
REFERENCE_HASHES = {
    "sully": "34786badbe615b71fd23db8e9651d8d89b78ee1a32a37e58f5ccff65d8cc5269",
    "series30": "65bf11aaa71df4c999f0a44c24458653e384818623ebfbae18abdf6fc305ccf7"}


def page(public, cards):
    result = PAGE.replace("__CARDS__", "\n".join(cards)).replace(
        "__DATA__", json.dumps(public).replace("<", "\\u003c"))
    result = result.replace("__PROMPT__", "See each pair's exact prompt below; the two pairs have different targets.")
    result = result.replace("Action: requested action order, contact, physical response and ending.",
        "Effect: the intended style or exact action, independently of character retention. Identity: head/face/feature fidelity to the creator and local character-only anchors, not matching clothes/background.")
    result = result.replace("Appearance: coherent subjects, texture and lighting (not brightness alone).", "")
    result = result.replace("Methods are hidden.",
        "Merge methods are hidden. Explicit character-only anchors are also among the blind candidates and may be recognizable. This is not a fully double-blind trial. Record visibility failures in the notes; never discard missing or occluded subjects.")
    result = result.replace("Specific faults, event timing, unwanted speech/music, or reasons for preference",
        "Note visibility failures (missing, occluded, out of frame, too small), visible identity details, action count/timing, and sound faults. Do not drop failures.")
    return result


def validate_labels(labels, key):
    """Character dimensions intentionally differ from the old AV-only rubric."""
    if (labels.get("review_id"), labels.get("plan_sha256")) != (key["review_id"], key["plan_sha256"]):
        raise ValueError("Ratings belong to a different review")
    if not isinstance(labels.get("reviewer"), str) or not labels["reviewer"].strip():
        raise ValueError("Reviewer identity required")
    by_id, seen, rows = {c["blind_id"]: c for c in key["cases"]}, set(), []
    for row in labels.get("ratings", []):
        cid = row.get("blind_id")
        if cid not in by_id or cid in seen:
            raise ValueError("Unknown or repeated clip")
        seen.add(cid)
        if row.get("media_sha256") != by_id[cid]["media_sha256"]:
            raise ValueError("Different review media")
        if row.get("full_video_reviewed") is not True or row.get("audio_listened") is not True:
            raise ValueError("Full motion and audio must both be reviewed")
        scores = row.get("scores", {})
        if set(scores) != set(DIMENSIONS) or any(type(v) is not int or not 0 <= v <= 4 for v in scores.values()):
            raise ValueError("All character-review dimensions require integer 0..4 scores")
        rows.append(dict(by_id[cid], scores=scores, notes=str(row.get("notes", ""))))
    if not rows:
        raise ValueError("No completed ratings")
    return rows


def build(plan_path, sources, out, stage, seed):
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads(plan_path.read_text())
    cases = blinded_cases(plan["cases"], stage, seed)
    jobs = {j["id"]: j for j in plan["jobs"]}
    if len(cases) != 18:
        raise ValueError("Expected the complete two-pair nine-arm block")
    originals = {}
    for case in cases:
        job = jobs[case["job_id"]]
        run = Path(plan["root"]) / job["id"]
        if not completed(run, expected_graph(plan, job)):
            raise ValueError("Missing completed frozen output")
        video = output_video(json.loads((run / "history.json").read_text()))
        audit = json.loads((run / "media-audit/metrics.json").read_text())
        if digest(video) != audit["video_sha256"] or audit["decoded_frames"] != 124:
            raise ValueError("Missing or mismatched full-clip technical audit")
        originals[job["id"]] = video
    for character, sha in REFERENCE_HASHES.items():
        if digest(sources / f"{character}-reference.mp4") != sha:
            raise ValueError("Creator reference changed")
    out.mkdir(parents=True)
    review_id = hashlib.sha256(f"{digest(plan_path)}|{time.time_ns()}".encode()).hexdigest()[:20]
    key = dict(review_id=review_id, plan_sha256=digest(plan_path), stage=stage, seed=seed,
        cases=cases, dimensions=list(DIMENSIONS), anchors=[], ratings=[],
        runner_sha256=digest(__file__), template_sha256=hashlib.sha256(PAGE.encode()).hexdigest())
    def media(original, name):
        destination = out / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(original), "-map", "0:v:0", "-map", "0:a:0",
            "-map_metadata", "-1", "-map_chapters", "-1", "-c", "copy", "-movflags", "+faststart",
            "-n", str(destination)], check=True)
        return digest(destination), "data:video/mp4;base64," + base64.b64encode(destination.read_bytes()).decode()
    embedded = {}
    for case in cases:
        original = originals[case["job_id"]]
        sha, data = media(original, case["blind_id"])
        embedded[case["job_id"]] = data
        case.update(media_sha256=sha, original_sha256=digest(original), original_path=str(original))
    cards = []
    for pair in plan["pairs"]:
        character, effect = pair.split("_")
        pair_cases = [c for c in cases if c["pair"] == pair]
        prompts = {c["prompt"] for c in pair_cases}
        if len(prompts) != 1:
            raise ValueError("Each review pair must have one target prompt")
        prompt = plan["prompt_text"][next(iter(prompts))]
        ref = sources / f"{character}-reference.mp4"
        sha, data = media(ref, f"reference-{character}")
        key["anchors"].append(dict(character=character, original_path=str(ref),
            original_sha256=digest(ref), media_sha256=sha))
        control = next(c for c in pair_cases if c["variant"] == "character_only")
        cards.append(f'<h2>{html.escape(pair.replace("_", " + "))}</h2>'
            f'<details><summary>Exact target prompt for this group</summary><pre>{html.escape(prompt)}</pre></details>'
            '<h3>Creator identity anchor — different workflow/timing, not pixel ground truth</h3>'
            f'<video controls muted preload="none" src="{data}"></video>'
            '<h3>Matched local character-only anchor — same prompt/seed, no effect adapter</h3>'
            f'<video controls muted preload="none" src="{embedded[control["job_id"]]}"></video>')
        for case in pair_cases:
            selects = "".join(f'<label>{dimension.capitalize()} <select data-dim="{dimension}"><option value="">Unrated</option>' +
                ''.join(f'<option value="{i}">{i}</option>' for i in range(5)) + '</select></label>' for dimension in DIMENSIONS)
            cards.append(f'<article data-id="{case["blind_id"]}"><h3>{case["blind_id"]}</h3>'
                f'<video controls preload="none" src="{embedded[case["job_id"]]}"></video><div class="scores">{selects}</div>'
                '<p><label><input type="checkbox" data-video> I reviewed the full motion</label> '
                '<label><input type="checkbox" data-audio> I listened to the full audio</label></p>'
                '<textarea placeholder="Specific faults, event timing, unwanted speech/music, or reasons for preference"></textarea></article>')
    public = dict(review_id=review_id, plan_sha256=key["plan_sha256"], stage=stage,
        cases=[{k: c[k] for k in ("blind_id", "pair", "media_sha256")} for c in cases])
    save_new(out / "private-key.json", key)
    save_new(out / "public.json", public)
    with (out / "review.html").open("x") as stream:
        stream.write(page(public, cards))
    print(json.dumps(dict(review=str(out / "review.html"), cases=len(cases), ratings=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    build(args.plan, args.sources, args.out, args.stage, args.seed)
