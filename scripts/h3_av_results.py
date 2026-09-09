"""Import one reviewer's AV preferences without training or changing the tuner.

Checks frozen case and media identities, keeps calibration/held-out separate,
preserves repeated-control disagreement, and reports matched ordinal ratings.
No significance, population-quality, or automatic shipping claims are made.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

try:
    from .h3_av_review import DIMENSIONS, output_video, validate_labels
    from .h3_benchmark import digest, save_new
except ImportError:
    from h3_av_review import DIMENSIONS, output_video, validate_labels
    from h3_benchmark import digest, save_new

CASE_FIELDS = ("pair", "variant", "stage", "prompt", "seed", "job_id")
GROUP_FIELDS = ("pair", "stage", "prompt", "seed")


def summarize(rows, cases, stage):
    if not rows:
        raise ValueError("No human ratings; do not substitute technical metrics")
    if any(row["stage"] != stage for row in rows):
        raise ValueError("Do not mix calibration and held-out ratings")
    expected = defaultdict(set)
    for case in cases:
        if case["stage"] == stage:
            expected[tuple(case[k] for k in GROUP_FIELDS)].add(case["variant"])
    grouped, by_media = defaultdict(dict), defaultdict(list)
    for row in rows:
        group = tuple(row[k] for k in GROUP_FIELDS)
        variant = row["variant"]
        if group not in expected or variant not in expected[group]:
            raise ValueError("Rating is outside the frozen matrix")
        if variant in grouped[group]:
            raise ValueError("Duplicate rating for one reviewer/case; use the intended export once")
        grouped[group][variant] = row
        by_media[row["original_sha256"]].append(row)
    groups, comparisons = [], []
    for group in sorted(expected):
        found = grouped[group]
        identity = dict(zip(GROUP_FIELDS, group))
        missing = sorted(expected[group] - found.keys())
        best = max((row["scores"]["overall"] for row in found.values()), default=None)
        groups.append({**identity, "missing_variants": missing,
                       "best_observed_overall": best,
                       "best_observed_variants": sorted(v for v, row in found.items() if row["scores"]["overall"] == best)})
        for candidate in ("np", "ct"):
            if candidate not in found:
                continue
            for baseline in ("additive", "winner", "combat", "second", "base"):
                if baseline not in found:
                    continue
                a, b = found[candidate]["scores"], found[baseline]["scores"]
                delta = {d: a[d] - b[d] for d in DIMENSIONS}
                comparisons.append({**identity, "candidate": candidate, "baseline": baseline,
                                    "ordinal_grade_differences": delta,
                                    "overall_preference": "candidate" if delta["overall"] > 0 else
                                                          "baseline" if delta["overall"] < 0 else "tie"})
    repeated = []
    for sha, entries in by_media.items():
        if len(entries) > 1:
            repeated.append(dict(original_sha256=sha,
                presentations=[{k: row[k] for k in ("review_id", "blind_id", "pair", "variant")} for row in entries],
                grade_spread={d: max(row["scores"][d] for row in entries) - min(row["scores"][d] for row in entries)
                              for d in DIMENSIONS}))
    return dict(rating_entries=len(rows), unique_videos=len(by_media), groups=groups,
                stage_fully_reviewed=all(not group["missing_variants"] for group in groups),
                paired_comparisons=comparisons, repeated_controls=repeated,
                inferential_statistics=None, learned_ranking=None,
                note="One person's context-dependent ordinal ratings. Reused controls and shared prompt/seeds are not independent trials.")


def collect(plan_path, reviews, stage):
    plan = json.loads(plan_path.read_text())
    plan_hash = digest(plan_path)
    allowed = {tuple(c[k] for k in CASE_FIELDS) for c in plan["cases"]}
    rows, provenance, reviewers = [], [], set()
    original_hashes = {}
    for key_path, labels_path in reviews:
        key, labels = json.loads(key_path.read_text()), json.loads(labels_path.read_text())
        if key["plan_sha256"] != plan_hash or key["stage"] != stage:
            raise ValueError("Wrong frozen plan or review stage")
        validated = validate_labels(labels, key)
        reviewers.add(labels["reviewer"].strip())
        media = {digest(p) for p in key_path.parent.glob("C*.mp4")}
        for row in validated:
            if tuple(row[k] for k in CASE_FIELDS) not in allowed:
                raise ValueError("Private review mapping differs from the frozen matrix")
            job_id = row["job_id"]
            if job_id not in original_hashes:
                history = json.loads((Path(plan["root"]) / job_id / "history.json").read_text())
                original_hashes[job_id] = digest(output_video(history))
            if row["original_sha256"] != original_hashes[job_id] or row["media_sha256"] not in media:
                raise ValueError("Original or review media identity mismatch")
            rows.append({**{k: v for k, v in row.items() if k != "original_path"},
                         "review_id": key["review_id"], "reviewer_id": "R1"})
        provenance.append(dict(review_id=key["review_id"], labels_sha256=digest(labels_path),
                               key_sha256=digest(key_path), submitted_at=labels.get("created_at")))
    if len(reviewers) != 1:
        raise ValueError("This importer expects one reviewer; do not silently pool people")
    return dict(plan_sha256=plan_hash, stage=stage, reviewer_count=1, reviewer_id="R1",
                provenance=provenance, ratings=rows, summary=summarize(rows, plan["cases"], stage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review", nargs=2, action="append", type=Path, required=True, metavar=("KEY", "LABELS"))
    parser.add_argument("--stage", choices=("calibration", "heldout"), default="calibration")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    record = collect(args.plan, args.review, args.stage)
    save_new(args.out, record)
    print(json.dumps(dict(record=str(args.out), summary=record["summary"])))


if __name__ == "__main__":
    main()
