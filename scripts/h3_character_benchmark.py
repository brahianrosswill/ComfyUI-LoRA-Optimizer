"""Frozen nine-arm character/effect benchmark on the existing local H3 server.

Stage one covers the two protocol-specified diagonal pairs, not the whole
four-pair study. Prepare pins actual files; install adds same-disk hard links;
run submits serial jobs with durable receipts and an explicit split/seed.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

try:
    from .h3_benchmark import digest, save_new
    from .h3_render_study import graph, request
    from .h3_identity_study import completed
except ImportError:
    from h3_benchmark import digest, save_new
    from h3_render_study import graph, request
    from h3_identity_study import completed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/research/data"
ARTIFACTS = ROOT / ".h3-study-artifacts/20260908"
INSTALL = Path("/media/p5/model_temps/h3-autotuner-study-20260908")
BASE = "http://127.0.0.1:8189"
PAIRS = ("series30_cinema", "sully_combat")
ARMS = ("base", "character_only", "effect_only", "additive", "slerp", "ties", "stable_tuner_winner", "np_lora", "ct_merge")
SUFFIXES = dict(additive="additive-01", slerp="slerp-01", ties="ties-01-mapped",
                stable_tuner_winner="winner-01", np_lora="np-01", ct_merge="ct-01")
PROTOCOL_SHA = "b54c938768bc311b065479c6568700aa0b7ce303eadc37e0c8d9b3da535b9ecf"


def identity(path):
    st = Path(path).stat()
    return dict(device=st.st_dev, inode=st.st_ino, size=st.st_size, mtime_ns=st.st_mtime_ns)


def pin_asset(path, sha, api_name, strength, installed=None):
    path = Path(path).resolve()
    before = identity(path)
    if digest(path) != sha or identity(path) != before:
        raise ValueError(f"Asset identity changed: {path}")
    return dict(local_path=str(path), sha256=sha, identity=before, api_name=api_name,
                strength=strength, installed_path=str(installed) if installed else str(path))


def matrix(protocol, assets):
    jobs = []
    for stage, split in protocol["splits"].items():
        for seed in split["seeds"]:
            block = []
            for pair in PAIRS:
                character, effect = pair.split("_")
                prompt_key = f"{character}_{effect}_{stage}"
                for variant in ARMS:
                    keys = [] if variant == "base" else [
                        character if variant == "character_only" else effect if variant == "effect_only" else f"{pair}:{variant}"]
                    block.append(dict(id=f"idm-{pair.replace('_', '-')}-{seed}-{variant.replace('_', '-')}",
                        pair=pair, character=character, effect=effect, stage=stage,
                        seed=seed, prompt=prompt_key, variant=variant, asset_keys=keys,
                        loras=[[assets[k]["api_name"], assets[k]["strength"]] for k in keys]))
            random.Random(seed ^ 2026090830).shuffle(block)
            jobs.extend(block)
    return jobs


def expected_graph(plan, job):
    return graph(plan["prompt_text"][job["prompt"]], job["seed"], job["loras"],
                 f"h3_character_merge_20260908/{job['id']}", profile="local")


def validate_plan(plan, *, verify_files=True, installed=False):
    if (plan["base"], plan["root"], tuple(plan["pairs"]), tuple(plan["variants"])) != (
            BASE, str(ARTIFACTS), PAIRS, ARMS):
        raise ValueError("Execution scope differs from the frozen local stage")
    if digest(plan["protocol_path"]) != plan["protocol_sha256"] or plan["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Original protocol changed")
    protocol = json.loads(Path(plan["protocol_path"]).read_text())
    expected_prompts = {f"{c}_{e}_{stage}": protocol["templates"][f"{e}_{split['prompt_suffix']}"].format(**protocol["characters"][c])
        for pair in PAIRS for c, e in [pair.split("_")] for stage, split in protocol["splits"].items()}
    if plan["prompt_text"] != expected_prompts or plan["jobs"] != matrix(protocol, plan["assets"]):
        raise ValueError("Prompts or balanced matrix changed")
    if plan["profile"] != protocol["profile"] or len(plan["jobs"]) != 72:
        raise ValueError("Profile or case count changed")
    expected_keys = set(protocol["characters"]) | set(protocol["effects"]) | {
        f"{pair}:{variant}" for pair in PAIRS for variant in SUFFIXES}
    if set(plan["assets"]) != expected_keys:
        raise ValueError("Asset inventory changed")
    for name, spec in {**protocol["characters"], **protocol["effects"]}.items():
        asset = plan["assets"][name]
        if (asset["sha256"], asset["strength"]) != (spec["sha256"], spec["strength"]):
            raise ValueError("Raw adapter identity or control strength changed")
        if name in protocol["effects"] and asset["api_name"] != spec["api_name"]:
            raise ValueError("Effect loader path changed")
    for pair in PAIRS:
        for variant, suffix in SUFFIXES.items():
            key = f"{pair}:{variant}"
            asset = plan["assets"][key]
            filename = f"{pair.replace('_', '-')}-{suffix}.safetensors"
            if (asset["strength"] != 1. or asset["installed_path"] != str(INSTALL / filename)
                    or asset["api_name"] != f"h3-autotuner-study-20260908/{filename}"
                    or asset["sha256"] != plan["merge_evidence"][key]["numerical"]["export_sha256"]):
                raise ValueError("Merged adapter identity, loader path or scale changed")
    for filename, sha in plan["evidence_records"].items():
        if digest(filename) != sha:
            raise ValueError("Numerical evidence changed after freeze")
    for name, sha in plan["harness_sha256"].items():
        if Path(name).name != name or digest(ROOT / "scripts" / name) != sha:
            raise ValueError("Harness changed after freeze")
    if verify_files:
        # Full hashes were checked at freeze; inexpensive inode/size/mtime
        # guards avoid re-reading 157 GB before every short video.
        for asset in plan["assets"].values():
            source = Path(asset["local_path"])
            if identity(source) != asset["identity"]:
                raise ValueError(f"Asset changed after full hash: {source}")
            if installed and not os.path.samefile(source, asset["installed_path"]):
                raise ValueError("Installed path no longer points to verified data")


def prepare(path, numerical_records):
    if path.exists():
        raise FileExistsError(path)
    protocol_path = DATA / "2026-09-08-h3-character-merge-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    qualification = json.loads((DATA / "2026-09-08-h3-identity-qualification-plan.json").read_text())
    records, evidence = {}, {}
    for record_path in numerical_records:
        data = json.loads(record_path.read_text())
        evidence[str(record_path.resolve())] = digest(record_path)
        for run in data["merge_runs"]:
            if run["run"] in records:
                raise ValueError("Duplicate run evidence")
            records[run["run"]] = run
    assets, merges = {}, {}
    for character in protocol["characters"]:
        a = qualification["adapters"][character]
        assets[character] = pin_asset(a["local_path"], protocol["characters"][character]["sha256"],
                                      a["api_name"], protocol["characters"][character]["strength"])
    for effect, a in protocol["effects"].items():
        assets[effect] = pin_asset(Path("/media/p5/ComfyUI-Model-CIFS-Cache/loras") / a["api_name"],
                                   a["sha256"], a["api_name"], a["strength"])
    for pair in PAIRS:
        for variant, suffix in SUFFIXES.items():
            name = f"idm-{pair.replace('_', '-')}-{suffix}"
            record = records[name]
            check = record["dense_precision" if variant == "ties" else "precision"]
            if (not check["all_finite"] or not check["all_targets"] or check["groups_checked"] != 312
                    or check["native_targets"] != 208 or not check["all_source_and_export_keys_consumed"]
                    or check["export_sha256"] != record["export_sha256"]
                    or check["manifest_sha256"] != record["manifest_sha256"]):
                raise ValueError(f"Incomplete numerical qualification: {name}")
            if variant in ("additive", "np_lora", "ct_merge") and check["max_relative_export_error"] > .0005:
                raise ValueError("Protocol numerical gate failed")
            if variant == "ties" and check["max_relative_export_error"] > 1e-6:
                raise ValueError("Mapped TIES differs from ordinary stored patches")
            character, effect = pair.split("_")
            if [a["sha256"] for a in record["adapters"]] != [assets[character]["sha256"], assets[effect]["sha256"]]:
                raise ValueError("Merge source identities or role ordering differ")
            if record["strengths"] != [assets[character]["strength"], assets[effect]["strength"]]:
                raise ValueError("Merge strengths differ from controls")
            if variant == "stable_tuner_winner" and (record["action"] != "replay" or record["selected"]["rank"] != 1
                    or record["replay_per_prefix_decisions"] != record["selected"]["per_prefix_decisions"]):
                raise ValueError("Not an exact rank-one replay")
            destination = INSTALL / (name.removeprefix("idm-") + ".safetensors")
            key = f"{pair}:{variant}"
            print(f"Hashing qualified {key}", flush=True)
            assets[key] = pin_asset(record["export"], record["export_sha256"],
                f"h3-autotuner-study-20260908/{destination.name}", 1., destination)
            merges[key] = dict(run=name, manifest_sha256=record["manifest_sha256"],
                production_sha256=record["source_sha256"], numerical=check)
    prompts = {f"{c}_{e}_{stage}": protocol["templates"][f"{e}_{split['prompt_suffix']}"].format(**protocol["characters"][c])
        for pair in PAIRS for c, e in [pair.split("_")] for stage, split in protocol["splits"].items()}
    jobs = matrix(protocol, assets)
    plan = dict(version=1, created_at=time.time(), base=BASE, root=str(ARTIFACTS),
        pairs=list(PAIRS), deferred_pairs=["sully_cinema", "series30_combat"], variants=list(ARMS),
        profile=protocol["profile"], protocol_path=str(protocol_path), protocol_sha256=digest(protocol_path),
        evidence_records=evidence, assets=assets, merge_evidence=merges, prompt_text=prompts, jobs=jobs,
        cases=[dict(pair=j["pair"], variant=j["variant"], stage=j["stage"], prompt=j["prompt"], seed=j["seed"], job_id=j["id"]) for j in jobs],
        harness_sha256={name: digest(ROOT / "scripts" / name) for name in (
            "h3_character_benchmark.py", "h3_render_study.py", "h3_identity_study.py", "h3_benchmark.py")},
        scope="72 executable jobs for two diagonal pairs; first batch is only 18 calibration seed-31 clips. Other two pairs remain required by the full protocol.",
        numerical_review="Normal SLERP is an implementation baseline, not a lossless FP32 oracle; retain its documented BF16 rounding/compression. TIES preserves normal storage rounding. No tolerance is a perceptual quality threshold.",
        evaluation_policy=protocol["evaluation_policy"], quality_labels=[])
    validate_plan(plan)
    save_new(path, plan)
    print(json.dumps(dict(plan=str(path), sha256=digest(path), jobs=len(jobs))))


def install(plan):
    """Publish exact validated files without copying or replacing any model."""
    validate_plan(plan)
    if not INSTALL.is_dir():
        raise FileNotFoundError("Expected pre-existing temporary model directory")
    for key, asset in plan["assets"].items():
        source, destination = Path(asset["local_path"]), Path(asset["installed_path"])
        if source == destination:
            continue
        if destination.parent != INSTALL or destination.suffix != ".safetensors":
            raise ValueError("Installation escaped the explicit temporary directory")
        if destination.exists() or destination.is_symlink():
            if not os.path.samefile(source, destination):
                raise FileExistsError(f"Refusing to replace {destination}")
        else:
            if source.stat().st_dev != INSTALL.stat().st_dev:
                raise ValueError("Expected same-filesystem hard link, not a large copy")
            os.link(source, destination)
        print(json.dumps(dict(installed=key, path=str(destination), same_inode=True)), flush=True)
    validate_plan(plan, installed=True)


def prepare_job(plan, job, plan_sha):
    run = Path(plan["root"]) / job["id"]
    fields = dict(base=BASE, plan_sha256=plan_sha, job=job, prompt_name=job["prompt"],
        seed=job["seed"], loras=job["loras"], output_prefix=f"h3_character_merge_20260908/{job['id']}",
        turbo=False, steps=20, profile="local", width=640, height=384, length=124,
        quality_claim=None, stage=job["stage"], pair=job["pair"], variant=job["variant"])
    if not run.exists():
        run.mkdir(parents=True)
        save_new(run / "manifest.json", dict(fields, client_id="h3-idm-" + uuid.uuid4().hex))
        save_new(run / "prompt_api.json", expected_graph(plan, job))
        with (run / "prompt.txt").open("x") as stream:
            stream.write(plan["prompt_text"][job["prompt"]])
    manifest = json.loads((run / "manifest.json").read_text())
    if any(manifest[k] != value for k, value in fields.items()):
        raise ValueError("Prepared manifest differs from the frozen case")
    if (json.loads((run / "prompt_api.json").read_text()) != expected_graph(plan, job)
            or (run / "prompt.txt").read_text() != plan["prompt_text"][job["prompt"]]):
        raise ValueError("Prepared graph/prompt differs from the frozen case")
    return run, manifest


def run_batch(plan, plan_sha, stage, seed, limit):
    if not 1 <= limit <= 18:
        raise ValueError("Batch limit must be 1..18")
    validate_plan(plan, installed=True)
    jobs = [j for j in plan["jobs"] if (j["stage"], j["seed"]) == (stage, seed)]
    if len(jobs) != 18:
        raise ValueError("Explicit split/seed must select exactly 18 balanced cases")
    count = 0
    for job in jobs:
        run, manifest = prepare_job(plan, job, plan_sha)
        expected = expected_graph(plan, job)
        if completed(run, expected):
            continue
        validate_plan(plan, installed=True)
        if not (run / "submission.json").exists():
            if (run / "submission-intent.json").exists():
                raise RuntimeError(f"Ambiguous prior submission at {run}; reconcile the same client/graph with history, never resubmit blindly")
            queue = request(BASE, "/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                print(json.dumps(dict(paused="Existing queued work; nothing submitted", next_job=job["id"])), flush=True)
                return
            schema = request(BASE, "/object_info/LoraLoaderModelOnly")
            names = schema["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
            if any(name not in names for name, strength in job["loras"]):
                raise ValueError("Pinned adapter is not visible to the existing server")
            # Existing receipts are never overwritten; an incomplete prior
            # preparation stops for inspection rather than inventing history.
            save_new(run / "queue_before.json", {k: [j[1] for j in v] for k, v in queue.items()})
            save_new(run / "system_stats.json", request(BASE, "/system_stats"))
            submitted_at = time.time()
            save_new(run / "submission-intent.json", dict(client_id=manifest["client_id"],
                graph_sha256=digest(run / "prompt_api.json"), plan_sha256=plan_sha,
                submitted_at=submitted_at, runner_sha256=digest(__file__)))
            result = request(BASE, "/prompt", dict(prompt=expected, client_id=manifest["client_id"]))
            save_new(run / "submission.json", dict(**result, submitted_at=submitted_at))
        submission = json.loads((run / "submission.json").read_text())
        print(json.dumps(dict(watching=job["id"], prompt_id=submission["prompt_id"])), flush=True)
        with (run / "watch.log").open("a") as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name("h3_render_study.py")),
                "watch", "--out", str(run)], stdout=log, stderr=subprocess.STDOUT, check=True)
        if not completed(run, expected):
            raise RuntimeError("No successful terminal history; inspect the same prompt handle")
        count += 1
        print(json.dumps(dict(completed=job["id"], batch_completed=count)), flush=True)
        if count >= limit:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "run"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--numerical-record", action="append", default=[], type=Path)
    parser.add_argument("--stage", choices=("calibration", "heldout"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.plan, args.numerical_record)
    elif args.action == "install":
        install(json.loads(args.plan.read_text()))
    else:
        if args.stage is None or args.seed is None:
            parser.error("run requires an explicit --stage and --seed")
        run_batch(json.loads(args.plan.read_text()), digest(args.plan), args.stage, args.seed, args.limit)


if __name__ == "__main__":
    main()
