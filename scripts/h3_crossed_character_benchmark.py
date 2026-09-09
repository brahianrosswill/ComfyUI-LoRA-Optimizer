"""Execute the remaining two protocol pairings without rewriting frozen studies.

Uses a private module instance of the hash-pinned nine-arm engine with only its
pair selection changed. The original imported engine and all old plan/helper
hashes remain unchanged. New plans pin this extension as well as that engine.
No new prompts, strengths, methods, seeds, servers or quality labels are added.
"""
import argparse
import importlib.util
import json
from pathlib import Path

try:
    from . import h3_character_benchmark as diagonal
    from .h3_benchmark import digest
except ImportError:
    import h3_character_benchmark as diagonal
    from h3_benchmark import digest

PAIRS = ("series30_combat", "sully_cinema")
ENGINE_SHA = "69f75f3de7a3ae679e103a85bdcce4d78f7a40507d8e62fba433ab589111d26f"
PARENT = diagonal.DATA / "2026-09-08-h3-character-benchmark-plan.json"
PARENT_SHA = "1f179b668839a4379a1743fd3c8f507b0202a254ccb5f47a3047f25a7a33ed40"
SELF = Path(__file__).resolve()
HELPERS = {"h3_character_benchmark.py", "h3_render_study.py",
           "h3_identity_study.py", "h3_benchmark.py", SELF.name}


def engine():
    if digest(diagonal.__file__) != ENGINE_SHA:
        raise ValueError("Frozen nine-arm engine changed")
    name = ((diagonal.__package__ + ".") if diagonal.__package__ else "") + "_h3_crossed_engine"
    spec = importlib.util.spec_from_file_location(name, diagonal.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PAIRS = PAIRS
    return module


def extension_fields(plan):
    """Applied before the first save, never as an edit to an existing plan."""
    plan.update(execution_phase="crossed_pairs", parent_plan_path=str(PARENT),
        parent_plan_sha256=PARENT_SHA, deferred_pairs=[],
        scope="72 additional jobs for Series30/Combat and Sully/Cinema. Together with the unchanged parent plan these specify all 144 protocol cases; this is not a completion claim.")
    plan["harness_sha256"][SELF.name] = digest(SELF)
    return plan


def validate_plan(plan, *, study=None, verify_files=True, installed=False):
    study = study or engine()
    if (plan.get("execution_phase") != "crossed_pairs"
            or plan.get("parent_plan_path") != str(PARENT)
            or plan.get("parent_plan_sha256") != PARENT_SHA
            or plan.get("deferred_pairs") != []
            or set(plan.get("harness_sha256", {})) != HELPERS
            or plan["harness_sha256"].get(SELF.name) != digest(SELF)):
        raise ValueError("Crossed-stage scope or implementation pin changed")
    if digest(PARENT) != PARENT_SHA:
        raise ValueError("Original diagonal plan changed")
    parent = json.loads(PARENT.read_text())
    diagonal.validate_plan(parent, verify_files=False)
    study.validate_plan(plan, verify_files=verify_files, installed=installed)
    expected_cases = [dict(pair=j["pair"], variant=j["variant"], stage=j["stage"],
        prompt=j["prompt"], seed=j["seed"], job_id=j["id"]) for j in plan["jobs"]]
    if plan.get("cases") != expected_cases:
        raise ValueError("Review cases differ from executable crossed jobs")
    if (plan["protocol_sha256"] != parent["protocol_sha256"]
            or plan["profile"] != parent["profile"]
            or plan["variants"] != parent["variants"]):
        raise ValueError("Crossed stage differs from the original protocol")
    for key in ("sully", "series30", "cinema", "combat"):
        if plan["assets"][key] != parent["assets"][key]:
            raise ValueError("Raw control identity or loading differs from parent")
    parent_ids = {j["id"] for j in parent["jobs"]}
    new_ids = {j["id"] for j in plan["jobs"]}
    if parent_ids & new_ids or len(parent_ids | new_ids) != 144:
        raise ValueError("The two execution phases must be disjoint and complete")
    if set(parent["pairs"]) | set(plan["pairs"]) != {
            "series30_cinema", "series30_combat", "sully_cinema", "sully_combat"}:
        raise ValueError("Full two-character/two-effect crossing is incomplete")


def prepare(path, numerical_records):
    if path.exists():
        raise FileExistsError(path)
    study = engine()
    save_original = study.save_new

    def save_extended(destination, value):
        if destination != path:
            raise ValueError("Unexpected plan destination")
        value = extension_fields(value)
        validate_plan(value, study=study)
        save_original(destination, value)

    # A private instance confines the pre-save hook and pair selection. Engine
    # preparation still checks all numerical evidence and hashes every asset.
    study.save_new = save_extended
    study.prepare(path, numerical_records)


def execute(plan_path, action, stage=None, seed=None, limit=18):
    study = engine()
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, study=study, installed=action == "run")
    if action == "install":
        study.install(plan)
    elif action == "run":
        study.run_batch(plan, digest(plan_path), stage, seed, limit)
    else:
        raise ValueError("Expected install or run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "install", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, action="append", default=[])
    parser.add_argument("--stage", choices=("calibration", "heldout"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, default=18)
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.numerical:
            parser.error("prepare requires numerical evidence")
        prepare(args.plan, args.numerical)
    elif args.action == "run" and (args.stage is None or args.seed is None):
        parser.error("run requires an explicit split and seed")
    else:
        execute(args.plan, args.action, args.stage, args.seed, args.limit)
