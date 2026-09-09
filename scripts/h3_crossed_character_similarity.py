"""Freeze/apply the unchanged head-region diagnostic to the crossed pairs.

Separate provenance, same CPU metric/reference regions/frame samples. No
held-out access, fitted thresholds, quality labels or autotuner feedback.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

try:
    from . import h3_character_similarity as diagonal
    from . import h3_crossed_character_benchmark as crossed
    from .h3_benchmark import digest, save_new
except ImportError:
    import h3_character_similarity as diagonal
    import h3_crossed_character_benchmark as crossed
    from h3_benchmark import digest, save_new

SELF = Path(__file__).resolve()
ENGINE_SHA = "2b0aa3025055255f8bfdbf6cf878fdc18053757b898e44610d4bf3e286217690"
PARENT_RECIPE = diagonal.DATA / "2026-09-08-h3-character-similarity-recipe.json"
PARENT_SHA = "5b8b8ffc6e2cf390e33feba12ee7a6b56cb40f11acc31981886ced15abfeea81"
HELPERS = (*diagonal.HELPERS, "h3_crossed_character_benchmark.py", SELF.name)


def engine():
    if digest(diagonal.__file__) != ENGINE_SHA:
        raise ValueError("Frozen similarity engine changed")
    name = ((diagonal.__package__ + ".") if diagonal.__package__ else "") + "_h3_crossed_similarity"
    spec = importlib.util.spec_from_file_location(name, diagonal.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PAIRS = crossed.PAIRS
    return module


def parent_recipe():
    if digest(PARENT_RECIPE) != PARENT_SHA:
        raise ValueError("Parent calibration recipe changed")
    parent = json.loads(PARENT_RECIPE.read_text())
    for name, sha in parent["helper_sha256"].items():
        if digest(diagonal.ROOT / "scripts" / name) != sha:
            raise ValueError("Parent calibration helper changed")
    return parent


def prepare(recipe_path, render_path):
    if recipe_path.exists():
        raise FileExistsError(recipe_path)
    parent = parent_recipe()
    render_plan = json.loads(render_path.read_text())
    crossed.validate_plan(render_plan)
    metric = engine()
    jobs = metric.calibration_jobs(render_plan)
    if any((Path(render_plan["root"]) / j["id"] / "history.json").exists() for j in jobs):
        raise ValueError("Freeze crossed recipe before any crossed calibration outputs")
    recipe = dict(version=1, created_at=time.time(), execution_phase="crossed_pairs",
        parent_recipe_path=str(PARENT_RECIPE), parent_recipe_sha256=PARENT_SHA,
        render_plan_path=str(render_path.resolve()), render_plan_sha256=digest(render_path),
        probe_plan_path=parent["probe_plan_path"], probe_plan_sha256=parent["probe_plan_sha256"],
        inherited_probe=parent["inherited_probe"], jobs=jobs,
        helper_sha256={name: digest(diagonal.ROOT / "scripts" / name) for name in HELPERS},
        purpose="Unchanged paired raw similarity diagnostics for the remaining two calibration pairings, not a learned merge-quality score.",
        exposure="Both diagonal calibration seeds, their observations and similarity scores were exposed. No crossed calibration output exists at recipe freeze. No parameters, masks, frame indices or processor changed.",
        reporting=parent["reporting"], heldout_allowed=False, evaluator_feedback=False,
        quality_labels=[])
    validate_scope(recipe)
    save_new(recipe_path, recipe)
    print(json.dumps(dict(recipe=str(recipe_path), sha256=digest(recipe_path), jobs=len(jobs))))


def validate_scope(recipe):
    if (recipe.get("execution_phase") != "crossed_pairs"
            or recipe.get("parent_recipe_path") != str(PARENT_RECIPE)
            or recipe.get("parent_recipe_sha256") != PARENT_SHA
            or set(recipe.get("helper_sha256", {})) != set(HELPERS)
            or recipe.get("heldout_allowed") is not False
            or recipe.get("evaluator_feedback") is not False):
        raise ValueError("Expected the frozen research-only crossed calibration recipe")
    parent = parent_recipe()
    if any(recipe[k] != parent[k] for k in ("inherited_probe", "probe_plan_path", "probe_plan_sha256")):
        raise ValueError("Calibrated reference, processor or sampling changed")
    for name, sha in recipe["helper_sha256"].items():
        if digest(diagonal.ROOT / "scripts" / name) != sha:
            raise ValueError("Crossed evaluator helper changed after freeze")


def evaluate(recipe_path, seed, out):
    if seed not in diagonal.SEEDS:
        raise ValueError("Only the two declared calibration seeds may be evaluated")
    if out.exists():
        raise FileExistsError(out)
    recipe = json.loads(recipe_path.read_text())
    validate_scope(recipe)
    render_path = Path(recipe["render_plan_path"])
    if digest(render_path) != recipe["render_plan_sha256"]:
        raise ValueError("Frozen crossed render plan changed")
    crossed.validate_plan(json.loads(render_path.read_text()))
    # The pinned engine verifies complete matched output/audit hashes, model,
    # references, every sampled frame and all helpers before saving any score.
    engine().evaluate(recipe_path, seed, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--render-plan", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.render_plan is None:
            parser.error("prepare requires --render-plan")
        prepare(args.recipe, args.render_plan)
    elif args.seed is None or args.out is None:
        parser.error("evaluate requires --seed and --out")
    else:
        evaluate(args.recipe, args.seed, args.out)
