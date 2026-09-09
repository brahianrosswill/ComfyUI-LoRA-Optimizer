import copy

import pytest

from scripts.h3_fate_calibration import join_labels, validate_cases


def fixture():
    cases = [{"case_id": f"A{i:02d}", "sha256": f"{i:064x}"} for i in range(1, 13)]
    predictions = {c["sha256"]: i / 100 for i, c in enumerate(cases)}
    arms = {"combat_cinema": [(1, "second"), (2, "np"), (3, "additive"), (5, "combat"), (8, "winner"), (9, "base"), (12, "ct")],
            "combat_repair": [(4, "winner"), (5, "combat"), (6, "second"), (7, "ct"), (9, "base"), (10, "np"), (11, "additive")]}
    labels = [{"pair": pair, "variant": variant, "blind_id": f"{pair}-{i}", "original_sha256": f"{i:064x}",
               "reviewer_id": "R1", "seed": 2026090803, "stage": "calibration",
               "scores": {"synchronization": 2 if pair == "combat_cinema" else 1}}
              for pair, group in arms.items() for i, variant in group]
    return cases, predictions, labels


def test_shared_sources_retain_all_contextual_labels():
    cases, predictions, labels = fixture()
    rows = join_labels(cases, predictions, labels)
    assert len(rows) == 14 and len({r["source_sha256"] for r in rows}) == 12
    base = [r for r in rows if r["case_id"] == "A09"]
    assert [r["human_sync"] for r in base] == [2, 1]
    assert base[0]["fate_score"] == base[1]["fate_score"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reordered", "heldout_identity"])
def test_exact_twelve_source_scope(kind):
    cases, _, _ = fixture()
    if kind == "missing": cases.pop()
    if kind == "duplicate": cases[-1]["sha256"] = cases[0]["sha256"]
    if kind == "reordered": cases.reverse()
    if kind == "heldout_identity": cases[-1]["case_id"] = "H12"
    with pytest.raises(ValueError): validate_cases(cases)


@pytest.mark.parametrize("kind", ["label_missing", "label_duplicate", "wrong_hash", "wrong_pair", "wrong_arm", "wrong_reviewer",
                                  "heldout_seed", "wrong_stage", "nonfinite_model", "boolean_human", "out_of_range", "prediction_missing"])
def test_corrupt_or_wrong_scope_joins_fail(kind):
    cases, predictions, labels = copy.deepcopy(fixture())
    if kind == "label_missing": labels.pop()
    if kind == "label_duplicate": labels[-1] = labels[-2].copy()
    if kind == "wrong_hash": labels[0]["original_sha256"] = "foreign"
    if kind == "wrong_pair": labels[0]["pair"] = "foreign"
    if kind == "wrong_arm": labels[0]["variant"] = "foreign"
    if kind == "wrong_reviewer": labels[0]["reviewer_id"] = "R2"
    if kind == "heldout_seed": labels[0]["seed"] = 2026090804
    if kind == "wrong_stage": labels[0]["stage"] = "heldout"
    if kind == "nonfinite_model": predictions[cases[0]["sha256"]] = float("nan")
    if kind == "boolean_human": labels[0]["scores"]["synchronization"] = True
    if kind == "out_of_range": labels[0]["scores"]["synchronization"] = 5
    if kind == "prediction_missing": predictions.pop(cases[0]["sha256"])
    with pytest.raises(ValueError): join_labels(cases, predictions, labels)
