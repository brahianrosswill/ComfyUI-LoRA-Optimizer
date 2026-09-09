"""Conflict-aware skip decisions must describe the operation actually performed."""
import unittest
import tempfile
from unittest import mock

import torch

from tests.test_lora_optimizer import lora_optimizer as m
from tests.test_phase1_correctness import ToyPatcher, entry


class SparsificationContract(unittest.TestCase):
    def setUp(self):
        self.opt = m.LoRAOptimizer()
        self.a = torch.ones(16, 20)
        self.b = torch.full((16, 20), .25)
        self.b[:, ::2] *= -1  # exactly 50% signed disagreement

    def merge(self, mode, sparsification, generator=None):
        return self.opt._merge_diffs(
            [(self.a.clone(), .8), (self.b.clone(), .6)], mode,
            sparsification=sparsification, sparsification_density=.7,
            sparsification_generator=generator)

    def test_skipped_conflict_sparsification_is_the_disabled_merge(self):
        for mode in ("weighted_sum", "weighted_average", "normalize", "slerp"):
            expected = self.merge(mode, "disabled")
            for sparsification in ("dare_conflict", "della_conflict"):
                with self.subTest(mode=mode, sparsification=sparsification):
                    actual = self.merge(mode, sparsification, torch.Generator().manual_seed(81))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_skip_does_not_call_unconditional_sparsifiers_or_consume_rng(self):
        for sparsification in ("dare_conflict", "della_conflict"):
            with self.subTest(sparsification=sparsification):
                gen = torch.Generator().manual_seed(81)
                before = gen.get_state().clone()
                with mock.patch.object(self.opt, "_della_sparsify", wraps=self.opt._della_sparsify) as della:
                    self.merge("weighted_sum", sparsification, gen)
                della.assert_not_called()
                self.assertTrue(torch.equal(before, gen.get_state()))

    def test_low_conflict_still_uses_the_requested_conflict_sparsifier(self):
        self.b.fill_(.25)
        self.b[:, :4] *= -1  # below the existing guard; not a skip
        for sparsification, method in (("dare_conflict", "_dare_sparsify_conflict"),
                                       ("della_conflict", "_della_sparsify_conflict")):
            with self.subTest(sparsification=sparsification), mock.patch.object(
                    self.opt, method, wraps=getattr(self.opt, method)) as sparsify:
                self.merge("weighted_sum", sparsification, torch.Generator().manual_seed(81))
                self.assertEqual(sparsify.call_count, 2)


class SparsificationIntegration(unittest.TestCase):
    def setUp(self):
        self.opt = m.LoRAOptimizer()
        self.model = ToyPatcher()
        self.model.model.layer = torch.nn.Linear(20, 16)
        down = torch.full((1, 20), .25)
        down[:, ::2] *= -1
        self.stack = [entry({"layer.lora_A.weight": torch.ones(1, 20),
                             "layer.lora_B.weight": torch.ones(16, 1)}, name="a"),
                      entry({"layer.lora_A.weight": down,
                             "layer.lora_B.weight": torch.ones(16, 1)}, name="b")]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(m._LoRAMergeBase, "_get_compute_device", return_value=torch.device("cpu")).start()
        mock.patch.object(m.comfy.lora, "model_lora_keys_unet", return_value={"layer": "layer.weight"}).start()
        mock.patch.object(m, "AUTOTUNER_MEMORY_DIR", self.tmp.name).start()
        mock.patch.object(m.LoRAOptimizer, "_save_report_to_disk").start()

    def test_skipped_linear_group_retains_native_factors_without_compression(self):
        with mock.patch.object(self.opt, "_compress_to_lowrank", wraps=self.opt._compress_to_lowrank) as compress:
            result = self.opt.optimize_merge(self.model, self.stack, 1.,
                optimization_mode="global", merge_strategy_override="weighted_sum",
                sparsification="della_conflict", patch_compression="aggressive")
        compress.assert_not_called()
        patch = result[4]["model_patches"]["layer.weight"]
        self.assertIsInstance(patch, m.LoRAAdapter)
        self.assertEqual(patch.weights[1].shape[0], 2)
        self.assertEqual(result[4]["sparsification_summary"], {"applied_groups": 0, "skipped_conflict_groups": 1})
        expected = torch.ones(16, 20) + self.stack[1]["lora"]["layer.lora_B.weight"] @ self.stack[1]["lora"]["layer.lora_A.weight"]
        torch.testing.assert_close(self.opt._expand_patch_to_diff(patch), expected)

    def test_skipped_candidate_has_the_same_measured_score_as_disabled(self):
        base = dict(merge_mode="weighted_sum", optimization_mode="global", auto_strength="disabled",
                    sparsification="disabled", sparsification_density=.7, dare_dampening=0.,
                    merge_refinement="none", strategy_set="full")
        grid = [base, dict(base, sparsification="della_conflict")]
        for formula in ("v1", "v2"):
            with self.subTest(formula=formula), mock.patch.object(m, "_generate_param_grid", return_value=grid):
                result = m.LoRAAutoTuner().auto_tune(self.model, self.stack, 1., top_n=2,
                    scoring_device="cpu", scoring_svd="disabled", scoring_formula=formula,
                    memory_mode="disabled", community_cache="disabled", diff_cache_mode="disabled",
                    cache_patches="disabled", output_mode="tuning_only")
            rows = result[4]["top_n"]
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]["score_measured"], rows[1]["score_measured"], places=7)
            skipped = next(row for row in rows if row["config"]["sparsification"] != "disabled")
            self.assertEqual(skipped["metrics"]["sparsification_summary"]["applied_groups"], 0)
