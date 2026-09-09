"""Retired nodes must not return to the menu or generated node schemas."""
from tests.test_lora_optimizer import lora_optimizer as m


def test_retired_nodes_are_not_registered():
    removed = {"LoRAOptimizer", "WanVideoLoRAOptimizer", "MergedLoRAToWanVideo"}
    assert removed.isdisjoint(m.NODE_CLASS_MAPPINGS)
    assert removed.isdisjoint(m.NODE_DISPLAY_NAME_MAPPINGS)
    assert set(m.NODE_CLASS_MAPPINGS) == set(m.NODE_DISPLAY_NAME_MAPPINGS)
    assert not any("Legacy" in name or "WanVideo" in name
                   for name in m.NODE_DISPLAY_NAME_MAPPINGS.values())


def test_shared_engine_and_supported_nodes_remain_available():
    assert m.NODE_CLASS_MAPPINGS["LoRAOptimizerSimple"] is m.LoRAOptimizerSimple
    assert m.NODE_CLASS_MAPPINGS["LoRAOptimizerInline"] is m.LoRAOptimizerInline
    assert m.NODE_CLASS_MAPPINGS["LoRAAutoTuner"] is m.LoRAAutoTuner
    assert m.NODE_CLASS_MAPPINGS["LoRAMergeSelector"] is m.LoRAMergeSelector
    assert issubclass(m.LoRAOptimizerSimple, m.LoRAOptimizer)
    assert not hasattr(m, "WanVideoLoRAOptimizer")
    assert not hasattr(m, "MergedLoRAToWanVideo")
