# WanVideoLoRAOptimizer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `WanVideoLoRAOptimizer` node that accepts `WANVIDEOMODEL` and outputs a patched `WANVIDEOMODEL` with merged LoRA patches, reusing all existing merging algorithms via inheritance.

**Architecture:** Thin subclass of `LoRAOptimizer` (~30 lines). Overrides `INPUT_TYPES` (replaces `MODEL` → `WANVIDEOMODEL`, removes CLIP inputs, changes defaults), `RETURN_TYPES` (drops CLIP output), and wraps `optimize_merge()` to pass `clip=None` and remap the return tuple.

**Tech Stack:** Python, ComfyUI node system, existing `LoRAOptimizer` class

---

### Task 1: Add WanVideoLoRAOptimizer class

**Files:**
- Modify: `lora_optimizer.py:2938-2940` (insert after `LoRAOptimizer` class, before `SaveMergedLoRA`)

**Step 1: Add the class after line 2939 (the blank line between LoRAOptimizer and SaveMergedLoRA)**

Insert the following class:

```python
class WanVideoLoRAOptimizer(LoRAOptimizer):
    """
    WanVideo variant of the LoRA Optimizer. Accepts WANVIDEOMODEL instead of
    MODEL, skips CLIP, and applies merged LoRA patches in-memory.

    All merging algorithms (TIES, DARE/DELLA, SVD compression, auto-strength,
    conflict analysis) are inherited from LoRAOptimizer. Wan LoRA key
    normalization (LyCORIS, diffusers, Fun LoRA, finetrainer, RS-LoRA) is
    already handled by the parent's _normalize_keys_wan.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = LoRAOptimizer.INPUT_TYPES()
        # Replace MODEL with WANVIDEOMODEL
        base["required"]["model"] = ("WANVIDEOMODEL", {
            "tooltip": "Your WanVideo model from WanVideoModelLoader."
        })
        # Remove CLIP-related inputs (WanVideo doesn't use CLIP)
        base["optional"].pop("clip", None)
        base["optional"].pop("clip_strength_multiplier", None)
        base["optional"].pop("free_vram_between_passes", None)
        # Change defaults for video models
        base["optional"]["cache_patches"] = (["disabled", "enabled"], {
            "default": "disabled",
            "tooltip": "Keep the merge result in memory so re-running the workflow is instant. Disabled by default for large video models to save RAM."
        })
        base["optional"]["normalize_keys"] = (["enabled", "disabled"], {
            "default": "enabled",
            "tooltip": "Normalizes LoRA keys from different training tools (LyCORIS, diffusers, finetrainer, etc.) to a common format. Enabled by default for WanVideo LoRAs."
        })
        return base

    RETURN_TYPES = ("WANVIDEOMODEL", "STRING", "LORA_DATA")
    RETURN_NAMES = ("model", "analysis_report", "lora_data")
    CATEGORY = "loaders/lora"
    DESCRIPTION = (
        "WanVideo LoRA Optimizer — merges multiple WanVideo LoRAs using "
        "conflict-aware algorithms (TIES, DARE, auto-strength). "
        "Connect after WanVideoModelLoader, before WanVideoSampler."
    )

    def optimize_merge(self, model, lora_stack, output_strength, **kwargs):
        kwargs.pop("clip", None)
        kwargs.pop("clip_strength_multiplier", None)
        kwargs.pop("free_vram_between_passes", None)
        result = super().optimize_merge(
            model, lora_stack, output_strength,
            clip=None, clip_strength_multiplier=0, **kwargs
        )
        # Parent returns (model, clip, report, lora_data) — drop clip
        return (result[0], result[2], result[3])

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, output_strength, **kwargs):
        kwargs.pop("clip", None)
        kwargs.pop("clip_strength_multiplier", None)
        kwargs.pop("free_vram_between_passes", None)
        return LoRAOptimizer.IS_CHANGED(
            model, lora_stack, output_strength,
            clip=None, clip_strength_multiplier=0, **kwargs
        )
```

**Step 2: Verify the class is syntactically correct**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"` from the project directory.
Expected: `OK`

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add WanVideoLoRAOptimizer node"
```

---

### Task 2: Register in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

**Files:**
- Modify: `lora_optimizer.py:3513-3529` (the mappings dicts at end of file)

**Step 1: Add entries to both dicts**

In `NODE_CLASS_MAPPINGS` (after `MergedLoRAToHook` entry):
```python
    "WanVideoLoRAOptimizer": WanVideoLoRAOptimizer,
```

In `NODE_DISPLAY_NAME_MAPPINGS` (after `MergedLoRAToHook` entry):
```python
    "WanVideoLoRAOptimizer": "WanVideo LoRA Optimizer",
```

**Step 2: Verify ComfyUI can load the module**

Run: `cd /media/p5/Comfyui && python -c "import importlib.util; spec = importlib.util.spec_from_file_location('lora_optimizer', '/media/p5/ComfyUI-ZImage-LoRA-Merger/lora_optimizer.py'); mod = importlib.util.module_from_spec(spec); print('Import OK')"`
Expected: `Import OK` (note: full exec may fail outside ComfyUI runtime, but import/parse should succeed)

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: register WanVideoLoRAOptimizer in node mappings"
```

---

### Task 3: Update README with WanVideoLoRAOptimizer documentation

**Files:**
- Modify: `README.md`

**Step 1: Add a section for WanVideo LoRA Optimizer**

Add after the existing "Merged LoRA to Hook" section, following the same formatting pattern (heading, description, inputs/outputs table, collapsible details for workflow).

Key content:
- Node name and purpose
- Inputs table (WANVIDEOMODEL, LORA_STACK, same options as main optimizer minus CLIP)
- Outputs table (WANVIDEOMODEL, analysis_report, lora_data)
- Workflow diagram showing WanVideoModelLoader → WanVideoLoRAOptimizer → WanVideoSampler
- Collapsible section with usage notes (chaining with individual LoRAs, normalize_keys default, etc.)

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add WanVideoLoRAOptimizer to README"
```
