# LoRA Extract From Model — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `LoRAExtractFromModel` node that extracts a baked-in LoRA from a finetuned model by SVD-decomposing the delta against a clean base model, outputting both a `LORA_STACK` entry (feeds into the merger) and a `LORA_DATA` entry (feeds into `SaveMergedLoRA`).

**Architecture:** Subtract base model weights from finetuned weights per layer, SVD-decompose each delta to rank-r, and package the result in both formats. LORA_STACK uses the raw safetensors-style dict (same format `LoRAStack` already produces). LORA_DATA uses `LoRAAdapter` patches (same format the merger already produces).

**Tech Stack:** PyTorch (`torch.linalg.svd`), ComfyUI model APIs (`comfy.utils.get_attr`, `comfy.lora.model_lora_keys_unet/clip`), `LoRAAdapter` from `comfy.weight_adapter.lora` (already imported at line 24).

---

### Task 1: Add unit tests for the SVD extraction helper

**Files:**
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write the failing tests**

Add to the bottom of `tests/test_lora_optimizer.py`, after the existing test classes:

```python
@unittest.skipIf(torch is None, "torch not available")
class TestExtractLoRAFromDelta(unittest.TestCase):
    """Tests for _extract_lora_svd() helper used by LoRAExtractFromModel."""

    def _make_delta(self, rows, cols, rank):
        """Create a synthetic rank-r delta matrix."""
        U = torch.randn(rows, rank)
        S = torch.rand(rank) + 0.1  # positive singular values
        V = torch.randn(rank, cols)
        return (U * S.unsqueeze(0)) @ V

    def test_fixed_rank_output_shape(self):
        """Fixed mode: output lora_up/down have correct shapes."""
        mod = _load_module()
        delta = self._make_delta(64, 32, 8)
        up, down, alpha = mod._extract_lora_svd(delta, rank=4, rank_mode="fixed", energy_threshold=0.99)
        self.assertEqual(up.shape, (64, 4))
        self.assertEqual(down.shape, (4, 32))
        self.assertEqual(alpha, 4.0)

    def test_auto_rank_energy_retained(self):
        """Auto mode: retained energy >= threshold."""
        mod = _load_module()
        delta = self._make_delta(64, 32, 16)
        up, down, alpha = mod._extract_lora_svd(delta, rank=32, rank_mode="auto", energy_threshold=0.95)
        # Reconstruct and check energy
        reconstructed = up @ down
        original_energy = (delta ** 2).sum().item()
        reconstructed_energy = (reconstructed ** 2).sum().item()
        self.assertGreaterEqual(reconstructed_energy / original_energy, 0.90)  # allow small numerical error

    def test_near_zero_delta_returns_none(self):
        """Near-zero delta (unaffected layer) returns None."""
        mod = _load_module()
        delta = torch.zeros(64, 32)
        result = mod._extract_lora_svd(delta, rank=4, rank_mode="fixed", energy_threshold=0.99)
        self.assertIsNone(result)

    def test_singular_value_floor_applied(self):
        """Noise-only singular values below floor are excluded even in fixed mode."""
        mod = _load_module()
        # Create a rank-2 signal with tiny noise
        signal = self._make_delta(32, 32, 2)
        noise = torch.randn(32, 32) * 1e-6
        delta = signal + noise
        up, down, alpha = mod._extract_lora_svd(delta, rank=16, rank_mode="fixed", energy_threshold=0.99)
        # Rank should be clamped to signal rank (2), not 16, due to floor
        self.assertLessEqual(alpha, 4.0)  # floor should cut most of the 16 slots
```

Also add a `_load_module()` helper near the top of the test file (after `_install_stubs()`):

```python
def _load_module():
    """Load lora_optimizer module with stubs in place."""
    _install_stubs()
    spec = importlib.util.spec_from_file_location(
        "lora_optimizer",
        os.path.join(os.path.dirname(__file__), "..", "lora_optimizer.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

**Step 2: Run tests to verify they fail**

```bash
cd /media/p5/ComfyUI-ZImage-LoRA-Merger
python -m pytest tests/test_lora_optimizer.py::TestExtractLoRAFromDelta -v
```

Expected: `ERROR` — `module '_load_module' not found` or `AttributeError: module has no attribute '_extract_lora_svd'`

**Step 3: Commit the failing tests**

```bash
git add tests/test_lora_optimizer.py
git commit -m "test: add failing tests for _extract_lora_svd helper"
```

---

### Task 2: Implement `_extract_lora_svd()` module-level helper

**Files:**
- Modify: `lora_optimizer.py` — add after `_compress_to_lowrank` (search for it, around line 2800)

**Step 1: Find the insertion point**

Search for `def _compress_to_lowrank` in `lora_optimizer.py` and insert the new function immediately after its closing line.

**Step 2: Add the helper**

```python
def _extract_lora_svd(delta: torch.Tensor, rank: int, rank_mode: str, energy_threshold: float):
    """
    SVD-decompose a weight delta into (lora_up, lora_down, alpha).

    Returns None if the delta is near-zero (layer unaffected by the LoRA).
    Returns (lora_up, lora_down, alpha) otherwise.

    lora_up  shape: (rows, r)
    lora_down shape: (r, cols)
    alpha = float(r)  — effective scale = 1.0 when loaded at strength 1.0
    """
    NEAR_ZERO_NORM = 1e-8
    SV_FLOOR_RATIO = 1e-4  # drop singular values below floor_ratio * S[0]

    if delta.norm().item() < NEAR_ZERO_NORM:
        return None

    # Ensure 2D: conv layers arrive as [C_out, C_in, kH, kW]
    original_shape = delta.shape
    if delta.ndim > 2:
        delta = delta.reshape(delta.shape[0], -1)

    delta = delta.float()

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

    # Apply singular value floor to exclude noise
    sv_floor = SV_FLOOR_RATIO * S[0].item()
    valid = S > sv_floor
    if not valid.any():
        return None
    U, S, Vh = U[:, valid], S[valid], Vh[valid, :]

    # Determine rank r
    if rank_mode == "auto":
        energy_cumsum = S.pow(2).cumsum(0) / S.pow(2).sum()
        r = int((energy_cumsum < energy_threshold).sum().item()) + 1
        r = min(r, S.shape[0])
    else:
        r = min(rank, S.shape[0])

    sqrt_s = S[:r].sqrt()
    lora_up = U[:, :r] * sqrt_s.unsqueeze(0)    # (rows, r)
    lora_down = sqrt_s.unsqueeze(1) * Vh[:r, :]  # (r, cols)

    return lora_up, lora_down, float(r)
```

**Step 3: Run the tests**

```bash
python -m pytest tests/test_lora_optimizer.py::TestExtractLoRAFromDelta -v
```

Expected: all 4 tests PASS.

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add _extract_lora_svd helper for LoRA extraction"
```

---

### Task 3: Implement `LoRAExtractFromModel` node class

**Files:**
- Modify: `lora_optimizer.py` — add the class before `NODE_CLASS_MAPPINGS` (near line 10476)

**Step 1: Find the insertion point**

Search for `NODE_CLASS_MAPPINGS` in `lora_optimizer.py`. Insert the new class in the block of node classes just before it.

**Step 2: Add the node class**

```python
class LoRAExtractFromModel:
    """
    Extracts a baked-in LoRA from a finetuned model by subtracting the clean
    base model weights and SVD-decomposing the per-layer delta.

    Requires the original base model as a reference.
    Outputs:
      - LORA_STACK: feeds directly into LoRAOptimizerSimple / LoRAAutoTuner
      - LORA_DATA:  feeds directly into SaveMergedLoRA
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL", {
                    "tooltip": "The original clean base model (e.g. the Flux or SDXL base checkpoint "
                               "that was used as the starting point for finetuning)."
                }),
                "finetuned_model": ("MODEL", {
                    "tooltip": "The finetuned model with a LoRA already baked into its weights."
                }),
                "rank": ("INT", {
                    "default": 32, "min": 1, "max": 512, "step": 1,
                    "tooltip": "Maximum rank for SVD decomposition. Used directly in 'fixed' mode; "
                               "acts as an upper bound in 'auto' mode."
                }),
                "rank_mode": (["auto", "fixed"], {
                    "default": "auto",
                    "tooltip": "'auto': choose rank to retain the given energy fraction (recommended). "
                               "'fixed': always use exactly the specified rank."
                }),
                "energy_threshold": ("FLOAT", {
                    "default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of delta energy to retain when rank_mode='auto'. "
                               "0.99 = retain 99% of the signal. Higher = more accurate, higher rank."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Strength to assign this extracted LoRA in the output stack."
                }),
            }
        }

    RETURN_TYPES = ("LORA_STACK", "LORA_DATA")
    RETURN_NAMES = ("lora_stack", "lora_data")
    FUNCTION = "extract"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Extracts a LoRA that was baked into a finetuned model by comparing it against "
        "the original base model. Requires the base model as a reference. "
        "Connect lora_stack to LoRAOptimizerSimple to merge additional LoRAs on top, "
        "or connect lora_data to SaveMergedLoRA to save the extracted LoRA to disk."
    )

    def extract(self, base_model, finetuned_model, rank, rank_mode, energy_threshold, strength):
        import comfy.lora

        base_sd = base_model.model.state_dict()
        fine_sd = finetuned_model.model.state_dict()

        # Build inverse key map: model_weight_key → lora_prefix
        # comfy.lora.model_lora_keys_unet returns {lora_prefix: model_key}
        lora_to_model_unet = {}
        lora_to_model_clip = {}
        try:
            lora_to_model_unet = comfy.lora.model_lora_keys_unet(base_model.model, {})
        except Exception:
            pass
        try:
            lora_to_model_clip = comfy.lora.model_lora_keys_clip(base_model.model, {})
        except Exception:
            pass

        model_to_lora = {}  # model_key → (lora_prefix, is_clip)
        for lora_prefix, model_key in lora_to_model_unet.items():
            model_to_lora[model_key] = (lora_prefix, False)
        for lora_prefix, model_key in lora_to_model_clip.items():
            model_to_lora[model_key] = (lora_prefix, True)

        # Validate that both models share the same keys
        base_keys = set(base_sd.keys())
        fine_keys = set(fine_sd.keys())
        if base_keys != fine_keys:
            missing = base_keys - fine_keys
            extra = fine_keys - base_keys
            logging.warning(
                f"[LoRAExtract] Key mismatch between base and finetuned models. "
                f"Missing in finetuned: {len(missing)}, extra in finetuned: {len(extra)}. "
                f"Proceeding with shared keys only."
            )

        # Counters for diagnostics
        n_processed = 0
        n_skipped_zero = 0
        n_skipped_no_map = 0
        n_rank_capped = 0

        # Output containers
        raw_lora_dict = {}        # for LORA_STACK: {prefix.lora_up.weight: tensor, ...}
        model_patches = {}        # for LORA_DATA
        clip_patches = {}
        key_map = {}              # target_key → {"canonical_prefix": lora_prefix}

        shared_keys = base_keys & fine_keys

        for model_key in sorted(shared_keys):
            if model_key not in model_to_lora:
                n_skipped_no_map += 1
                continue

            W_base = base_sd[model_key]
            W_fine = fine_sd[model_key]

            if W_base.shape != W_fine.shape:
                logging.warning(f"[LoRAExtract] Shape mismatch for {model_key}, skipping.")
                continue

            # Skip non-float layers (embeddings, norms, etc.)
            if not W_base.is_floating_point():
                continue

            delta = W_fine.float() - W_base.float()
            result = _extract_lora_svd(delta, rank=rank, rank_mode=rank_mode, energy_threshold=energy_threshold)

            if result is None:
                n_skipped_zero += 1
                continue

            lora_up, lora_down, alpha = result
            lora_prefix, is_clip = model_to_lora[model_key]

            if alpha >= rank and rank_mode == "fixed":
                n_rank_capped += 1

            # Build raw LoRA dict entry (for LORA_STACK)
            raw_lora_dict[f"{lora_prefix}.lora_up.weight"] = lora_up
            raw_lora_dict[f"{lora_prefix}.lora_down.weight"] = lora_down
            raw_lora_dict[f"{lora_prefix}.alpha"] = torch.tensor(alpha)

            # Build patch entry (for LORA_DATA)
            patch = LoRAAdapter(set(), (lora_up, lora_down, alpha, None, None, None))
            if is_clip:
                clip_patches[model_key] = patch
            else:
                model_patches[model_key] = patch

            key_map[model_key] = {"canonical_prefix": lora_prefix}
            n_processed += 1

        logging.info(
            f"[LoRAExtract] Extracted {n_processed} layers "
            f"({len(model_patches)} model, {len(clip_patches)} CLIP). "
            f"Skipped: {n_skipped_zero} near-zero, {n_skipped_no_map} unmapped. "
            f"Rank-capped layers: {n_rank_capped}."
        )

        if n_processed == 0:
            logging.warning("[LoRAExtract] No layers extracted — models may be identical or incompatible.")

        # Warn if this looks like a full finetune rather than a LoRA merge
        total_float_layers = sum(
            1 for k in shared_keys
            if k in base_sd and base_sd[k].is_floating_point()
        )
        if total_float_layers > 0 and n_processed > total_float_layers * 0.5:
            logging.warning(
                f"[LoRAExtract] {n_processed}/{total_float_layers} layers have significant deltas. "
                f"This may be a full finetune rather than a LoRA merge — "
                f"extraction will produce a high-rank approximation."
            )

        # LORA_STACK output
        lora_stack = [{
            "name": "<extracted from finetuned_model>",
            "lora": raw_lora_dict,
            "strength": strength,
            "conflict_mode": "all",
            "key_filter": "all",
            "metadata": {},
        }]

        # LORA_DATA output
        lora_data = {
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "key_map": key_map,
            "output_strength": strength,
            "clip_strength": strength,
            "suggested_max_strength": None,
            "sum_rank": rank,
            "merge_metadata": {
                "source_loras": [{"name": "<extracted>", "strength": strength}],
                "mode": "extract",
                "architecture": "unknown",
            },
        }

        return (lora_stack, lora_data)
```

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add LoRAExtractFromModel node class"
```

---

### Task 4: Register the node in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

**Files:**
- Modify: `lora_optimizer.py` — lines around 10476

**Step 1: Find NODE_CLASS_MAPPINGS**

Search for `NODE_CLASS_MAPPINGS = {` in `lora_optimizer.py`.

**Step 2: Add the entry**

Add to `NODE_CLASS_MAPPINGS`:
```python
"LoRAExtractFromModel": LoRAExtractFromModel,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS` (search for it immediately below):
```python
"LoRAExtractFromModel": "LoRA Extract From Model",
```

**Step 3: Run the existing tests to verify no regressions**

```bash
python -m pytest tests/test_lora_optimizer.py -v
```

Expected: all existing tests PASS + the 4 new extraction tests PASS.

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: register LoRAExtractFromModel in node mappings"
```

---

### Task 5: End-to-end verification

No automated test is possible here (requires real ComfyUI models), but verify the module loads cleanly:

**Step 1: Verify module imports without error**

```bash
cd /media/p5/ComfyUI-ZImage-LoRA-Merger
python -c "
import sys, types, tempfile, os
tmpdir = tempfile.gettempdir()

# Minimal stubs
fp = types.ModuleType('folder_paths')
fp.models_dir = tmpdir
fp.get_temp_directory = lambda: tmpdir
fp.get_user_directory = lambda: tmpdir
fp.get_folder_paths = lambda _: [tmpdir]
fp.get_filename_list = lambda _: []
fp.get_full_path_or_raise = lambda _, n: n
sys.modules['folder_paths'] = fp

import importlib.util
spec = importlib.util.spec_from_file_location('lora_optimizer', 'lora_optimizer.py')
mod = importlib.util.module_from_spec(spec)
# Expect ImportError on comfy — that's fine, means the file parsed correctly
try:
    spec.loader.exec_module(mod)
except Exception as e:
    if 'comfy' in str(e) or 'safetensors' in str(e):
        print('OK: module parsed, expected import error on ComfyUI deps')
    else:
        raise
"
```

Expected output: `OK: module parsed, expected import error on ComfyUI deps`

**Step 2: Verify node appears in mappings**

```bash
python -c "
# Same stubs as above, then:
# print('LoRAExtractFromModel' in mod.NODE_CLASS_MAPPINGS)
"
```

Expected: `True`

**Step 3: Final commit if any fixups needed, then tag**

```bash
git log --oneline -5
```
