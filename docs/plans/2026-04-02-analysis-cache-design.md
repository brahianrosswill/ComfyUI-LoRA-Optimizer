# AutoTuner Analysis Cache Design

**Date:** 2026-04-02
**Status:** Approved

## Problem

The AutoTuner analysis pass (computing LoRA diffs, pairwise conflict sampling, magnitude statistics) is the most expensive part of a tuning run. Currently the results cache (`lora_hash`) includes LoRA strengths in its key, so changing any strength invalidates the cache entirely and forces a full re-analysis — even though conflict ratios, cosine similarities, and SVD directions are scale-invariant and do not change with strength.

This makes strength exploration slow: each strength sweep runs the full GPU-intensive analysis from scratch.

## Goal

Cache the scale-invariant analysis results separately, keyed only by LoRA file identity (not strengths or settings), so that strength changes reuse the analysis pass and only recompute the cheap magnitude-dependent synthesis.

## Design

### Cache Key

`names_only_hash`: 16-char SHA256 of the sorted list of `(name, mtime, file_size)` tuples for all LoRAs in the stack. No strengths, no settings.

- Fast: single `os.stat()` call per file, no content read
- Reliable: covers 99%+ of real changes (file replacement changes mtime or size)
- Invalidates automatically on: file replacement, LoRA set changes, algo version bumps

### File Format

Stored at `AUTOTUNER_MEMORY_DIR/<names_only_hash>.analysis.json`.

```json
{
  "analysis_version": 1,
  "algo_version": "<AUTOTUNER_ALGO_VERSION>",
  "created_at": "2026-04-02T...",
  "source_loras": [
    {"name": "lora_a.safetensors", "mtime": 1234567890.0, "size": 102400}
  ],
  "per_prefix": {
    "<lora_prefix>": {
      "pair_conflicts": {
        "0,1": [n_overlap, n_conflict, dot, norm_a_sq, norm_b_sq]
      },
      "diff_norms": {"0": 1.234, "1": 0.987},
      "ranks": {"0": 16, "1": 32},
      "target_key": "...",
      "is_clip": false
    }
  }
}
```

`pair_conflicts` stores raw scale-invariant components (not derived ratios) so strength rescaling can be correctly applied at synthesis time. `diff_norms` stores `diff.norm()` without strength applied.

### Read/Write Flow

In `auto_tune`, before the analysis pass:

1. Compute `names_only_hash` from `(name, mtime, size)` per LoRA file
2. Attempt to load `<names_only_hash>.analysis.json`
3. **Cache hit** (and `algo_version` matches):
   - Reconstruct `pair_conflicts` directly (scale-invariant, no change needed)
   - Reconstruct magnitude stats by scaling `diff_norms` by `abs(strength)` per LoRA
   - Check for sign flips vs cached data: if any strength changed sign, treat as miss and run full analysis (conservative, sign flips are rare)
   - Skip `_analyze_prefix` entirely
4. **Cache miss**: run analysis as normal, then save `.analysis.json` atomically (`tmp` → `os.replace`)

The existing results cache (`memory_lora_hash + settings_hash`) is unchanged — it sits on top and continues to work as before.

### Dataset Persistence

`_save_tuner_dataset_entry` gets a `raw_analysis` field added with the same `per_prefix` structure. This ensures analysis data survives even if `.analysis.json` files are manually cleared, and feeds any future cross-run reuse from the dataset.

### Invalidation Rules

| Trigger | Effect |
|---|---|
| File `mtime` or `size` changes | `names_only_hash` changes → miss |
| LoRA added/removed from stack | Different set → miss |
| `algo_version` bump | File rejected → miss |
| Strength sign flip | Conservative full-analysis fallback |
| `memory_mode = "clear_and_run"` | Also deletes `.analysis.json` for current set |
| Corrupted/unparseable file | Caught, treated as miss |

### What Does Not Change

- No new UI settings or toggles — fully transparent to the user
- `memory_mode` semantics unchanged
- Existing `.memory.json` file format unchanged
- `lora_hash` and `settings_hash` computation unchanged

## Expected Benefit

For a strength sweep of N candidates over the same LoRA set:

- **Before**: O(N × full_analysis_cost)
- **After**: O(analysis_cost + N × synthesis_cost)

Since `_analyze_prefix` (diff matmuls + GPU conflict sampling) dominates cost, synthesis (magnitude rescaling + heuristic scoring) is cheap arithmetic. Speedup scales with N.

Cross-run benefit: returning to the same LoRA set in a future ComfyUI session skips the full analysis pass on first run too.
