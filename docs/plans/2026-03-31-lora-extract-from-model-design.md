# LoRA Extract From Model — Design

## Context

Users sometimes work with finetuned checkpoints that already have a LoRA baked in (merged into the full model weights). When they try to add additional LoRAs via the merger, it creates issues: the merger's conflict analysis and auto-strength assume a clean base model, so scoring goes wrong and oversaturation occurs because the pre-merged LoRA is invisible to the pipeline.

The fix is to extract the baked-in LoRA as a proper low-rank file by subtracting the original base model weights, then inject it into the merge stack as a first-class entry.

## Approach

**SVD delta extraction with base model as prerequisite.**

Compute `delta = W_finetuned - W_base` per layer, SVD-decompose each delta to rank-r, and output a proper LoRA dict. The result is a virtual LoRA fully compatible with the existing merge pipeline — no special-casing needed downstream.

## New Node: `LoRAExtractFromModel`

**Inputs:**

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `base_model` | MODEL | — | Clean base checkpoint |
| `finetuned_model` | MODEL | — | Model with LoRA baked in |
| `rank` | INT | 32 | Max rank for SVD truncation |
| `rank_mode` | COMBO | `auto` | `auto` = retain % energy; `fixed` = always use `rank` |
| `energy_threshold` | FLOAT | 0.99 | Fraction of delta energy to retain (auto mode only) |
| `strength` | FLOAT | 1.0 | Scale applied to extracted LoRA in the stack |

**Outputs:**

| Name | Type | Purpose |
|------|------|---------|
| `LORA_STACK` | LORA_STACK | Feeds into LoRAOptimizerSimple / LoRAAutoTuner |
| `LORA_DATA` | LORA_DATA | Feeds directly into SaveMergedLoRA (reuses existing node) |

## Extraction Pipeline

Per layer:

1. Fetch `W_base` and `W_finetuned` via `comfy.utils.get_attr()`
2. Cast both to `float32`, compute `delta = W_finetuned - W_base`
3. Skip layers where `delta.norm() < epsilon` (unaffected — avoids noise)
4. Reshape delta to 2D if needed (conv layers: `[C_out, C_in * kH * kW]`)
5. `U, S, Vh = torch.linalg.svd(delta, full_matrices=False)`
6. Apply singular value floor: drop `S[i] < 1e-4 * S[0]` before energy calc (prevents noise inflation)
7. Determine rank `r`:
   - `auto`: smallest `r` where `S[:r].pow(2).sum() / S.pow(2).sum() >= energy_threshold`
   - `fixed`: `min(rank, S.shape[0])`
8. `lora_up = U[:, :r] * S[:r].sqrt()`; `lora_down = S[:r].sqrt().unsqueeze(1) * Vh[:r]`
9. Set `alpha = r` (effective scale = 1.0; user controls via `strength`)

**Key mapping:** model weight keys → LoRA prefix format via `comfy.lora.model_lora_keys_unet()` / `model_lora_keys_clip()` (inverse lookup, already used elsewhere in the codebase).

**Memory:** process one layer at a time, free tensors immediately — never materialize the full delta dict at once. Critical for large models (Flux ~23GB × 2).

## Warnings & Validation

| Condition | Action |
|-----------|--------|
| Architectures differ between models | Raise error before processing |
| `delta.norm()` low on >80% of layers | Warn: models may be identical or same base |
| Significant deltas on nearly all layers | Warn: looks like a full finetune, not a LoRA merge — extraction may produce a noisy high-rank result |
| Extracted rank consistently hitting `rank` cap | Warn: consider increasing `rank` or switching to `auto` mode |

## Integration

- `LORA_STACK` output is structurally identical to `LoRAStack` output — plugs into `LoRAOptimizerSimple`, `LoRAAutoTuner`, and `LoRAConflictEditor` with no changes
- `LORA_DATA` output matches the merger's output format (`model_patches`, `clip_patches`, `key_map`, `output_strength`, `clip_strength`) — plugs directly into the existing `SaveMergedLoRA` node
- Architecture detection reuses `_detect_architecture()` (line 683)
- Node registered in `NODE_CLASS_MAPPINGS` in `lora_optimizer.py` (line 10476)

## Files to Modify

- `lora_optimizer.py` — add `LoRAExtractFromModel` class, register in `NODE_CLASS_MAPPINGS`
