# Architecture-Aware Key Normalization Design

## Problem

The LoRA Optimizer merges multiple LoRAs by grouping their weights by key prefix. When two LoRAs use different key formats for the same model layer (e.g., one trained with Kohya, another with AI-Toolkit), their keys don't align — the optimizer treats them as separate prefixes and they never interact during conflict analysis or merging. This is a silent failure.

The most severe case is Z-Image Turbo, which uses fused QKV attention (`attention.qkv [11520, 3840]`) instead of separate `to_q`/`to_k`/`to_v`. A LoRA trained on diffusers format has separate Q/K/V keys that cannot be directly merged with a LoRA that targets Z-Image's native fused format.

## Solution

Add architecture detection and per-architecture key normalization as a pre-processing step inside `_normalize_stack()`, gated by a new `normalize_keys` boolean input (default: disabled).

When enabled:
1. Auto-detect architecture from LoRA key patterns
2. Normalize all LoRAs to a canonical key format for that architecture
3. The rest of the pipeline (prefix grouping, analysis, merge) works unchanged
4. For Z-Image: re-fuse split Q/K/V back to `qkv` at patch application time

## Architecture Detection

New `_detect_architecture(lora_sd)` method on `_LoRAMergeBase`. Returns `"zimage"`, `"flux"`, `"wan"`, `"sdxl"`, or `"unknown"`.

Detection order (first match wins):
- **Z-Image**: `diffusion_model.layers.N` or `single_transformer_blocks.N` (without `transformer.` prefix) or `lora_unet_layers_`
- **FLUX**: `double_blocks`/`single_blocks` or `transformer.transformer_blocks`/`transformer.single_transformer_blocks`
- **Wan**: `blocks.N` with `self_attn`/`cross_attn`/`ffn`
- **SDXL**: `lora_te1_`/`lora_te2_` or `input_blocks`/`output_blocks`/`down_blocks`/`up_blocks`

## Per-Architecture Normalization

New `_normalize_keys(lora_sd, architecture)` method. Returns a new dict with normalized keys.

### Z-Image (Lumina2)
1. **QKV splitting**: Fused `attention.qkv.lora_A/B` split into `attention.to_q`, `attention.to_k`, `attention.to_v` (slice along dim=0, /3). Alpha copied to all three.
2. **Output remap**: `attention.out` -> `attention.to_out.0`
3. **Prefix standardization**: Ensure `diffusion_model.layers.N.` prefix
4. **Musubi Tuner format**: `lora_unet_layers_N_attention_...` -> `diffusion_model.layers.N.attention...`

### FLUX
1. **AI-Toolkit**: `transformer.transformer_blocks.N` -> `diffusion_model.double_blocks.N`; `transformer.single_transformer_blocks.N` -> `diffusion_model.single_blocks.N`
2. **Kohya**: `lora_transformer_double_blocks_N` -> `diffusion_model.double_blocks.N`
3. **Standard**: Ensure `diffusion_model.` prefix

### Wan
Port existing `_wan_standardize_lora_key_format` logic:
1. LyCORIS/diffusers/Fun LoRA/finetrainer format normalization
2. RS-LoRA alpha compensation

### SDXL
1. Ensure consistent `lora_unet_`/`lora_te1_`/`lora_te2_` prefixes
2. Handle diffusers-format block naming variants

## Z-Image Re-Fusion at Apply Time

After merging at the `to_q`/`to_k`/`to_v` level, the merged patches must be re-fused to `qkv` before applying to a Z-Image model. This happens in `_process_prefix()`:
- Detect groups of `to_q`/`to_k`/`to_v` patches for the same layer
- Concatenate into single `qkv` patch
- Remap `to_out.0` -> `out`

## Data Flow

```
LoRA stack input
    |
_normalize_stack()
    |-- Load/unpack each LoRA
    |-- if normalize_keys == "enabled":
    |     |-- _detect_architecture(first_lora_sd)
    |     +-- For each LoRA: _normalize_keys(lora_sd, arch)
    +-- Return normalized dicts
    |
Pass 1: Analysis (unchanged)
    |
Pass 2: Merge (unchanged)
    |
_process_prefix() / patch application
    |-- if arch == "zimage": re-fuse to_q/to_k/to_v -> qkv
    +-- else: apply normally
    |
Patched model output
```

## Node Input Change

Add to `LoRAOptimizer.INPUT_TYPES` optional:
```python
"normalize_keys": (["disabled", "enabled"], {"default": "disabled"})
```

## Report Change

New line at top of analysis report:
```
Architecture: Z-Image Turbo (auto-detected)
Key normalization: enabled (3 LoRAs normalized)
```

## Files Changed

| File | Change |
|------|--------|
| `lora_optimizer.py` | Add `_detect_architecture()`, `_normalize_keys()` to `_LoRAMergeBase`. Modify `_normalize_stack()`. Modify `_process_prefix()` for Z-Image re-fusion. Add `normalize_keys` input. |

## Design Decisions

1. **Disabled by default**: Normalization is opt-in to avoid changing behavior for existing users
2. **Split QKV, not fuse**: Splitting fused QKV into separate Q/K/V preserves per-component granularity for conflict analysis. Re-fuse only at apply time.
3. **Single file change**: All changes in `lora_optimizer.py` — no new files needed
4. **Shared on `_LoRAMergeBase`**: Detection and normalization methods available to both LoRAOptimizer and WanVideoLoRAOptimizer
