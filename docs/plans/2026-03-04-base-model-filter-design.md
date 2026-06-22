# Design: Base Model Filter for LoRAStackDynamic

## Problem

Users with large LoRA collections (mixed SDXL, Flux, Wan, etc.) see all LoRAs in the dropdown regardless of which model they're using. ComfyUI-Lora-Manager already categorizes LoRAs by base model type — we can leverage its API to filter the dropdown.

## Solution

Add a `base_model_filter` dropdown to `LoRAStackDynamic` that queries Lora Manager's API to filter LoRA dropdowns. Hidden if Lora Manager is not installed.

### Data flow

1. Node created → JS calls `GET /api/lm/loras/base-models` → populates filter dropdown with `["All"] + model types`
2. User selects a model type (e.g., "Wan") → JS calls `GET /api/lm/loras/list?base_model=Wan&page_size=10000` → gets filtered LoRA filenames
3. JS updates all `lora_name_N` COMBO widget options with filtered list
4. User selects "All" → JS restores original full LoRA list

### Graceful degradation

- Lora Manager not installed: API call fails → filter widget hidden, dropdowns show all LoRAs
- API call fails mid-session: revert to full LoRA list, no error
- Filter set to "All": show full LoRA list

## Files to modify

- `lora_optimizer.py` — add `base_model_filter` widget to `LoRAStackDynamic.INPUT_TYPES()` (hidden input, not used in execution)
- `js/lora_stack_dynamic.js` — add filter logic: detect Lora Manager, populate filter, intercept changes, update LoRA dropdowns

## Implementation details

### Python side

Add to `LoRAStackDynamic.INPUT_TYPES()` optional inputs:
```python
"base_model_filter": (["All"], {
    "default": "All",
    "tooltip": "Filter LoRA list by base model type. Requires ComfyUI-Lora-Manager."
})
```

The `build_stack` method ignores this widget — it's purely for UI filtering. The options list starts with just `["All"]` and is dynamically populated by the JS extension.

### JS side (in lora_stack_dynamic.js)

On `LoRAStackDynamic` node creation:

1. **Detect Lora Manager**: `fetch("/api/lm/loras/base-models")`
   - Success → populate `base_model_filter` widget options with `["All", ...names]`
   - Failure → hide the `base_model_filter` widget

2. **Intercept filter changes**: same `Object.defineProperty` pattern already used for `mode` and `lora_count`

3. **On filter change**:
   - If "All" → restore original LoRA list (cached from initial widget options)
   - Otherwise → `fetch("/api/lm/loras/list?base_model=X&page_size=10000")` → extract filenames → update all `lora_name_N` widget options
   - Preserve currently selected values if they exist in the filtered list

4. **Cache**: store the full LoRA list on first load so "All" restoration is instant
