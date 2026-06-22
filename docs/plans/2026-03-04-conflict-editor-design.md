# LoRA Conflict Editor Node — Design

## Overview

An advanced node that sits between the stacker and optimizer. It analyzes pairwise conflicts between LoRAs, auto-suggests per-LoRA conflict modes, and lets users manually override both conflict modes and merge strategy. Pass-through design: enriches the LORA_STACK with resolved settings.

## Position in Workflow

Stacker → **LoRA Conflict Editor** → Optimizer

## Inputs

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `lora_stack` | LORA_STACK | required | From any stacker node |
| `merge_strategy` | dropdown | `auto` | `auto` / `ties` / `weighted_average` / `weighted_sum` |
| `conflict_mode_1..10` | dropdown | `auto` | `auto` / `all` / `low_conflict` / `high_conflict` |

`conflict_mode_i` maps to LoRAs by stack order (1st LoRA = `conflict_mode_1`).

## Outputs

| Output | Type | Notes |
|--------|------|-------|
| `lora_stack` | LORA_STACK | Enriched with resolved `conflict_mode` per entry |
| `analysis_report` | STRING | Pairwise conflicts, per-LoRA stats, applied settings |
| `merge_strategy` | STRING | Resolved strategy for optimizer to consume |

## Override Logic (Manual-First)

- `auto`: node computes suggestion each run, writes to **output stack only**, never modifies the widget
- `all` / `low_conflict` / `high_conflict`: user's explicit choice, passed through unchanged
- Widget values are never overwritten — no re-run surprises

Same for `merge_strategy`: `auto` runs heuristic, explicit values pass through.

## Execution Logic

1. Normalize incoming stack (load LoRAs, cache)
2. Run pairwise conflict analysis (sign conflicts, cosine similarity) — reuse optimizer's `_sample_conflict` logic
3. For each LoRA position `i`:
   - If `conflict_mode_i == "auto"`: compute suggestion from conflict stats, bake into output
   - Otherwise: use the explicit value
4. For `merge_strategy`:
   - If `auto`: auto-select based on avg conflict ratio (same heuristic as optimizer)
   - Otherwise: pass through
5. Build analysis report
6. Output enriched stack + report + resolved strategy

## Auto-Suggestion Heuristic

For each LoRA, compute average pairwise sign conflict ratio with all others:
- < 15% avg conflict → suggest `all` (compatible, no filtering needed)
- 15–40% avg conflict → suggest `low_conflict` (conservative, safe)
- > 40% avg conflict → suggest `high_conflict` (very different, let it dominate contested regions)

Report shows reasoning for each suggestion.

## Optimizer Integration

Add optional `merge_strategy_override` input (STRING) to the optimizer. When connected and non-empty, it overrides auto-detection. When not connected, existing behavior is unchanged.

## JS Widget Visibility

Show all 10 `conflict_mode_i` dropdowns. Hide unused ones. Since stack size is only known at execution time, two options:
- Show all 10 always (unused ones are harmless)
- After first run, use PromptServer to hide extras (nice but complex)

Start with showing all 10. Can refine later.

## Analysis Report Format

```
============ LORA CONFLICT EDITOR ============

--- Stack (3 LoRAs) ---
  1. style_lora.safetensors (strength: 1.0)
  2. character_lora.safetensors (strength: 0.8)
  3. lighting_lora.safetensors (strength: 0.6)

--- Pairwise Conflicts ---
  style × character:
    Overlap: 12,450 positions
    Sign conflicts: 3,112 (25.0%)
    Cosine similarity: 0.62
  style × lighting:
    Overlap: 8,200 positions
    Sign conflicts: 820 (10.0%)
    Cosine similarity: 0.85
  character × lighting:
    Overlap: 9,100 positions
    Sign conflicts: 3,640 (40.0%)
    Cosine similarity: 0.31

--- Applied Conflict Modes ---
  1. style_lora: all (auto-suggested — low avg conflict 17.5%)
  2. character_lora: high_conflict (manual override)
  3. lighting_lora: low_conflict (auto-suggested — high avg conflict 25.0%)

--- Merge Strategy ---
  Applied: ties (auto — avg conflict 25.0% > threshold)
```
