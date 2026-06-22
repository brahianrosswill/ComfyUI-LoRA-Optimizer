# Per-Prefix Adaptive Merge Strategy

## Problem

The optimizer currently computes per-prefix conflict stats in Pass 1 but averages them into a single global conflict ratio. One global decision (TIES vs weighted_average) is applied to all prefixes. This loses important information:

- Two LoRAs may only overlap in certain blocks (e.g., attention layers 4-7). Non-overlapping prefixes have zero conflict but get treated the same as high-conflict ones.
- When the global average falls below the TIES threshold, genuinely conflicting prefixes get simple averaging — conflicts go unresolved.
- When the global average is above the threshold, non-overlapping prefixes get TIES trimming — their weights are unnecessarily reduced.

## Solution

Per-prefix adaptive strategy: each prefix picks its own merge mode based on its own conflict data.

### Decision table (per prefix)

| Condition | Strategy |
|-----------|----------|
| Only 1 LoRA touches this prefix | `weighted_sum` — full strength, no dilution |
| 2+ LoRAs, conflict <= 25% | `weighted_average` — compatible, simple merge |
| 2+ LoRAs, conflict > 25% | `ties` — resolve conflicts with trim/elect/merge |

Density and sign method are also computed per-prefix when TIES is selected.

### Toggle

New input `optimization_mode` with values `per_prefix` (default) and `global` (original behavior).

### Architecture

```
Pass 1 → per-prefix stats stored in dict
          ↓
        global stats still computed for report + global mode fallback
          ↓
Pass 2 → if per_prefix: each prefix looks up its own stats, picks strategy
          if global: same mode/density/sign_method for all (original behavior)
```

### What stays the same

- Two-pass streaming architecture (no extra memory)
- Auto-strength (LoRA-level, not per-prefix)
- Cache mechanism (add optimization_mode to cache key)
- All merge modes unchanged
- Report format (additive sections)

### Report additions

```
--- Per-Prefix Strategy ---
  weighted_sum (single LoRA):      120 prefixes
  weighted_average (low conflict):  32 prefixes
  ties (high conflict):             40 prefixes
```

### Files modified

Only `lora_optimizer.py`.
