# LoRA AutoTuner & Merge Selector — Design

## Overview

Two new nodes that automatically sweep all merge parameters, score configurations using a hybrid heuristic+measurement approach, and output the best merge — with an optional selector node to pick alternatives.

## Nodes

### LoRAAutoTuner — "LoRA AutoTuner"

**Inherits**: `LoRAOptimizerBase`

**Inputs (required)**:
- `model` — MODEL
- `lora_stack` — LORA_STACK
- `output_strength` — float (0.0–10.0, default 1.0)

**Inputs (optional)**:
- `clip` — CLIP
- `clip_strength_multiplier` — float (0.0–10.0, default 1.0)
- `top_n` — int (1–10, default 3)

**Outputs**:
- `MODEL` — merged model using #1 ranked config
- `CLIP` — merged clip (if clip input provided)
- `REPORT` — STRING, ranked table of top-N configs with scores
- `TUNER_DATA` — custom type, top-N configs + analysis cache

**Behavior**:
1. Run Pass 1 analysis once (existing `_analyze_prefix`)
2. Score ~500–600 parameter combos via Phase 1 heuristic
3. Merge top-N candidates via Phase 2, measure output quality
4. Apply #1 config to model → output MODEL/CLIP
5. Pack all top-N configs + scores + analysis cache into TUNER_DATA

### LoRAMergeSelector — "Merge Selector"

**Inherits**: `LoRAOptimizerBase`

**Inputs (required)**:
- `model` — MODEL (original, unpatched)
- `lora_stack` — LORA_STACK
- `tuner_data` — TUNER_DATA
- `selection` — int (1-based index into top-N)

**Inputs (optional)**:
- `clip` — CLIP

**Outputs**:
- `MODEL` — merged model using selected config
- `CLIP` — merged clip
- `REPORT` — STRING

**Behavior**:
1. Validate `lora_hash` in tuner_data matches current lora_stack
2. Read config at `tuner_data.top_n[selection - 1]`
3. Run Pass 2 merge only (skip analysis, use cached data)
4. Output merged model

## Workflow

```
Simple:    [Stack] -> [AutoTuner] -> [MODEL] -> [Sampler]

Override:  [Stack] -> [AutoTuner] -> [TUNER_DATA] -> [Selector] -> [MODEL]
                         |                               ^
                         +-> [MODEL (unpatched)] --------+
```

## Parameter Grid

| Parameter | Values | Count |
|-----------|--------|-------|
| merge_mode | weighted_average, slerp, consensus, ties | 4 |
| sparsification | disabled, dare, della | 3 |
| sparsification_density | 0.5, 0.7, 0.9 | 3 (when sparsification != disabled) |
| dare_dampening | 0.0, 0.3, 0.6 | 3 (when sparsification == dare) |
| merge_quality | standard, enhanced, maximum | 3 |
| auto_strength | enabled, disabled | 2 |
| optimization_mode | per_prefix, global | 2 |

Nonsensical combos pruned (density/dampening only when applicable). Estimated ~500–600 valid combos.

## Hybrid Scoring

### Phase 1 — Heuristic (no merge, milliseconds per combo)

Composite score from Pass 1 analysis metrics:

```
score = w1 * conflict_score      # lower sign conflicts = better
      + w2 * alignment_score     # cosine sim alignment with chosen mode
      + w3 * energy_score        # norm preservation prediction
      + w4 * mode_fit_score      # how well mode matches data profile
```

Scoring intuitions:
- **consensus** high when cos_sim > 0.5, conflict < 15%
- **slerp/weighted_avg** high when LoRAs near-orthogonal
- **ties** high when conflict > 25%
- **sparsification** bonus for heavy-tailed magnitude distributions
- **enhanced/maximum quality** bonus proportional to conflict level
- **auto_strength** bonus when magnitude_ratio > 2×

### Phase 2 — Merge-and-Measure (top-N only, ~30–60s each)

Actual merge, then measure:
- **Norm preservation ratio** — merged vs input energy
- **Effective rank** — spectral rank of merged tensors
- **Weight distribution stats** — kurtosis, sparsity

Phase 2 scores replace Phase 1 heuristics for final ranking.

## TUNER_DATA Format

```python
{
    "version": 1,
    "top_n": [
        {
            "rank": 1,
            "score_heuristic": 0.87,
            "score_measured": 0.92,
            "config": {
                "merge_mode": "slerp",
                "sparsification": "dare",
                "sparsification_density": 0.7,
                "dare_dampening": 0.0,
                "merge_quality": "enhanced",
                "auto_strength": "enabled",
                "optimization_mode": "per_prefix",
                "adjusted_strengths": [0.8, 1.0, 0.6],
            },
            "metrics": {
                "norm_preservation": 0.73,
                "avg_conflict_ratio": 0.12,
                "avg_cosine_sim": 0.45,
                "effective_rank_mean": 28.3,
            },
        },
        # ... rank 2, 3, etc.
    ],
    "analysis_cache": { ... },
    "lora_hash": "...",
}
```

## Report Format

```
══════════════════════════════════════════
  LoRA AutoTuner Results
══════════════════════════════════════════

  Analysis Summary:
    LoRAs: 3 | Prefixes: 142 | Avg conflict: 18.2%
    Avg cosine similarity: 0.31 | Magnitude ratio: 2.4x

  ──────────────────────────────────────
  Top 3 Configurations
  ──────────────────────────────────────

  #1 ★ (applied to output)          Score: 0.92
    Mode: slerp | Quality: enhanced
    Sparsification: dare (density=0.7, dampening=0.0)
    Auto-strength: enabled | Optimization: per_prefix
    Norm preservation: 0.73 | Effective rank: 28.3

  #2                                  Score: 0.88
    Mode: consensus | Quality: standard
    Sparsification: disabled
    Auto-strength: enabled | Optimization: per_prefix
    Norm preservation: 0.81 | Effective rank: 25.1

  #3                                  Score: 0.85
    Mode: ties (d=0.6) | Quality: enhanced
    Sparsification: della (density=0.5, dampening=0.3)
    Auto-strength: disabled | Optimization: global
    Norm preservation: 0.68 | Effective rank: 30.7

  Suggested max output_strength: 1.37

  To use a different config: connect TUNER_DATA
  to a Merge Selector node and set selection=N
══════════════════════════════════════════
```
