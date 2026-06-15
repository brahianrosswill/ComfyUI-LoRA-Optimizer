# How the LoRA Optimizer Works

This page explains the full pipeline — from building a LoRA stack through analysis, strategy selection, merge execution, and post-processing. Each section builds on the previous one to show how algorithms interact and why each step exists.

---

## Table of Contents

- [The Problem](#the-problem)
- [Pipeline Overview](#pipeline-overview)
- [Step 1 — Building the LoRA Stack](#step-1--building-the-lora-stack)
- [Step 2 — Key Normalization](#step-2--key-normalization)
- [Step 3 — Pass 1: Analysis](#step-3--pass-1-analysis)
  - [Computing Weight Diffs](#computing-weight-diffs)
  - [Conflict Detection](#conflict-detection)
  - [Cosine Similarity](#cosine-similarity)
  - [Auto-Strength Calibration](#auto-strength-calibration)
  - [Suggested Max Output Strength](#suggested-max-output-strength)
- [Step 4 — Strategy Selection](#step-4--strategy-selection)
  - [Per-Prefix Decision Tree](#per-prefix-decision-tree)
  - [Global Merge Mode Selection](#global-merge-mode-selection)
  - [Architecture Presets](#architecture-presets)
  - [Behavior Profiles](#behavior-profiles)
  - [Sign Method Selection](#sign-method-selection)
- [Step 5 — Pass 2: Merge Execution](#step-5--pass-2-merge-execution)
  - [Preprocessing: Sparsification](#preprocessing-sparsification)
  - [Quality Enhancements](#quality-enhancements)
  - [Merge Algorithms](#merge-algorithms)
  - [How Algorithms Interact](#how-algorithms-interact)
- [Step 6 — Post-Processing](#step-6--post-processing)
  - [SVD Patch Compression](#svd-patch-compression)
  - [Dtype Downcasting](#dtype-downcasting)
  - [VRAM Budget & Patch Placement](#vram-budget--patch-placement)
  - [Patch Application](#patch-application)
- [Algorithm Interaction Map](#algorithm-interaction-map)
- [Node Reference](#node-reference)
- [Appendix: AutoTuner](#appendix-autotuner)
- [Appendix: Conflict Editor](#appendix-conflict-editor)
- [Appendix: Per-LoRA Controls](#appendix-per-lora-controls)
- [Appendix: Output Nodes](#appendix-output-nodes)
- [Appendix: Supported Architectures](#appendix-supported-architectures)

---

## The Problem

When you stack multiple LoRAs in ComfyUI, each one is applied independently — their effects simply add together on the base model weights. This works fine when LoRAs touch different parts of the model, but breaks down when they overlap:

- **Oversaturation** — Two style LoRAs at strength 1.0 can produce blown-out results because their effects compound in the same weight regions
- **Sign conflicts** — LoRA A wants to push a weight positive while LoRA B pushes it negative. Simple addition cancels out both contributions
- **Noise amplification** — Minor disagreements across thousands of parameters accumulate into visible artifacts

The LoRA Optimizer solves this by analyzing _where_ LoRAs conflict and applying the right merge strategy to each region independently.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LORA OPTIMIZER PIPELINE                            │
│                                                                             │
│  LoRA Stack ──► Key Normalization ──► Pass 1: Analysis ──► Strategy Selection│
│                                           │                      │          │
│                                           ▼                      ▼          │
│                                      Pass 2: Merge Execution                │
│                                           │                                 │
│                                           ▼                                 │
│                                    Post-Processing                          │
│                              (SVD compress, dtype cast,                     │
│                               VRAM budget placement)                        │
│                                           │                                 │
│                                           ▼                                 │
│                            MODEL + CLIP + Report + LORA_DATA                │
└─────────────────────────────────────────────────────────────────────────────┘
```

The pipeline has two physical passes through the weights but several logical stages. This two-pass design keeps peak memory close to “one target group at a time,” but the exact peak still depends on the largest layer being processed, how many LoRAs overlap on it, and whether extra quality/compression steps are enabled.

---

## Step 1 — Building the LoRA Stack

Before anything is analyzed or merged, you assemble a list of LoRAs using **LoRA Stack** or **LoRA Stack (Dynamic)** nodes.

Each entry in the stack contains:
- **LoRA file** — the `.safetensors` file to load
- **Strength** — how much this LoRA should contribute (default 1.0)
- **Conflict mode** — controls where this LoRA's contributions are kept (`all`, `low_conflict`, `high_conflict`)
- **Key filter** — controls which weight prefixes this LoRA contributes to (`all`, `shared_only`, `unique_only`)

The stack is just a list — no analysis happens yet. LoRAs are loaded lazily when the optimizer needs them.

### LoRA Stack (Dynamic)

The Dynamic variant provides 1–10 adjustable LoRA slots in a single node, replacing the need to chain multiple Stack nodes. It has two visibility modes controlled by `settings_visibility`:

| Mode | Per-LoRA Controls |
|------|-------------------|
| **Simple** | Single `strength` slider per LoRA — clean and beginner-friendly |
| **Advanced** | Separate `model_strength` and `clip_strength`, plus `conflict_mode` and `key_filter` per LoRA |

It also supports two input modes via `input_mode`:
- **Dropdown** — standard file picker from your LoRAs folder
- **Text** — type a name or path; auto-matches against installed LoRAs

An optional `base_model_filter` integrates with ComfyUI-Lora-Manager to filter LoRAs by base model (when that extension is installed).

### Stack Formats

The optimizer accepts stacks from multiple sources:
- **Native format** — dict-based entries from LoRA Stack / LoRA Stack (Dynamic)
- **Tuple format** — `(lora_name, model_strength, clip_strength)` from Efficiency Nodes, Comfyroll, and other packs

Both are normalized internally before analysis begins.

---

## Step 2 — Key Normalization

> _Controlled by: `normalize_keys` (disabled / enabled)_

Different LoRA trainers name their weights differently. Kohya uses underscores (`lora_transformer_double_blocks_0_...`), AI-Toolkit uses dots (`transformer.transformer_blocks.0...`), diffusers uses yet another format. If you mix LoRAs from different trainers, the optimizer sees _no key overlap_ — it thinks they touch completely different parts of the model and can't merge them.

Key normalization solves this by:

1. **Detecting the model architecture** from LoRA key patterns (FLUX, SDXL, Wan, Z-Image, LTX, Qwen)
2. **Remapping all keys** to a canonical format so LoRAs from any trainer can be compared

### Special Case: Z-Image QKV Fusion

Z-Image LoRAs often fuse Q, K, and V projections into a single `attention.qkv` weight. The normalizer:
1. Detects fused QKV (lora_up shape = `[3×out_features, rank]`)
2. **Splits** into separate `to_q`, `to_k`, `to_v` components
3. Allows per-component conflict analysis during the merge
4. **Re-fuses** back to native `attention.qkv` format after merging

This splitting is critical — Q, K, and V weights serve different purposes. Two LoRAs might agree on Q weights but conflict on V weights. Without splitting, the optimizer would average their conflicts and get a suboptimal result for both.

### Special Case: WanVideo `_orig_mod` Keys

WanVideo models compiled with `torch.compile` prefix their keys with `_orig_mod.`. The normalizer detects this mismatch and strips the prefix so LoRA keys can match correctly against model weights.

---

## Step 3 — Pass 1: Analysis

Pass 1 iterates through every unique weight prefix (e.g., `input_blocks.4`, `middle_block.1`) across all LoRAs. For each prefix, it computes statistics and then **immediately discards the full weight diffs**. Only lightweight scalars survive — this is what keeps memory usage low.

### Computing Weight Diffs

For each LoRA at each prefix, the optimizer computes the **full-rank diff** — the actual change this LoRA would make to the base model weights:

```
diff = lora_up @ lora_down × (alpha / rank) × strength
```

For LoCon layers (with `lora_mid`), the reconstruction handles the additional intermediate matrix.

These diffs are the ground truth — they show exactly what each LoRA wants to do to each weight. All analysis operates on these diffs, not on the raw low-rank factors.

### Conflict Detection

For each pair of LoRAs that overlap at a given prefix, the optimizer measures **sign conflict ratio**: the fraction of weight positions where the two LoRAs push in opposite directions.

```
conflict_ratio = count(sign(diff_A) ≠ sign(diff_B)) / total_overlapping_positions
```

A high conflict ratio (>25%) means the LoRAs are fighting each other at this prefix — they want opposite things. A low ratio means they mostly agree.

The conflict ratio is measured per-prefix, not globally. Two LoRAs might have 5% conflict in attention layers but 40% conflict in feed-forward layers. The optimizer handles each independently.

### Cosine Similarity

Beyond sign conflicts, the optimizer measures **cosine similarity** between LoRA diff vectors:

```
cos_sim = dot(diff_A, diff_B) / (||diff_A|| × ||diff_B||)
```

This captures directional alignment:
- **cos ≈ 1** → LoRAs point the same direction (reinforcing)
- **cos ≈ 0** → LoRAs are orthogonal (independent)
- **cos ≈ -1** → LoRAs point opposite directions (canceling)

Cosine similarity is used by:
- **Auto-strength calibration** — to compute interference energy
- **Global mode selection** — consensus and SLERP are preferred when similarity is high
- **Behavior profiles** — orthogonal detection triggers SLERP upgrade

### Auto-Strength Calibration

> _Controlled by: `auto_strength` (enabled / disabled)_

When enabled, the optimizer reduces all LoRA strengths proportionally to prevent oversaturation from stacking. The algorithm is **interference-aware energy normalization**:

1. Measure pairwise cosine similarity between all LoRAs
2. Compute total combined energy accounting for directional alignment:
   ```
   ||Σ v_i||² = Σ||v_i||² + 2 × Σ ||v_i|| × ||v_j|| × cos(v_i, v_j)
   ```
3. Scale all strengths so combined energy = energy of the strongest single LoRA alone

The key insight is that aligned LoRAs (cos ≈ 1) produce more combined energy than orthogonal ones (cos ≈ 0), which produce more than opposing ones (cos ≈ -1). The calibration adapts to the actual geometry:

| Scenario | Scale Factor |
|----------|-------------|
| 2 aligned LoRAs (cos≈1) | ~0.50 each |
| 2 orthogonal LoRAs (cos≈0) | ~0.71 each |
| 2 opposing LoRAs (cos≈-1) | ~1.0 each |

Original strength _ratios_ are always preserved — only the overall scale changes.

### Suggested Max Output Strength

The analysis report now includes a **suggested maximum output strength** — automatically calculated from the magnitude ratio of the merged LoRAs:

```
suggested_max = min(1.0 / norm_ratio, architecture_cap)
```

The cap varies by architecture preset (3.0 for UNet, 5.0 for DiT, 3.0 for LLM). This helps you avoid oversaturation by showing a safe strength ceiling. It appears in the report alongside the current `output_strength` value.

---

## Step 4 — Strategy Selection

After analysis, the optimizer decides how to merge each prefix. This is where `optimization_mode` matters:

| Mode | Behavior |
|------|----------|
| `per_prefix` (default) | Each weight prefix gets its own strategy based on local conflict data |
| `global` | One strategy for all prefixes, chosen from aggregate statistics |
| `additive` | Force simple addition everywhere |

### Per-Prefix Decision Tree

For each prefix independently:

```
                    How many LoRAs touch this prefix?
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 Only 1              2 or more
                    │                   │
              weighted_sum         Sign conflict ratio?
           (full strength,              │
            no dilution)      ┌─────────┴─────────┐
                              │                   │
                           ≤ 25%               > 25%
                              │                   │
                      weighted_average          TIES
                       (compatible,        (resolve conflicts
                        blend fairly)    via trim/elect/merge)
```

This is the fundamental innovation: non-overlapping regions get 100% of their LoRA's effect (no dilution from averaging), while genuinely conflicting regions get proper TIES conflict resolution.

### Global Merge Mode Selection

When using `global` optimization (or as a fallback), the optimizer picks a single mode based on aggregate statistics. The `full` strategy set adds more nuanced selection:

```
                        Aggregate conflict ratio
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 ≤ 25%               > 25%
                    │                   │
           Cosine similarity?        TIES
                    │
          ┌─────┬──┴──┬─────┐
          │     │     │     │
       > 0.7  0.3-0.7  < 0.3  orthogonal
          │     │     │     │
      consensus  weighted   SLERP
                 average   (full)
```

### Architecture Presets

> _Controlled by: `architecture_preset` (auto / sd_unet / dit / acestep_dit / llm)_

Different model architectures have different numerical characteristics. The architecture preset tunes the optimizer's internal thresholds — density ranges, noise floors, and strength caps — to match:

| Preset | Architectures | Density Range | Noise Floor | Max Strength Cap |
|--------|--------------|---------------|-------------|-----------------|
| `sd_unet` | SD 1.5, SDXL | 0.1 – 0.9 | 10% | 3.0 |
| `dit` | Flux, Wan, Z-Image, LTX, Ideogram 4, HunyuanVideo | 0.4 – 0.95 | 5% | 5.0 |
| `acestep_dit` | ACE-Step (music DiT) | 0.4 – 0.95 | 5% | 5.0 |
| `llm` | Qwen, LLaMA | 0.1 – 0.8 | 15% | 3.0 |
| `auto` (default) | Auto-detected from LoRA keys | Selected automatically | — | — |

DiT models are denser and tolerate higher strengths. UNet models are sparser. LLM models need tighter density bounds and lower strength caps. ACE-Step uses DiT-class thresholds but with a wider orthogonal band and higher TIES threshold, since music LoRAs are conflict-prone and need a gentler merge to keep the singing voice intact. **HunyuanVideo** uses the `dit` preset but is not in the auto-detector — select `dit` manually.

The architecture preset is **orthogonal** to the behavior profile — the preset controls _numeric thresholds_, while the profile controls _strategy selection logic_.

### Behavior Profiles

> _Controlled by: `strategy_set` (full / no_slerp / basic)_

The behavior profile controls _which_ merge strategies are considered during auto-selection:

- **`full`** (default) — Full detection: consensus for high similarity, SLERP for orthogonal, weighted_average otherwise
- **`no_slerp`** — Same detection but SLERP never auto-selected; stays as weighted_average
- **`basic`** — Pre-1.2 behavior: only TIES vs weighted_average, no SLERP or consensus

### Sign Method Selection

For TIES merging, the optimizer picks how to resolve sign conflicts:

| Condition | Method | Logic |
|-----------|--------|-------|
| Magnitude ratio > 2× | `total` | Stronger LoRA dominates the vote |
| Magnitude ratio ≤ 2× | `frequency` | Equal vote per LoRA |

---

## Step 5 — Pass 2: Merge Execution

Pass 2 walks through every prefix again, recomputes the diffs, and applies the selected strategy. Several processing stages happen in sequence for each prefix:

```
┌──────────────────────────────────────────────────────────────────┐
│                    PER-PREFIX MERGE PIPELINE                      │
│                                                                  │
│  Recompute      Quality           Sparsify       Apply Merge     │
│    Diffs    ──► Enhancements  ──► (if enabled) ──► Algorithm     │
│                (if enhanced/                         │            │
│                 maximum)                             ▼            │
│                                              Merged Patch        │
│                                                  │               │
│                                                  ▼               │
│                                           SVD Compress           │
│                                          (if enabled)            │
│                                                  │               │
│                                                  ▼               │
│                                        Dtype Downcast            │
│                                                  │               │
│                                                  ▼               │
│                                      VRAM Budget Placement       │
│                                                  │               │
│                                                  ▼               │
│                                           Free & Continue        │
└──────────────────────────────────────────────────────────────────┘
```

### Preprocessing: Sparsification

> _Controlled by: `sparsification`, `sparsification_density`, `dare_dampening`_

Sparsification reduces parameter interference by zeroing out a fraction of each LoRA's diff before merging. Two algorithms are available, each with a standard and conflict-aware variant:

#### DARE (Drop And REscale)

1. Generate a random binary mask where each weight has probability `density` of being kept
2. Zero out masked positions
3. Rescale survivors by `1/density` to preserve expected magnitude

**DAREx enhancement:** The `dare_dampening` parameter (0.0–1.0) interpolates the rescaling factor toward 1.0, reducing noise amplification from aggressive rescaling. At 0.0: standard DARE. Higher values: less noise at the cost of slightly biased magnitudes.

#### DELLA (magnitude-aware dropout)

1. Rank weights per-row by absolute magnitude
2. Low-magnitude weights get higher drop probability
3. High-magnitude weights are preserved
4. Rescale survivors to maintain expected value

DELLA is more surgical than DARE — it preferentially preserves the weights that matter most.

#### Conflict-Aware Variants

Standard sparsification drops weights everywhere, including positions where only one LoRA contributes or where LoRAs agree. This destroys useful signal.

Conflict-aware variants (`dare_conflict`, `della_conflict`) compute a sign-conflict mask first:
```
For each position:
  if 2+ LoRAs have opposite signs → sparsify (this is interference)
  if LoRAs agree or only 1 present → keep full weight (this is useful signal)
```

Result: interference is reduced without sacrificing unique features.

#### Interaction with TIES

When the merge strategy is TIES:
- **Sparsification replaces the TIES trim step.** Both achieve sparsification (removing low-value weights), so running both would be redundant. The sparsification density parameter controls how aggressive the combined effect is.

When the merge strategy is anything else:
- **Sparsification runs as preprocessing** before the merge algorithm sees the diffs.

### Quality Enhancements

> _Controlled by: `merge_refinement` (none / refine / full)_

Quality enhancements are additional processing steps applied before or during the merge. They compose with sparsification — sparsification runs first, then quality enhancements process the (possibly sparsified) diffs.

#### None (baseline)

No additional processing. Element-wise sign voting in TIES mode.

#### Refine (adds three techniques)

**1. DO-Merging (Decouple & Orthogonalize)**
- Orthogonalizes LoRA direction vectors via Modified Gram-Schmidt
- Preserves original magnitudes while reducing directional interference
- LoRAs that were partially aligned become truly independent

**2. Column-Wise Conflict Resolution** (replaces element-wise)
- Instead of each weight position voting on sign direction independently, entire output neurons (rows) vote as a unit
- Preserves structural coherence — a neuron's weights work together, so their signs should be resolved together
- Falls back to element-wise for 1D tensors (biases, norms)

**3. TALL-Masks (Task-Aware weight protection)**
- Identifies "selfish" weights — positions where one LoRA dominates and others contribute little
- Separates selfish contributions from the consensus
- After merging, adds selfish contributions back, protecting unique features from being averaged away

#### Full (adds SVD alignment)

Everything from Refine, plus:

**KnOTS SVD Alignment**
- Concatenates all LoRA diffs column-wise: `M = [diff_1 | diff_2 | ... | diff_N]`
- Computes truncated SVD (rank ≤ 256)
- Reconstructs each diff in the shared singular value basis
- Makes diffs more directly comparable by aligning their representation spaces
- Falls back to CPU on GPU OOM; skips gracefully if both fail

### Merge Algorithms

After preprocessing and quality enhancements, the actual merge algorithm runs. The algorithm was selected in Step 4.

#### Weighted Sum

```
output = Σ (strength_i × diff_i)
```

The simplest merge: plain addition. Used when only one LoRA touches a prefix (no conflict possible). Also available as `additive` optimization mode to force it everywhere.

**Properties:**
- Preserves every contribution exactly
- No information loss
- Fully compressible by SVD (linear operation)

#### Weighted Average

```
output = Σ (strength_i / Σ_strengths × diff_i)
```

Normalized addition. Fair blending when LoRAs agree on direction.

**Properties:**
- Prevents magnitude inflation from stacking
- Works well for compatible LoRAs (low conflict)
- Fully compressible by SVD (linear operation)

#### TIES (Trim, Elect Sign, Disjoint Merge)

A three-step process specifically designed for sign conflicts:

```
Step 1 — Trim: Keep only top-k% weights by magnitude
         (or: use DARE/DELLA sparsification instead)
              │
              ▼
Step 2 — Elect Sign: For each position, vote on the majority sign direction
         ┌─ frequency: each LoRA gets one vote
         └─ total: votes weighted by magnitude
              │
              ▼
Step 3 — Disjoint Merge: Average only the LoRAs that agree with the elected sign
         Contributors opposing the majority are excluded for this position
```

**Properties:**
- Resolves sign conflicts without cancellation
- Nonlinear (trim + sign election) → produces full-rank results
- Lossy SVD compression (can't perfectly capture nonlinear output)

#### SLERP (Spherical Linear Interpolation)

Magnitude-preserving directional blend:

1. For 2 LoRAs: standard SLERP formula along the geodesic on a hypersphere
2. For 3+ LoRAs: iterative pairwise SLERP, sorted by descending weight (strongest anchors direction)
3. Falls back to linear interpolation for near-parallel vectors (where SLERP is numerically unstable)
4. Final magnitude corrected to match weighted average of input norms

**Properties:**
- Preserves weight magnitude (no inflation or deflation)
- Smooth directional interpolation
- Best for low-conflict, similar-magnitude LoRAs

#### Consensus Merging

Three-stage algorithm for highly similar LoRAs:

```
Stage 1 — Fisher-Proxy Importance Weighting:
    weight each parameter by |diff|² as importance proxy
    numerator = Σ(diff_i × strength_i × |diff_i|²)
    denominator = Σ(|strength_i| × |diff_i|²)
    output = numerator / denominator

Stage 2 — MAGIC Calibration:
    rescale merged result so L2 norm matches
    weighted average of input norms

Stage 3 — MonoSoup Spectral Cleanup:
    SVD-based entropy filtering on 2D+ tensors
    keeps effective rank based on singular value entropy
    removes noise while preserving signal
```

**Properties:**
- Best for highly similar LoRAs (high cosine similarity)
- Importance-weighted → parameters that matter more get more influence
- Spectral cleanup removes merge noise

### How Algorithms Interact

The key to understanding the optimizer is knowing that these algorithms **compose in a fixed order**, not as alternatives:

```
                     ┌──────────────────────────────────────────────┐
                     │            COMPOSITION ORDER                 │
                     │                                              │
                     │  1. Key Normalization (if enabled)           │
                     │        ↓                                     │
                     │  2. Architecture Preset applied              │
                     │        ↓ (sets density/strength thresholds)  │
                     │  3. Auto-Strength (if enabled)               │
                     │        ↓ (adjusts strength scalars)          │
                     │  4. Quality: KnOTS alignment (if full)       │
                     │        ↓ (transforms diff vectors)           │
                     │  5. Quality: DO-Merging (if refine+)         │
                     │        ↓ (orthogonalizes directions)         │
                     │  6. Sparsification (if enabled)              │
                     │        ↓ (zeros out weights)                 │
                     │        ├─ TIES mode? → replaces trim step    │
                     │        └─ Other modes? → preprocessing       │
                     │  7. Quality: TALL-masks (if refine+)         │
                     │        ↓ (separates selfish weights)         │
                     │  8. Merge Algorithm                          │
                     │        ↓ (combines diffs)                    │
                     │  9. TALL-masks re-add (if refine+)           │
                     │        ↓ (restores selfish weights)          │
                     │  10. SVD Compression (if enabled)            │
                     │        ↓ (reduces to low-rank)               │
                     │  11. Dtype downcast to native precision      │
                     │        ↓ (reduces memory footprint)          │
                     │  12. VRAM budget placement (GPU or CPU)      │
                     │        ↓                                     │
                     │  13. Patch application to model              │
                     └──────────────────────────────────────────────┘
```

Important interactions:

| Combination | Interaction |
|-------------|-------------|
| DARE/DELLA + TIES | Sparsification **replaces** TIES trim (no double-sparsification) |
| DARE/DELLA + other modes | Sparsification runs as **preprocessing** before merge |
| Conflict-aware + per_prefix | Conflict mask computed per-prefix using local conflict data |
| KnOTS + TIES | KnOTS aligns diffs in shared basis _before_ TIES sign voting |
| DO-Merging + any mode | Orthogonalization happens _after_ KnOTS but _before_ sparsification |
| TALL-masks + any mode | Selfish weights separated _before_ merge, added back _after_ |
| Column-wise + TIES | Replaces element-wise sign voting (structural coherence) |
| Auto-strength + everything | Only adjusts strength scalars — all algorithms see adjusted strengths |
| Architecture preset + behavior profile | Preset sets _numeric thresholds_, profile sets _strategy logic_ — independent |

---

## Step 6 — Post-Processing

### SVD Patch Compression

> _Controlled by: `patch_compression` (smart / aggressive / disabled), `svd_device` (gpu / cpu)_

After merging, each prefix produces a full-rank diff tensor. For a 4096×4096 weight, that's ~64MB per key (vs ~0.5MB for a rank-32 LoRA patch). SVD compression re-factors these back to low-rank:

```
full_rank_diff ──► truncated SVD ──► lora_up (rank R) + lora_down (rank R)
```

The compression rank is automatically set to the **sum of all input LoRA ranks**. Three rank-32 LoRAs → rank-96 compressed patch. This is enough to represent the full merge without quality loss for linear operations.

| Mode | Compresses | Quality |
|------|-----------|---------|
| `smart` (default) | weighted_sum and weighted_average prefixes only | **Lossless** — linear operations are exactly representable |
| `aggressive` | Everything including TIES | **Lossy on TIES** — nonlinear ops (trim, sign election) produce full-rank results |
| `disabled` | Nothing | No loss, but ~32× more RAM |

### Dtype Downcasting

After merge and optional SVD compression, patches are automatically downcast to the **native dtype** of the input LoRA weights (typically `float16` or `bfloat16`). Merging internally uses `float32` for numerical precision, but the final patches don't need that precision. This reduces memory footprint significantly — a `float32` patch is 2× the size of a `float16` patch.

### VRAM Budget & Patch Placement

> _Controlled by: `vram_budget` (0.0 – 1.0)_

By default (`vram_budget=0.0`), all merged patches live on CPU. The model patcher moves them to GPU on demand during sampling, which adds latency.

When `vram_budget` > 0, the optimizer keeps merged patches on GPU up to a fraction of available free VRAM:

```
usable_vram = free_vram - current_model_size
gpu_budget = usable_vram × vram_budget
```

Patches are placed on GPU until the budget is exhausted; remaining patches stay on CPU. This trades GPU memory for faster sampling — especially useful for iterative workflows where you sample repeatedly with the same merge.

### Patch Application

The merged patches (compressed or full-rank) are applied to the model via ComfyUI's model patcher system. CLIP patches are applied separately with the `clip_strength_multiplier` scaling.

The optimizer outputs:
- **MODEL** — the patched model ready for sampling
- **CLIP** — the patched text encoder (if CLIP input was provided)
- **STRING** — detailed analysis report with per-prefix strategy map and suggested max strength
- **LORA_DATA** — the raw merged patches for use with Save Merged LoRA, Merged LoRA to Hook, or Merged LoRA to WanVideo

---

## Algorithm Interaction Map

This diagram shows every algorithm in the system and how they relate:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  INPUT                    ANALYSIS                    STRATEGY              │
│  ─────                    ────────                    ────────              │
│  LoRA Stack ────► Key Normalization                                         │
│                         │                                                   │
│                         ▼                                                   │
│               Architecture Detection ──► Architecture Preset                │
│                         │                (density/strength thresholds)       │
│                         ▼                                                   │
│                   Compute Diffs ──────► Sign Conflict ──► Per-Prefix        │
│                         │                  Ratio          Decision          │
│                         │                                    │              │
│                         ├──────► Cosine ─────► Auto-        │              │
│                         │       Similarity     Strength      │              │
│                         │           │             │          │              │
│                         │           └──► Behavior  │          │              │
│                         │               Profile    │          │              │
│                         │               (strategy  │          │              │
│                         │                logic)    │          │              │
│                         │                          │          │              │
│                         ├──────► Magnitude ──► Sign Method   │              │
│                         │         Ratio       Selection      │              │
│                         │                                    │              │
│                         ├──────► Norms ──► Suggested Max     │              │
│                         │                  Strength          │              │
│                         │                                    │              │
│                   Discard Diffs                              │              │
│                                                              │              │
│  ═══════════════════════════════════════════════════════════════════════     │
│                                                              │              │
│  MERGE (per prefix)                                          │              │
│  ──────────────────                                          ▼              │
│                                                                             │
│  Recompute Diffs ──► KnOTS ──► DO-Merge ──► DARE/DELLA ──► ┌──────────┐   │
│  (with adjusted       (if       (if          (if            │  Select  │   │
│   strengths)        full)     refine+)     enabled)         │  Merge:  │   │
│                                                  │          │          │   │
│                                    ┌─────────────┤          │ wt_sum   │   │
│                                    │             │          │ wt_avg   │   │
│                                    │  TALL-mask  │          │ TIES     │   │
│                                    │  separate   │          │ SLERP    │   │
│                                    │  (if enh+)  │          │ consensus│   │
│                                    │             │          └────┬─────┘   │
│                                    │             ▼               │         │
│                                    │        Merge Execute ◄──────┘         │
│                                    │             │                         │
│                                    │  TALL-mask  │                         │
│                                    │  re-add ────┘                         │
│                                    │             │                         │
│                                    └─────────────┘                         │
│                                                  │                         │
│                                                  ▼                         │
│                                           SVD Compress                     │
│                                                  │                         │
│                                                  ▼                         │
│                                         Dtype Downcast                     │
│                                                  │                         │
│                                                  ▼                         │
│                                       VRAM Budget Placement                │
│                                                  │                         │
│                                                  ▼                         │
│                                           Apply to Model                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Node Reference

All nodes registered by the LoRA Optimizer:

| Node | Display Name | Purpose |
|------|-------------|---------|
| `LoRAStack` | LoRA Stack | Build LoRA stacks by chaining |
| `LoRAStackDynamic` | LoRA Stack (Dynamic) | Single node with 1–10 adjustable slots, simple/advanced modes |
| `LoRAMergeSettings` | LoRA Merge Settings | Shared config: normalization, architecture preset, smoothing, VRAM budget |
| `LoRAOptimizerSettings` | LoRA Optimizer Settings | Optimizer-specific settings: auto-strength, sparsification, compression, etc. |
| `LoRAAutoTunerSettings` | LoRA AutoTuner Settings | Tuner-specific settings: top_n, scoring, diff cache, etc. |
| `LoRAOptimizerSimple` | LoRA Optimizer | Simplified optimizer with sensible defaults, accepts optional `settings` and `tuner_data` |
| `LoRAOptimizer` | LoRA Optimizer (Legacy) | All parameters on one node — for bridge workflow with `settings_source` |
| `LoRAAutoTuner` | LoRA AutoTuner | Automated parameter sweep to rank merge configs |
| `LoRAMergeSelector` | Merge Selector | Select alternative configs from AutoTuner results |
| `LoRAConflictEditor` | LoRA Conflict Editor | Interactive conflict analysis with per-LoRA overrides |
| `SaveMergedLoRA` | Save Merged LoRA | Export merged patches as standalone `.safetensors` |
| `MergedLoRAToHook` | Merged LoRA to Hook | Wrap merged patches as conditioning hooks |
| `LoRACompatibilityAnalyzer` | LoRA Compatibility Analyzer | Pre-merge planning: overlap analysis, grouping, optional node creation |
| `WanVideoLoRAOptimizer` | WanVideo LoRA Optimizer (WIP) | Optimizer variant for WanVideo models |
| `MergedLoRAToWanVideo` | Merged LoRA → WanVideo (WIP) | Bridge merged LORA_DATA to WanVideo wrapper models |

### Node Variants

**LoRA Optimizer** (recommended):
Exposes `model`, `lora_stack`, `output_strength`, `clip` (optional), and `clip_strength_multiplier`. Accepts optional `settings` (from a Settings node) and `tuner_data` (from AutoTuner) inputs. When no settings node is connected, uses built-in defaults: `auto_strength=enabled`, `optimization_mode=per_prefix`, `merge_refinement=none`, `patch_compression=smart`, `vram_budget=0.0`.

**LoRA Optimizer (Legacy):**
All parameters on one node. Has `settings_source` for the AutoTuner ↔ Optimizer bridge workflow. Superseded by LoRA Optimizer + Settings nodes for new workflows.

**Settings nodes** (LoRA Merge Settings, LoRA Optimizer Settings, LoRA AutoTuner Settings):
Separate shared and node-specific configuration from the main optimizer. Merge Settings feeds into Optimizer Settings or AutoTuner Settings, which feed into the optimizer's `settings` input.

**WanVideo LoRA Optimizer:**
Accepts `WANVIDEOMODEL` instead of `MODEL`, skips CLIP. Defaults differ from the standard optimizer: `normalize_keys=enabled` (WanVideo LoRAs come from many trainers), `cache_patches=disabled` (video models are large), `architecture_preset=dit`.

---

## Appendix: AutoTuner

The **LoRA AutoTuner** node automates parameter selection by sweeping all combinations and ranking them with internal merge metrics, plus an optional external evaluator hook.

### Architecture

The AutoTuner runs a **single Pass 1 analysis** and caches the results. All parameter combinations are scored against this cached analysis without re-analyzing — making the sweep fast regardless of how many combinations are tested.

### Phase 1 — Fast Heuristic Scoring

Generates a full parameter grid across all configurable dimensions:

| Parameter | Values Swept |
|-----------|-------------|
| Merge modes | `weighted_average`, `slerp`, `consensus`, `ties` |
| Sparsification | `disabled`, `dare`, `della`, `dare_conflict`, `della_conflict` |
| Density | `0.5`, `0.7`, `0.9` |
| DARE dampening | `0.0`, `0.3`, `0.6` |
| Quality | `none`, `refine`, `full` |
| Auto-strength | `enabled`, `disabled` |
| Optimization mode | `per_prefix`, `global` |

This produces **2,000+ combinations**. Each is scored heuristically in ~2-5 seconds total using the analysis data from Pass 1 — no actual merges are performed. Scoring considers conflict, excess conflict, cosine similarity, subspace overlap, and magnitude distribution.

### Phase 2 — Merge & Measure Top-N

The top-N configs from Phase 1 are actually merged and scored by measuring:
- **Norm consistency** across patches
- **Effective rank** (SVD-based entropy) — when `scoring_svd=enabled`
- **Sparsity distribution** (estimated via column-sampling for efficiency)
- **Composite score** combining all metrics, optionally blended with an external evaluator

Progress is tracked via ComfyUI's progress bar (spanning both Phase 1 analysis and Phase 2 merges). Each candidate logs merge time and scoring time for diagnostics.

### Memory Management

Phase 2 is designed to avoid RAM exhaustion:
- Only the **current best candidate** is kept in memory
- Non-best candidates' model/clip patches are freed immediately after scoring
- `gc.collect()` is forced after freeing each non-best candidate
- Magnitude samples from Phase 1 are freed before Phase 2 begins

### AutoTuner Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `top_n` | 3 | How many candidates to actually merge and score |
| `scoring_svd` | disabled | SVD-based scoring: `disabled` (norm-only), `merge_quality` (SVD on merged diffs), `lora_rank` (effective rank, experimental), `full` (both) |
| `scoring_device` | gpu | Where to run scoring math (`gpu` is much faster with SVD modes) |
| `scoring_speed` | turbo | Subsample prefix scoring for faster sweeps (`full`, `fast`, `turbo`, `turbo+`) |
| `scoring_formula` | v2 | `v2` (arch-aware sparsity + energy) or `v1` (legacy, fixed 40% sparsity target) |
| `memory_mode` | auto | Persistent cross-session result cache (`disabled`, `auto`, `auto_ignore_strength`, `read_only`, `clear_and_run`) |
| `community_cache` | disabled | HF community cache (`disabled`, `upload_only`, `upload_and_download`) |
| `auto_strength_floor` | -1.0 | Minimum auto-strength floor for orthogonal LoRAs (`-1` = architecture default) |
| `decision_smoothing` | 0.25 | Smooth per-prefix decision metrics toward block averages |
| `evaluator` | — | External evaluator hook for prompt/reference scoring |
| `diff_cache_mode` | auto | Reuse raw LoRA diffs across candidates (`disabled`, `auto`, `ram`, `disk`) |
| `diff_cache_ram_pct` | 0.5 | RAM budget for `auto` diff caching before spilling to disk |
| `cache_patches` | enabled | Keep the final AutoTuner merge cached in RAM for fast re-execution |
| `vram_budget` | 0.0 | Fraction of free VRAM for patch placement |
| `record_dataset` | disabled | Save analysis metrics to JSONL for threshold tuning research |

### Dataset Recording

When `record_dataset=enabled`, the AutoTuner appends a JSONL entry to `<user dir>/lora_optimizer_reports/autotuner_dataset.jsonl` after each full sweep (cache and memory replays don't add entries). Each entry records:
- Timestamp and detected architecture
- LoRA names and analysis summary
- Per-prefix strategy distribution
- All top-N configs with their measured scores

This data is used for refining architecture presets and threshold tuning.

### AutoTuner Outputs

| Output | Type | Purpose |
|--------|------|---------|
| `MODEL` | MODEL | Patched model from the top-ranked config |
| `CLIP` | CLIP | Patched text encoder |
| `report` | STRING | Ranked results with scores and parameters |
| `analysis_report` | STRING | Full optimizer analysis report for the top-ranked config |
| `TUNER_DATA` | TUNER_DATA | All ranked configs for Merge Selector |
| `LORA_DATA` | LORA_DATA | Merged patches for Save Merged LoRA / Merged LoRA to Hook |

Both AutoTuner and Merge Selector are marked as `OUTPUT_NODE = True` for ComfyUI's output recording and re-execution optimization.

### AutoTuner Workflow

```
LoRA Stack ──► LoRA AutoTuner ──► MODEL + CLIP + Report + Analysis Report + TUNER_DATA + LORA_DATA
                                                                                │            │
                                                                                ▼            ▼
                                                                         Merge Selector   Save Merged LoRA
                                                                      (pick alternative)
                                                                                │
                                                                                ▼
                                                                    MODEL + CLIP + LORA_DATA
```

---

## Appendix: Conflict Editor

The **LoRA Conflict Editor** node provides interactive analysis and override control:

1. Loads all LoRAs and computes pairwise conflict ratios
2. Auto-suggests per-LoRA conflict modes:
   - < 15% average conflict → `all`
   - 15–40% → `low_conflict`
   - \> 40% → `high_conflict`
3. Allows manual override of conflict modes and merge strategy
4. Outputs enriched stack + merge strategy override for the optimizer

### Conflict Editor Workflow

```
LoRA Stack ──► LoRA Conflict Editor ──► Enriched LORA_STACK ──► LoRA Optimizer
                     │                                                ▲
                     └── merge_strategy (STRING) ─────────────────────┘
                                                   (strategy override)
```

---

## Appendix: Per-LoRA Controls

### Conflict Modes

Per-LoRA control over where each LoRA's contributions are applied:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `all` (default) | Apply everywhere | Normal merging |
| `low_conflict` | Only where this LoRA agrees with the majority | Conservative — keep only safe contributions |
| `high_conflict` | Only where this LoRA disagrees with the majority | Force dominance in contested regions |

### Key Filters

Per-LoRA control over which weight prefixes each LoRA contributes to:

| Filter | Behavior | Use Case |
|--------|----------|----------|
| `all` (default) | All prefixes | Normal merging |
| `shared_only` | Only prefixes present in 2+ LoRAs | Strip variant-specific keys (e.g., I2V keys from a mixed LoRA) |
| `unique_only` | Only prefixes present in exactly 1 LoRA | Extract lightweight variant-specific adapters |

---

## Appendix: Output Nodes

### Save Merged LoRA

Exports the optimizer's merged result as a standalone `.safetensors` file. Connects to `LORA_DATA` from LoRA Optimizer, AutoTuner, or Merge Selector.

| Option | Default | Effect |
|--------|---------|--------|
| `save_folder` | first configured LoRA folder | Choose which configured ComfyUI LoRA directory to save into |
| `filename` | `merged_lora` | File name relative to `save_folder`. Subdirectories allowed; traversal blocked |
| `save_rank` | 0 (auto) | 0 = use each layer's existing rank. Non-zero = force this rank via SVD compression |
| `bake_strength` | enabled | Bakes `output_strength` into saved weights so the LoRA reproduces your merge at strength 1.0 |

### Merged LoRA to Hook

Wraps merged patches as a **conditioning hook** (`HOOKS`) for per-conditioning LoRA application:

- **Per-prompt LoRA** — apply different merged LoRAs to positive vs negative conditioning
- **Scheduled application** — combine with hook keyframes for step-specific LoRA
- **Regional conditioning** — apply LoRA to specific image regions
- **Preserving the base model** — keep MODEL unpatched while using the merge through hooks

### Merged LoRA → WanVideo

Bridges `LORA_DATA` to a `WANVIDEOMODEL`. Handles the `_orig_mod.` key mismatch from `torch.compile` and injects patches in the format WanVideo's model loader expects.

---

## Appendix: Supported Architectures

| Architecture | Key Detection Patterns | Special Handling |
|-------------|----------------------|-----------------|
| **FLUX** | `double_blocks`, `single_blocks` | Sliced QKV offsets, multi-trainer unification |
| **SD 1.5** | `lora_te_`, `input_blocks`/`down_blocks` | Text encoder + UNet unified |
| **SDXL** | `lora_te1_`, `input_blocks` | Text encoder + UNet unified |
| **Z-Image** (Lumina2) | `diffusion_model.layers.N.attention` | Fused QKV split/re-fuse, Musubi Tuner format |
| **Ideogram 4** (NextDiT) | `layers.N.attention.qkv`/`attention.o`, fal `conditional_transformer.` prefix | ai-toolkit / fal / PEFT prefixes unified; qkv stays fused (detected **before** Z-Image) |
| **Wan** 2.1/2.2 | `blocks.N` with `self_attn`/`ffn` | LyCORIS / diffusers / Musubi / Fun LoRA / finetrainer unified, RS-LoRA alpha fix |
| **ACE-Step** v1.0/v1.5 | `layers.N` with `self_attn`/`cross_attn` + `q_proj`/`k_proj`/`v_proj` | Attention key unification, music-DiT preset |
| **LTX Video** | `adaln_single`, `attn1`/`attn2` | Trainer format unification |
| **Qwen-Image** | `img_mlp`/`txt_mlp`/`img_mod`/`txt_mod` | Dual-stream key unification |

**Supported trainers** (auto-normalized when `normalize_keys=enabled`):
Kohya, AI-Toolkit, LyCORIS, Musubi Tuner, diffusers/PEFT, Fun LoRA, finetrainer

**Supported LoRA formats:**
Standard LoRA, LoCon (with `lora_mid`), LyCORIS, diffusers/PEFT format, RS-LoRA (with alpha fix)

---

## References

- **TIES-Merging:** [Yadav et al., NeurIPS 2023](https://arxiv.org/abs/2306.01708) — Trim, Elect Sign, Disjoint Merge
- **DARE:** [Yu et al., ICML 2024](https://arxiv.org/abs/2311.03099) — Drop And REscale
- **DAREx:** ICLR 2025 — Dampened DARE rescaling
- **DELLA:** [Deep et al., 2024](https://arxiv.org/abs/2406.11617) — Magnitude-aware dropout
- **KnOTS:** [Ramé et al., 2024](https://arxiv.org/abs/2407.09095) — SVD alignment for model merging
- **TALL-masks:** [Wang et al., 2024](https://arxiv.org/abs/2406.12832) — Task-aware selfish weight protection
- **ZipLoRA:** [Shah et al., 2025](https://arxiv.org/abs/2311.13600) — Column-wise structural sparsity (inspiration)
- **DO-Merging:** [arXiv 2505.15875](https://arxiv.org/abs/2505.15875) — Decouple & Orthogonalize
