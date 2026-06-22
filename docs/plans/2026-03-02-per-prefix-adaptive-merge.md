# Per-Prefix Adaptive Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the optimizer pick merge strategy (TIES vs weighted_average vs weighted_sum) per-prefix instead of globally, so non-overlapping LoRA regions keep full strength and only genuinely conflicting regions use TIES.

**Architecture:** Pass 1 already computes per-prefix conflict/magnitude data. Instead of averaging into one global decision, store per-prefix stats. Pass 2 looks up each prefix's stats to pick its own strategy. A toggle (`optimization_mode`) lets users switch between `per_prefix` (new default) and `global` (original behavior).

**Tech Stack:** Python, PyTorch (existing dependencies only)

---

### Task 1: Add `optimization_mode` input and update cache key

**Files:**
- Modify: `lora_optimizer.py:426-447` (INPUT_TYPES)
- Modify: `lora_optimizer.py:455-475` (_compute_cache_key)
- Modify: `lora_optimizer.py:477-482` (IS_CHANGED)

**Step 1: Add `optimization_mode` to INPUT_TYPES**

In `INPUT_TYPES` (line 435), add to the `optional` dict after `free_vram_between_passes`:

```python
"optimization_mode": (["per_prefix", "global"], {
    "default": "per_prefix",
    "tooltip": "per_prefix: each weight group picks its own merge strategy based on local conflict. global: single strategy for all (original behavior)."
}),
```

**Step 2: Add `optimization_mode` to cache key**

In `_compute_cache_key` (line 474), change the hash suffix to include the new param:

```python
h.update(f"|os={output_strength}|csm={clip_strength_multiplier}|as={auto_strength}|om={optimization_mode}".encode())
```

Add `optimization_mode` as a parameter to the method signature:

```python
@staticmethod
def _compute_cache_key(lora_stack, output_strength, clip_strength_multiplier, auto_strength, optimization_mode="per_prefix"):
```

**Step 3: Update IS_CHANGED to pass optimization_mode**

Update IS_CHANGED signature and call (lines 477-482):

```python
@classmethod
def IS_CHANGED(cls, model, clip, lora_stack, output_strength,
               clip_strength_multiplier=1.0, auto_strength="disabled",
               free_vram_between_passes="disabled", optimization_mode="per_prefix"):
    return cls._compute_cache_key(lora_stack, output_strength,
                                  clip_strength_multiplier, auto_strength,
                                  optimization_mode)
```

**Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add optimization_mode input (per_prefix/global toggle)"
```

---

### Task 2: Store per-prefix stats during Pass 1

**Files:**
- Modify: `lora_optimizer.py:1117-1166` (Pass 1 accumulators and _collect_analysis_result)

**Step 1: Add per-prefix stats accumulator**

After the existing accumulators (line 1132), add:

```python
prefix_stats = {}  # prefix -> {conflict_ratio, n_loras, magnitude_samples, magnitude_ratio}
```

**Step 2: Update `_collect_analysis_result` to store per-prefix data**

After the existing accumulation logic (line 1150), add per-prefix stat storage. The function already has `partial_stats` (which loras contribute), `pair_conflicts` (conflict data), and `mag_samples`. Store these per-prefix:

```python
# Store per-prefix stats for per_prefix optimization mode
if len(partial_stats) > 0:
    # Number of LoRAs contributing to this prefix
    n_contributing = len(partial_stats)

    # Per-prefix conflict ratio
    pf_overlap = sum(ov for ov, _ in pair_conflicts.values())
    pf_conflict = sum(conf for _, conf in pair_conflicts.values())
    pf_conflict_ratio = pf_conflict / pf_overlap if pf_overlap > 0 else 0.0

    # Per-prefix magnitude ratio (max/min L2 among contributing LoRAs)
    pf_l2s = [l2 for _, _, l2 in partial_stats if l2 > 0]
    if len(pf_l2s) >= 2:
        pf_mag_ratio = max(pf_l2s) / min(pf_l2s)
    else:
        pf_mag_ratio = 1.0

    prefix_stats[prefix] = {
        "n_loras": n_contributing,
        "conflict_ratio": pf_conflict_ratio,
        "magnitude_ratio": pf_mag_ratio,
        "magnitude_samples": list(mag_samples),  # copy, not reference
    }
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: store per-prefix conflict and magnitude stats during Pass 1"
```

---

### Task 3: Per-prefix decision logic in Pass 2

**Files:**
- Modify: `lora_optimizer.py:1282-1398` (Pass 2 merge logic)

**Step 1: Update optimize_merge signature**

Add `optimization_mode="per_prefix"` to the `optimize_merge` method signature (line 1018):

```python
def optimize_merge(self, model, clip, lora_stack, output_strength, clip_strength_multiplier=1.0, auto_strength="disabled", free_vram_between_passes="disabled", optimization_mode="per_prefix"):
```

**Step 2: Update cache_key call in optimize_merge**

Update the cache_key computation (line 1066) to include optimization_mode:

```python
cache_key = self._compute_cache_key(lora_stack, output_strength,
                                    clip_strength_multiplier, auto_strength,
                                    optimization_mode)
```

**Step 3: Modify `_merge_one_prefix` to use per-prefix decisions**

The inner function `_merge_one_prefix` (line 1292) currently uses closure variables `mode`, `density`, `sign_method`. Modify it to accept and use per-prefix params. Replace the merge call section (lines 1367-1371):

Before (current code):
```python
merged_diff = self._merge_diffs(
    diffs_list, mode,
    density=density, majority_sign_method=sign_method,
    compute_device=compute_device
)
```

After:
```python
# Determine strategy for this prefix
if optimization_mode == "per_prefix" and lora_prefix in prefix_stats:
    pf = prefix_stats[lora_prefix]
    if pf["n_loras"] <= 1 or len(diffs_list) <= 1:
        # Single LoRA on this prefix: weighted_sum, full strength
        pf_mode = "weighted_sum"
        pf_density = 0.5
        pf_sign = "frequency"
    else:
        pf_mode, pf_density, pf_sign, _ = self._auto_select_params(
            pf["conflict_ratio"], pf["magnitude_ratio"],
            magnitude_samples=pf.get("magnitude_samples")
        )
else:
    pf_mode = mode
    pf_density = density
    pf_sign = sign_method

merged_diff = self._merge_diffs(
    diffs_list, pf_mode,
    density=pf_density, majority_sign_method=pf_sign,
    compute_device=compute_device
)
```

**Step 4: Track per-prefix strategy counts for reporting**

Add counters before the Pass 2 loop (after line 1290):

```python
strategy_counts = {"weighted_sum": 0, "weighted_average": 0, "ties": 0}
```

Update `_merge_one_prefix` to return the chosen mode. Change the return (line 1374) to:

```python
return (target_key, is_clip_key, merged_diff, pf_mode)
```

Update `_collect_merge_result` (line 1376) to count strategies:

```python
def _collect_merge_result(result):
    nonlocal processed_keys
    if result is None:
        return
    target_key, is_clip_key, merged_diff, used_mode = result
    if is_clip_key:
        clip_patches[target_key] = ("diff", (merged_diff,))
    else:
        model_patches[target_key] = ("diff", (merged_diff,))
    processed_keys += 1
    strategy_counts[used_mode] = strategy_counts.get(used_mode, 0) + 1
```

**Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: per-prefix adaptive merge strategy in Pass 2"
```

---

### Task 4: Update report to show per-prefix strategy breakdown

**Files:**
- Modify: `lora_optimizer.py:938-1016` (_build_report)
- Modify: `lora_optimizer.py:1424-1437` (report call site)

**Step 1: Add strategy_counts parameter to _build_report**

Update `_build_report` signature (line 938) to accept `strategy_counts` and `optimization_mode`:

```python
def _build_report(self, lora_stats, pairwise_conflicts, collection_stats,
                  mode, density, sign_method, reasoning, merge_summary,
                  auto_strength_info=None, strategy_counts=None, optimization_mode="global"):
```

**Step 2: Add per-prefix strategy section to report**

After the "Auto-Selected Parameters" section (after line 995), add:

```python
# Per-Prefix Strategy breakdown (only in per_prefix mode)
if optimization_mode == "per_prefix" and strategy_counts:
    lines.append("")
    lines.append("--- Per-Prefix Strategy ---")
    total_pf = sum(strategy_counts.values())
    if strategy_counts.get("weighted_sum", 0) > 0:
        n = strategy_counts["weighted_sum"]
        lines.append(f"  weighted_sum (single LoRA):     {n:>4} prefixes ({n/total_pf:.0%})")
    if strategy_counts.get("weighted_average", 0) > 0:
        n = strategy_counts["weighted_average"]
        lines.append(f"  weighted_average (low conflict): {n:>4} prefixes ({n/total_pf:.0%})")
    if strategy_counts.get("ties", 0) > 0:
        n = strategy_counts["ties"]
        lines.append(f"  ties (high conflict):           {n:>4} prefixes ({n/total_pf:.0%})")
    lines.append(f"  Total:                          {total_pf:>4} prefixes")
```

**Step 3: Update the Auto-Selected Parameters section**

When in `per_prefix` mode, the "Auto-Selected Parameters" section should clarify that these are the *global fallback* parameters. Add a note:

After line 995 (the sign_method line), add:

```python
if optimization_mode == "per_prefix":
    lines.append("  (global fallback — each prefix uses its own parameters)")
```

**Step 4: Update report call site**

Update the `_build_report` call (line 1433) to pass the new params:

```python
report = self._build_report(
    lora_stats, pairwise_conflicts, collection_stats,
    mode, density, sign_method, reasoning, merge_summary,
    auto_strength_info=auto_strength_info,
    strategy_counts=strategy_counts if optimization_mode == "per_prefix" else None,
    optimization_mode=optimization_mode
)
```

**Step 5: Update saved report params**

Update the `selected_params` dict (line 1444) to include the mode:

```python
selected_params = {"mode": mode, "density": density, "sign_method": sign_method, "optimization_mode": optimization_mode}
if optimization_mode == "per_prefix":
    selected_params["strategy_counts"] = dict(strategy_counts)
```

**Step 6: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add per-prefix strategy breakdown to analysis report"
```

---

### Task 5: Add logging for per-prefix mode

**Files:**
- Modify: `lora_optimizer.py` (Pass 2 logging section, around line 1400)

**Step 1: Add per-prefix strategy logging**

After the Pass 2 patches log line (line 1400), add:

```python
if optimization_mode == "per_prefix":
    logging.info(f"[LoRA Optimizer]   Per-prefix strategies: "
                 f"{strategy_counts.get('weighted_sum', 0)} weighted_sum, "
                 f"{strategy_counts.get('weighted_average', 0)} weighted_average, "
                 f"{strategy_counts.get('ties', 0)} ties")
```

Also update the Pass 2 start log (line 1285) to indicate mode:

```python
logging.info(f"[LoRA Optimizer] Pass 2: Merging {len(all_key_targets)} keys "
             f"({optimization_mode} strategy, "
             f"{'sequential' if use_gpu else 'threaded'})...")
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: add per-prefix strategy logging"
```

---

### Task 6: Update DESCRIPTION and README

**Files:**
- Modify: `lora_optimizer.py:453` (DESCRIPTION)
- Modify: `README.md`

**Step 1: Update node DESCRIPTION**

Change line 453:

```python
DESCRIPTION = "Auto-analyzes LoRA stack and selects optimal merge strategy per weight group. Outputs merged model + analysis report."
```

**Step 2: Update README decision table**

In the "How it decides" section, add a note about per-prefix mode. After the existing decision table, add:

```markdown
**Per-prefix mode** (default): Each weight prefix picks its own strategy based on local conflict data. Non-overlapping prefixes use `weighted_sum` at full strength. Only genuinely conflicting prefixes use TIES. Set `optimization_mode` to `global` for the original single-strategy behavior.
```

**Step 3: Update the Example Report in README**

Add the per-prefix strategy section to the example report.

**Step 4: Commit**

```bash
git add lora_optimizer.py README.md
git commit -m "docs: update README and node description for per-prefix mode"
```

---

### Task 7: Final verification

**Step 1: Full syntax check**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 2: Verify node registration unchanged**

Run: `python -c "exec(open('lora_optimizer.py').read()); print(list(NODE_CLASS_MAPPINGS.keys()))"`
Expected: `['LoRAStack', 'LoRAOptimizer']`

**Step 3: Verify INPUT_TYPES has optimization_mode**

Run: `python -c "exec(open('lora_optimizer.py').read()); print([k for k in LoRAOptimizer.INPUT_TYPES()['optional'].keys()])"`
Expected: includes `optimization_mode`

**Step 4: Verify default is per_prefix**

Run: `python -c "exec(open('lora_optimizer.py').read()); om = LoRAOptimizer.INPUT_TYPES()['optional']['optimization_mode']; print(om[1]['default'])"`
Expected: `per_prefix`
