# AutoTuner Speed Optimizations

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce autotuner tuning loop time by 15-30% through 6 targeted micro-optimizations.

**Architecture:** All changes are in `lora_optimizer.py`. Each optimization is independent and touches a different function/section. No new files, no API changes, no behavior changes — only faster execution of the same logic.

**Tech Stack:** Python, PyTorch, existing test harness in `tests/test_lora_optimizer.py`

---

## Task 1: Lighten GC + remove `torch.cuda.empty_cache()` in Phase 2 loop

**Files:**
- Modify: `lora_optimizer.py:9270-9273`

**Context:** Inside the Phase 2 per-candidate loop, every iteration calls `gc.collect()` (full heap scan, 100-500ms on large processes) and `torch.cuda.empty_cache()` (forces CUDA sync). With `top_n=3`, that's 3 forced full GC cycles + 3 CUDA syncs.

**Why we can't remove gc.collect() entirely:** `merged_model` is a ComfyUI `ModelPatcher` wrapping a PyTorch `nn.Module`, which has circular references (parent→child→parent via `_modules`). Python's refcounting alone won't free these — gc is needed to break cycles. Removing it risks accumulating ~10-40GB of unreleased model clones until the post-loop `gc.collect()` at line 9354.

**Safe approach:** Replace `gc.collect()` (full scan of all 3 generations) with `gc.collect(0)` (youngest generation only — ~10x faster, catches most freshly-created cycles). Remove `torch.cuda.empty_cache()` — the CUDA allocator handles block reuse internally, and the forced sync is pure overhead. The post-loop cleanup at line 9354-9356 still does a full `gc.collect()` + `empty_cache()`.

**Step 1: Make the code change**

Current code (lines 9270-9273):
```python
            del m_patches, c_patches  # Drop patch-dict references so tensors can free
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
```

New code:
```python
            del m_patches, c_patches  # Drop patch-dict references so tensors can free
            gc.collect(0)  # gen-0 only: ~10x faster, catches fresh cycles from ModelPatcher
```

**Step 2: Verify existing tests still pass**

Run: `cd /media/p5/ComfyUI-ZImage-LoRA-Merger && python -m pytest tests/test_lora_optimizer.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "perf: lighten GC in Phase 2 loop — gen-0 only, drop empty_cache

Replace full gc.collect() with gc.collect(0) (youngest generation only,
~10x faster). Remove torch.cuda.empty_cache() — CUDA allocator handles
block reuse, forced sync is pure overhead. Post-loop cleanup at line
9354-9356 still does full gc.collect() + empty_cache()."
```

---

## Task 2: Replace `torch.randperm` with `torch.randint` in `_sample_pair_metrics`

**Files:**
- Modify: `lora_optimizer.py:2107-2113`
- Test: `tests/test_lora_optimizer.py`

**Context:** `_sample_pair_metrics` downsamples to 100k elements when vectors are large (line 2108). Currently uses `torch.randperm(n)[:100000]` which generates a full permutation of n elements (O(n) time and memory) then takes the first 100k. For a 10M-element vector, this allocates a 10M-element int64 tensor just to pick 100k indices. `torch.randint(0, n, (100000,))` is O(k) — only allocates the 100k indices needed. The analysis path at line 4632 already uses `randint` correctly.

Note: `randint` samples with replacement (may produce duplicate indices), but for 100k out of millions the collision rate is <1% and doesn't affect conflict statistics meaningfully.

**Step 1: Write the test**

```python
def test_sample_pair_metrics_downsamples_large_vectors(self):
    """Pair metrics should work correctly with large vectors that trigger downsampling."""
    optimizer = lora_optimizer.LoRAOptimizer()
    # Create vectors larger than 100k to trigger downsampling path
    a = torch.randn(200000)
    b = torch.randn(200000)
    result = optimizer._sample_pair_metrics(a, b)
    # Should still produce valid metrics
    self.assertIn("overlap", result)
    self.assertIn("conflict", result)
    self.assertIn("dot", result)
    self.assertGreater(result["overlap"], 0)
    # Verify determinism: same inputs should give same results
    result2 = optimizer._sample_pair_metrics(a, b)
    self.assertEqual(result["overlap"], result2["overlap"])
    self.assertEqual(result["conflict"], result2["conflict"])
    self.assertAlmostEqual(result["dot"], result2["dot"], places=4)
```

**Step 2: Run test to verify it passes (baseline)**

Run: `python -m pytest tests/test_lora_optimizer.py::LoRAOptimizerTests::test_sample_pair_metrics_downsamples_large_vectors -v`
Expected: PASS

**Step 3: Make the code change**

In `lora_optimizer.py` at lines 2107-2113, replace `randperm` with `randint`:

Current:
```python
        n = flat_a.numel()
        if n > 100000:
            target_device = flat_a.device
            g = torch.Generator(device=target_device).manual_seed(42)
            indices = torch.randperm(n, device=target_device, generator=g)[:100000]
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]
```

New:
```python
        n = flat_a.numel()
        if n > 100000:
            target_device = flat_a.device
            g = torch.Generator(device=target_device).manual_seed(42)
            indices = torch.randint(0, n, (100000,), device=target_device, generator=g)
            flat_a = flat_a[indices]
            flat_b = flat_b[indices]
```

**Step 4: Run tests to verify everything passes**

Run: `python -m pytest tests/test_lora_optimizer.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "perf: use randint instead of randperm in _sample_pair_metrics

randperm(n)[:100k] allocates O(n) memory for full permutation then
discards most of it. randint is O(k) — only allocates the 100k
indices needed. For 10M-element vectors, avoids ~80MB temp allocation."
```

---

## Task 3: Transfer tensors before float conversion in `_score_merge_result`

**Files:**
- Modify: `lora_optimizer.py:3696-3700`
- Test: `tests/test_lora_optimizer.py`

**Context:** In the LoRAAdapter scoring branch of `_score_merge_result`, the code converts to float32 THEN transfers to score_device. Moving fp32 data is 2x the bandwidth of moving fp16/bf16 data. The fix is to transfer first (in native dtype), then convert to float32 on the target device. The float conversion still happens before the gram matrix computation, so numerical accuracy is preserved.

**Step 1: Write the test**

```python
def test_score_merge_result_lora_adapter_on_device(self):
    """_score_merge_result should handle LoRAAdapter patches correctly."""
    LoRAAdapter = lora_optimizer.LoRAAdapter
    up = torch.randn(4, 8)
    down = torch.randn(4, 16)
    alpha = 4.0
    adapter = LoRAAdapter(
        loaded_keys=set(),
        weights=(up, down, alpha, None, None, None),
    )
    patches = {("key1",): adapter}
    result = lora_optimizer._score_merge_result(patches, {}, compute_svd=False)
    self.assertIn("norm_mean", result)
    self.assertGreater(result["norm_mean"], 0)
    self.assertIn("composite_score", result)
```

**Step 2: Run test to verify it passes (baseline)**

Run: `python -m pytest tests/test_lora_optimizer.py -k test_score_merge_result_lora_adapter -v`
Expected: PASS

**Step 3: Make the code change**

In `lora_optimizer.py` at lines 3696-3700, reorder to transfer then convert:

Current:
```python
                up_flat = mat_up.flatten(start_dim=1).float()
                down_flat = mat_down.flatten(start_dim=1).float()
                if score_device is not None:
                    up_flat = up_flat.to(score_device)
                    down_flat = down_flat.to(score_device)
```

New:
```python
                up_flat = mat_up.flatten(start_dim=1)
                down_flat = mat_down.flatten(start_dim=1)
                if score_device is not None:
                    up_flat = up_flat.to(score_device)
                    down_flat = down_flat.to(score_device)
                up_flat = up_flat.float()
                down_flat = down_flat.float()
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_lora_optimizer.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "perf: transfer LoRA factors before float conversion in scoring

Moving fp16/bf16 tensors to score_device then converting to fp32
halves the PCIe transfer bandwidth vs converting first. For ~240
LoRAAdapter patches per candidate, saves meaningful transfer time."
```

---

## Task 4: Gate psutil memory logging behind first/last iteration

**Files:**
- Modify: `lora_optimizer.py:9274-9283`

**Context:** Every Phase 2 iteration imports psutil and reads `/proc/self/status` for memory diagnostics. This is debug-only information. Gate it to only run on the first and last iteration of the loop.

**Step 1: Make the code change**

Current (lines 9274-9283):
```python
            # Log memory usage to help diagnose leaks on large models
            try:
                import psutil
                proc = psutil.Process()
                rss_gb = proc.memory_info().rss / (1024**3)
                dc_mb = _diff_cache.size_mb() if _diff_cache else 0
                logging.info(f"[LoRA AutoTuner]   Memory: process={rss_gb:.1f}GB"
                             f"{f', diff_cache={dc_mb:.0f}MB' if dc_mb > 0 else ''}")
            except ImportError:
                pass
```

New:
```python
            # Log memory usage on first/last iteration to diagnose leaks
            if rank_idx == 0 or rank_idx == len(top_candidates) - 1:
                try:
                    import psutil
                    proc = psutil.Process()
                    rss_gb = proc.memory_info().rss / (1024**3)
                    dc_mb = _diff_cache.size_mb() if _diff_cache else 0
                    logging.info(f"[LoRA AutoTuner]   Memory: process={rss_gb:.1f}GB"
                                 f"{f', diff_cache={dc_mb:.0f}MB' if dc_mb > 0 else ''}")
                except ImportError:
                    pass
```

**Step 2: Verify tests pass**

Run: `python -m pytest tests/test_lora_optimizer.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "perf: gate psutil memory logging to first/last Phase 2 iteration

Memory diagnostics on every iteration adds overhead without value.
First and last iteration still show whether memory leaks occur."
```

---

## Task 5: Cache LoRA format detection per lora_dict

**Files:**
- Modify: `lora_optimizer.py:684` (`_LoRAMergeBase.__init__`)
- Modify: `lora_optimizer.py:1786-1821` (`_get_lora_key_info`)
- Test: `tests/test_lora_optimizer.py`

**Context:** `_get_lora_key_info` (defined on `_LoRAMergeBase`, line 1786) tries 4 LoRA key formats (8 dict lookups) per call. It's called from 5 sites: `_prepare_group_diffs` (line 1672), `_process_prefix` (lines 4343, 4472), `_build_exact_linear_patch` (line 5400), and `analyze_and_enrich` (line 11302). A LoRA file uses one format consistently for all keys. Caching the detected format index avoids up to 6 wasted dict lookups per subsequent call.

**Important:** Initialize `_lora_format_cache` in `_LoRAMergeBase.__init__` (line 684), NOT `LoRAOptimizer.__init__`, since the method lives on `_LoRAMergeBase`. Stale cache entries are safe — if the cached format doesn't match a prefix, it falls through to the full cascade.

**Step 1: Write the failing test**

```python
def test_lora_format_cache_avoids_repeated_detection(self):
    """After detecting a LoRA's format once, subsequent prefixes should reuse it."""
    optimizer = lora_optimizer.LoRAOptimizer()
    # Create a LoRA with diffusers2 format (lora_B/lora_A) — the 3rd format tried
    lora_dict = {
        "unet.a.lora_B.weight": torch.tensor([[1.0]], dtype=torch.float32),
        "unet.a.lora_A.weight": torch.tensor([[1.0]], dtype=torch.float32),
        "unet.b.lora_B.weight": torch.tensor([[2.0]], dtype=torch.float32),
        "unet.b.lora_A.weight": torch.tensor([[1.0]], dtype=torch.float32),
    }
    # First call should detect format and cache it
    result1 = optimizer._get_lora_key_info(lora_dict, "unet.a")
    self.assertIsNotNone(result1)
    # Verify cache was populated
    self.assertIn(id(lora_dict), optimizer._lora_format_cache)
    # Second call with different prefix should use cache
    result2 = optimizer._get_lora_key_info(lora_dict, "unet.b")
    self.assertIsNotNone(result2)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_optimizer.py -k test_lora_format_cache -v`
Expected: FAIL — `_lora_format_cache` doesn't exist yet

**Step 3: Implement format caching**

Add `_lora_format_cache` to `_LoRAMergeBase.__init__` (line 684):

```python
def __init__(self):
    self.loaded_loras = {}
    self._lora_format_cache = {}  # id(lora_dict) -> format_index (0-3)
```

Modify `_get_lora_key_info` (lines 1786-1821). To avoid duplicating the extraction body, extract it into a local helper:

```python
    def _get_lora_key_info(self, lora_dict, key_prefix):
        formats = [
            ("{}.lora_up.weight", "{}.lora_down.weight"),           # regular
            ("{}_lora.up.weight", "{}_lora.down.weight"),           # diffusers
            ("{}.lora_B.weight", "{}.lora_A.weight"),               # diffusers2
            ("{}.lora.up.weight", "{}.lora.down.weight"),           # diffusers3
        ]

        def _extract(up_key, down_key):
            mat_up = lora_dict[up_key]
            mat_down = lora_dict[down_key]
            alpha_key = "{}.alpha".format(key_prefix)
            alpha = lora_dict.get(alpha_key, None)
            if alpha is not None:
                alpha = alpha.item()
            else:
                alpha = mat_down.shape[0]
            mid_key = "{}.lora_mid.weight".format(key_prefix)
            mid = lora_dict.get(mid_key, None)
            return (mat_up, mat_down, alpha, mid)

        # Try cached format first
        dict_id = id(lora_dict)
        cached_fmt = self._lora_format_cache.get(dict_id)
        if cached_fmt is not None:
            up_key = formats[cached_fmt][0].format(key_prefix)
            down_key = formats[cached_fmt][1].format(key_prefix)
            if up_key in lora_dict and down_key in lora_dict:
                return _extract(up_key, down_key)

        for fmt_idx, (up_fmt, down_fmt) in enumerate(formats):
            up_key = up_fmt.format(key_prefix)
            down_key = down_fmt.format(key_prefix)
            if up_key in lora_dict and down_key in lora_dict:
                self._lora_format_cache[dict_id] = fmt_idx
                return _extract(up_key, down_key)

        return None
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_lora_optimizer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "perf: cache LoRA format detection per lora_dict

A LoRA file uses one key format consistently. Caching the winning
format index (0-3) after first detection avoids up to 6 wasted
dict lookups per prefix on subsequent calls. Cache miss on stale
entries is safe — falls through to full cascade."
```

---

## Task 6: Skip single-LoRA patches in per-candidate scoring

**Files:**
- Modify: `lora_optimizer.py:9119-9138` (Phase 2 scoring call site)
- Modify: `lora_optimizer.py:3617-3850` (`_score_merge_result` — add `_baseline` and `_return_raw` params)
- Test: `tests/test_lora_optimizer.py`

**Context:** `_score_merge_result` iterates ALL patches (model + clip), including single-LoRA prefixes. For single-LoRA prefixes, the merge path returns early at line 2868 (`if len(diffs_with_weights) == 1: return diff * weight`) — no TIES/DARE/merge strategy is applied. The patches are produced identically across candidates.

**Auto-strength caveat:** Different candidates may have different `auto_strength` settings ("enabled" vs "disabled"), which applies a uniform scalar (`model_auto_scale`) to ALL patches. However, the scoring metrics used in composite_score (norm_cv, effective_rank, sparsity) are all scale-invariant — the scalar cancels out. The one non-scale-invariant metric, `norm_energy_sq`, is used in `energy_preservation` (lines 9160-9206), but that's computed separately in the Phase 2 loop from `branch_energy` data — not from `_score_merge_result`'s energy_sq. So the cached single-LoRA contribution to the composite score is valid regardless of auto_strength.

**Approach:** Add `_baseline` (pre-computed raw lists) and `_return_raw` (return raw lists) parameters to `_score_merge_result`. On the first candidate, score single-LoRA patches separately with `_return_raw=True` to capture their raw norms/sparsities/ranks. On all candidates, score only multi-LoRA patches with `_baseline` set to the cached single-LoRA lists. The accumulators get the cached lists prepended, so composite metrics (mean, CV, etc.) are computed correctly from the full combined data.

**Step 1: Write the test**

```python
def test_score_merge_result_baseline_matches_full(self):
    """Scoring multi-LoRA patches with single-LoRA baseline should match full scoring."""
    LoRAAdapter = lora_optimizer.LoRAAdapter
    # Create 10 patches: 5 "single-LoRA" and 5 "multi-LoRA"
    single_patches = {}
    multi_patches = {}
    all_patches = {}
    for i in range(10):
        up = torch.randn(4, 8)
        down = torch.randn(4, 16)
        adapter = LoRAAdapter(set(), (up, down, 4.0, None, None, None))
        key = f"key{i}"
        all_patches[key] = adapter
        if i < 5:
            single_patches[key] = adapter
        else:
            multi_patches[key] = adapter

    # Score everything together
    full = lora_optimizer._score_merge_result(all_patches, {}, compute_svd=False)

    # Score single-LoRA first with _return_raw, then multi with _baseline
    sl = lora_optimizer._score_merge_result(
        single_patches, {}, compute_svd=False, _return_raw=True)
    baseline = sl["_raw"]
    combined = lora_optimizer._score_merge_result(
        multi_patches, {}, compute_svd=False, _baseline=baseline)

    # Composite score should match
    self.assertAlmostEqual(full["composite_score"], combined["composite_score"], places=6)
    self.assertAlmostEqual(full["norm_mean"], combined["norm_mean"], places=6)
    self.assertAlmostEqual(full["norm_cv"], combined["norm_cv"], places=6)
    self.assertAlmostEqual(full["sparsity_mean"], combined["sparsity_mean"], places=6)
    self.assertAlmostEqual(full["norm_energy_sq"], combined["norm_energy_sq"], places=4)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_optimizer.py -k test_score_merge_result_baseline -v`
Expected: FAIL — `_return_raw` and `_baseline` parameters don't exist yet

**Step 3: Implement `_baseline` and `_return_raw` in `_score_merge_result`**

Modify the function signature (line 3617):
```python
def _score_merge_result(model_patches, clip_patches, compute_svd=True,
                        score_device=None, arch_preset=None, lora_svd=False,
                        _baseline=None, _return_raw=False):
```

Replace the accumulator initialization (lines 3630-3634):

Current:
```python
    norms = []
    importance_values = []
    effective_ranks = []
    sparsities = []
```

New:
```python
    if _baseline is not None:
        norms = list(_baseline["norms"])
        importance_values = list(_baseline["importance_values"])
        effective_ranks = list(_baseline["effective_ranks"])
        sparsities = list(_baseline["sparsities"])
    else:
        norms = []
        importance_values = []
        effective_ranks = []
        sparsities = []
```

Before the `return metrics` at the end (line ~3849), add:
```python
    if _return_raw:
        metrics["_raw"] = {
            "norms": norms,
            "importance_values": importance_values,
            "effective_ranks": effective_ranks,
            "sparsities": sparsities,
        }
```

**Step 4: Run test to verify _baseline/_return_raw works**

Run: `python -m pytest tests/test_lora_optimizer.py -k test_score_merge_result_baseline -v`
Expected: PASS

**Step 5: Wire up single-LoRA caching in Phase 2 loop**

Before the Phase 2 loop (around line 9070), build the set of single-LoRA target keys:

```python
        # Pre-identify single-LoRA target keys for scoring cache
        single_lora_keys = set()
        for pfx, info in all_key_targets.items():
            if prefix_stats.get(pfx, {}).get("n_loras", 0) <= 1:
                single_lora_keys.add(info[0])  # info is (target_key, is_clip)
        _cached_sl_baseline = None
```

Inside the loop, replace the scoring block (lines 9121-9138) with:

```python
            m_patches = lora_data["model_patches"] if lora_data else {}
            c_patches = lora_data["clip_patches"] if lora_data else {}
            is_ortho_score = (
                abs(avg_cos_sim) < tuner_arch_preset["orthogonal_cos_sim_max"]
                and avg_subspace_overlap < 0.35
            )
            compute_svd = scoring_svd in ("merge_quality", "full") and not is_ortho_score
            compute_lora_svd = scoring_svd in ("lora_rank", "full")
            score_dev = torch.device("cuda") if scoring_device == "gpu" and torch.cuda.is_available() else None
            t_score = time.time()
            score_arch = tuner_arch_preset if scoring_formula == "v2" else None

            if single_lora_keys:
                # Filter patches into single-LoRA and multi-LoRA sets
                def _target_key(k):
                    return k[0] if isinstance(k, tuple) else k

                m_multi = {k: v for k, v in m_patches.items()
                           if _target_key(k) not in single_lora_keys}
                c_multi = {k: v for k, v in c_patches.items()
                           if _target_key(k) not in single_lora_keys}

                if _cached_sl_baseline is None:
                    # First candidate: score single-LoRA patches, cache raw lists
                    m_single = {k: v for k, v in m_patches.items()
                                if _target_key(k) in single_lora_keys}
                    c_single = {k: v for k, v in c_patches.items()
                                if _target_key(k) in single_lora_keys}
                    sl_measured = _score_merge_result(
                        m_single, c_single, compute_svd=compute_svd,
                        score_device=score_dev, arch_preset=score_arch,
                        lora_svd=compute_lora_svd, _return_raw=True)
                    _cached_sl_baseline = sl_measured.get("_raw")
                    del m_single, c_single, sl_measured

                # Score multi-LoRA patches with single-LoRA baseline merged in
                measured = _score_merge_result(
                    m_multi, c_multi, compute_svd=compute_svd,
                    score_device=score_dev, arch_preset=score_arch,
                    lora_svd=compute_lora_svd,
                    _baseline=_cached_sl_baseline)
            else:
                measured = _score_merge_result(
                    m_patches, c_patches, compute_svd=compute_svd,
                    score_device=score_dev, arch_preset=score_arch,
                    lora_svd=compute_lora_svd)
```

**Step 6: Run all tests**

Run: `python -m pytest tests/test_lora_optimizer.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "perf: cache single-LoRA patch scoring across Phase 2 candidates

Single-LoRA patches bypass all merge strategies (early return at line
2868). Score them once on the first candidate and merge as baseline
for subsequent candidates. All composite_score components (norm_cv,
effective_rank, sparsity) are scale-invariant, so auto_strength
differences between candidates don't affect correctness."
```

---

## Summary

| Task | Optimization | Est. Impact |
|------|-------------|-------------|
| 1 | gc.collect(0) + drop empty_cache in loop | 5-15% Phase 2 |
| 2 | randperm -> randint | 1-3% Phase 1 |
| 3 | Transfer before float conversion | 2-5% scoring |
| 4 | Gate psutil logging | Negligible |
| 5 | Cache LoRA format detection | 1-3% Phase 1 |
| 6 | Skip single-LoRA patch re-scoring | 10-30% scoring |
