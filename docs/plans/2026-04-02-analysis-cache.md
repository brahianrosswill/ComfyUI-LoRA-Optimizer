# AutoTuner Analysis Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cache the scale-invariant AutoTuner analysis pass (diff matmuls + conflict sampling) in a separate `.analysis.json` file keyed by LoRA file identity (name + mtime + size, no strengths), so that re-runs with different strengths skip the expensive GPU work.

**Architecture:** New static methods split across the class hierarchy: I/O methods (`_analysis_cache_*`) and hash go on `LoRAAutoTuner`; helpers used inside `_run_group_analysis` (`_extract_for_analysis_cache`, `_reconstruct_from_analysis_cache`) go on `LoRAOptimizer` since that's where `_run_group_analysis` lives. `_run_group_analysis` returns `new_analysis_entries` in its existing return dict; `auto_tune` does the save. The dataset entry gains a `raw_analysis` field fed directly from `new_analysis_entries`.

**Tech Stack:** Python, PyTorch, JSON, `os.stat`, `hashlib.sha256`, existing `AUTOTUNER_MEMORY_DIR` storage.

---

### Task 1: Names-only hash computation

**Files:**
- Modify: `lora_optimizer.py` (add static method on `LoRAAutoTuner`, after `_memory_file_path` ~line 7420)
- Modify: `tests/test_lora_optimizer.py` (add `AnalysisCacheTests` class; extend `folder_paths` stub)

**Step 1: Extend the `folder_paths` stub**

In `_install_stubs()` at `tests/test_lora_optimizer.py:24`, add after the existing `get_full_path_or_raise` line:

```python
folder_paths.get_full_path = lambda _kind, name: name
```

**Step 2: Write failing tests**

Add at the bottom of `tests/test_lora_optimizer.py`:

```python
@unittest.skipIf(torch is None, "torch is not installed")
class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        self.tuner = lora_optimizer.LoRAAutoTuner()

    def test_names_only_hash_excludes_strength(self):
        """Same LoRA files at different strengths produce the same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "lora_a.safetensors")
            path_b = os.path.join(tmpdir, "lora_b.safetensors")
            open(path_a, "wb").close()
            open(path_b, "wb").close()

            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            side_effect=lambda _k, n: os.path.join(tmpdir, n)):
                stack_s1 = [
                    {"name": "lora_a.safetensors", "strength": 0.5},
                    {"name": "lora_b.safetensors", "strength": 1.0},
                ]
                stack_s2 = [
                    {"name": "lora_a.safetensors", "strength": 1.5},
                    {"name": "lora_b.safetensors", "strength": 0.3},
                ]
                h1, signs1 = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_s1)
                h2, signs2 = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_s2)
                self.assertEqual(h1, h2)
                self.assertEqual(signs1, {0: 1, 1: 1})
                self.assertEqual(signs2, {0: 1, 1: 1})

    def test_names_only_hash_captures_sign(self):
        """Negative strength is captured in the returned signs dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "lora_a.safetensors")
            open(path_a, "wb").close()
            with mock.patch("lora_optimizer.folder_paths.get_full_path",
                            side_effect=lambda _k, n: os.path.join(tmpdir, n)):
                stack_pos = [{"name": "lora_a.safetensors", "strength":  1.0}]
                stack_neg = [{"name": "lora_a.safetensors", "strength": -1.0}]
                _, signs_pos = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_pos)
                _, signs_neg = lora_optimizer.LoRAAutoTuner._compute_names_only_hash(stack_neg)
                self.assertEqual(signs_pos[0],  1)
                self.assertEqual(signs_neg[0], -1)
```

**Step 3: Run test to verify failure**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests::test_names_only_hash_excludes_strength -v
```
Expected: `AttributeError: type object 'LoRAAutoTuner' has no attribute '_compute_names_only_hash'`

**Step 4: Implement `_compute_names_only_hash` on `LoRAAutoTuner`**

Add after `_memory_file_path` (~line 7420) in `lora_optimizer.py`:

```python
@staticmethod
def _compute_names_only_hash(active_loras):
    """
    Compute a hash of LoRA file identity (name + mtime + size) independent
    of strength values. Returns (hash_str, signs) where signs is
    {lora_index: +1 or -1} for sign-flip detection at synthesis.
    """
    entries = []
    for item in active_loras:
        name = item["name"]
        path = folder_paths.get_full_path("loras", name)
        if path is not None:
            try:
                st = os.stat(path)
                entries.append((name, st.st_mtime, st.st_size))
            except OSError:
                entries.append((name, 0, 0))
        else:
            entries.append((name, 0, 0))
    hash_input = json.dumps(sorted(entries), separators=(",", ":"))
    names_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    signs = {i: (1 if item["strength"] >= 0 else -1)
             for i, item in enumerate(active_loras)}
    return names_hash, signs
```

**Step 5: Run tests**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests -v
```
Expected: All PASS.

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add _compute_names_only_hash for analysis cache key"
```

---

### Task 2: Analysis cache file I/O

**Files:**
- Modify: `lora_optimizer.py` (add static methods on `LoRAAutoTuner`, after `_compute_names_only_hash`)
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing tests**

Add to `AnalysisCacheTests`:

```python
def test_analysis_cache_roundtrip(self):
    """Save and load analysis cache; content survives round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            per_prefix = {
                "prefix_a": {
                    "pair_conflicts": {"0,1": {"overlap": 100, "conflict": 30,
                                               "dot": 0.5, "norm_a_sq": 1.0,
                                               "norm_b_sq": 1.0,
                                               "weighted_total": 0.8,
                                               "weighted_conflict": 0.2,
                                               "expected_conflict": 0.15,
                                               "excess_conflict": 0.05,
                                               "subspace_overlap": 0.3,
                                               "subspace_weight": 1.0}},
                    "per_lora_norm_sq": {"0": 1.5, "1": 0.8},
                    "magnitude_samples_unscaled": {"0": [0.1, 0.2], "1": [0.3]},
                    "ranks": {"0": 16, "1": 32},
                    "target_key": "model.layer.weight",
                    "is_clip": False,
                    "raw_n": 2,
                    "skip_count": 0,
                    "strength_signs": {"0": 1, "1": 1},
                }
            }
            source_loras = [{"name": "a.safetensors", "mtime": 1.0, "size": 100}]
            lora_optimizer.LoRAAutoTuner._analysis_cache_save(
                "abc123", per_prefix, source_loras)

            loaded = lora_optimizer.LoRAAutoTuner._analysis_cache_load("abc123")
            self.assertIsNotNone(loaded)
            self.assertIn("prefix_a", loaded)
            self.assertEqual(loaded["prefix_a"]["per_lora_norm_sq"]["0"], 1.5)

def test_analysis_cache_load_missing_returns_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            self.assertIsNone(
                lora_optimizer.LoRAAutoTuner._analysis_cache_load("nonexistent"))

def test_analysis_cache_load_stale_algo_version_returns_none(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.AUTOTUNER_MEMORY_DIR", tmpdir):
            # Write a file with a stale algo_version using the expected filename
            stale_hash = "staletest1"
            path = os.path.join(tmpdir, f"{stale_hash}.analysis.json")
            with open(path, "w") as f:
                json.dump({"analysis_version": 1,
                           "algo_version": "0.0.0",
                           "per_prefix": {"prefix_a": {}}}, f)
            result = lora_optimizer.LoRAAutoTuner._analysis_cache_load(stale_hash)
            self.assertIsNone(result)
```

**Step 2: Run to verify failure**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests::test_analysis_cache_roundtrip -v
```
Expected: `AttributeError: type object 'LoRAAutoTuner' has no attribute '_analysis_cache_path'`

**Step 3: Implement**

Add after `_compute_names_only_hash` in `lora_optimizer.py`:

```python
@staticmethod
def _analysis_cache_path(names_only_hash):
    return os.path.join(AUTOTUNER_MEMORY_DIR,
                        f"{names_only_hash}.analysis.json")

@staticmethod
def _analysis_cache_load(names_only_hash):
    """Load analysis cache. Returns per_prefix dict or None on miss/stale."""
    path = LoRAAutoTuner._analysis_cache_path(names_only_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("algo_version") != AUTOTUNER_ALGO_VERSION:
            logging.info("[AutoTuner Analysis Cache] Stale algo version, ignoring")
            return None
        return data.get("per_prefix")
    except Exception as e:
        logging.warning(f"[AutoTuner Analysis Cache] Failed to load: {e}")
        return None

@staticmethod
def _analysis_cache_save(names_only_hash, per_prefix, source_loras):
    """Atomic write of analysis cache to disk."""
    from datetime import datetime
    path = LoRAAutoTuner._analysis_cache_path(names_only_hash)
    entry = {
        "analysis_version": 1,
        "algo_version": AUTOTUNER_ALGO_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_loras": source_loras,
        "per_prefix": per_prefix,
    }
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(entry, f)
        os.replace(tmp_path, path)
        logging.info(f"[AutoTuner Analysis Cache] Saved: {path}")
    except Exception as e:
        logging.warning(f"[AutoTuner Analysis Cache] Failed to save: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
```

**Step 4: Run tests**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add analysis cache I/O methods"
```

---

### Task 3: Cache extraction helper (on `LoRAOptimizer`)

Takes a raw `_analyze_target_group` result and strips strength-dependent values into a JSON-serializable dict. Lives on `LoRAOptimizer` because `_run_group_analysis` (which calls it) is there.

**Files:**
- Modify: `lora_optimizer.py` (add static method on `LoRAOptimizer`, before `_run_group_analysis` ~line 4637)
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing test**

Add to `AnalysisCacheTests`:

```python
def test_extract_for_analysis_cache_strips_strength(self):
    """Extracted data has unscaled magnitude samples; tuple keys become strings."""
    partial_stats = [(0, 16, 1.5, 2.25), (1, 32, 0.9, 0.81)]
    pair_conflicts = {
        (0, 1): {"overlap": 50, "conflict": 10, "dot": 0.4,
                 "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                 "weighted_total": 0.6, "weighted_conflict": 0.1,
                 "expected_conflict": 0.12, "excess_conflict": 0.0,
                 "subspace_overlap": 0.2, "subspace_weight": 1.35}
    }
    magnitude_samples = [
        torch.tensor([1.5, 3.0]),   # LoRA 0: already scaled by abs(strength=1.5)
        torch.tensor([0.9, 1.8]),   # LoRA 1: already scaled by abs(strength=0.9)
    ]
    per_lora_norm_sq = {0: 2.25, 1: 0.81}
    result = (
        "prefix_x", partial_stats, pair_conflicts,
        magnitude_samples, ("layer.weight", False),
        0, 2, per_lora_norm_sq,
    )
    active_loras = [
        {"name": "a.safetensors", "strength": 1.5, "clip_strength": None},
        {"name": "b.safetensors", "strength": 0.9, "clip_strength": None},
    ]
    extracted = lora_optimizer.LoRAOptimizer._extract_for_analysis_cache(
        result, active_loras)

    # Pair key must be a string
    self.assertIn("0,1", extracted["pair_conflicts"])
    # Samples unscaled: divide by abs(strength)
    self.assertAlmostEqual(extracted["magnitude_samples_unscaled"]["0"][0],
                           1.5 / 1.5, places=5)
    self.assertAlmostEqual(extracted["magnitude_samples_unscaled"]["1"][0],
                           0.9 / 0.9, places=5)
    self.assertEqual(extracted["per_lora_norm_sq"]["0"], 2.25)
    self.assertEqual(extracted["target_key"], "layer.weight")
    self.assertFalse(extracted["is_clip"])
    self.assertEqual(extracted["strength_signs"]["0"], 1)
```

**Step 2: Run to verify failure**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests::test_extract_for_analysis_cache_strips_strength -v
```
Expected: `AttributeError: type object 'LoRAOptimizer' has no attribute '_extract_for_analysis_cache'`

**Step 3: Implement on `LoRAOptimizer`**

Add before `_run_group_analysis` (~line 4637) in `lora_optimizer.py`:

```python
@staticmethod
def _extract_for_analysis_cache(result, active_loras):
    """
    Extract the strength-invariant parts of an _analyze_target_group result
    into a JSON-serializable dict for .analysis.json storage.

    result: 8-tuple (prefix, partial_stats, pair_conflicts, magnitude_samples,
                     (target_key, is_clip), skip_count, raw_n, per_lora_norm_sq)
    """
    (prefix, partial_stats, pair_conflicts, magnitude_samples,
     target_info, skip_count, raw_n, per_lora_norm_sq) = result
    target_key, is_clip = target_info

    # Serialize target_key (may be str or tuple)
    tk_serial = list(target_key) if isinstance(target_key, tuple) else target_key

    # Serialize pair_conflict keys: (i,j) -> "i,j"; values are already plain dicts
    pc_serial = {f"{i},{j}": metrics for (i, j), metrics in pair_conflicts.items()}

    # Unscale magnitude_samples: divide by abs(strength) per LoRA
    lora_indices = [s[0] for s in partial_stats]
    mag_unscaled = {}
    for pos, i in enumerate(lora_indices):
        abs_strength = abs(active_loras[i]["strength"])
        if pos < len(magnitude_samples):
            raw = magnitude_samples[pos]
            mag_unscaled[str(i)] = (raw / abs_strength if abs_strength > 0
                                    else raw).tolist()
        else:
            mag_unscaled[str(i)] = []

    return {
        "pair_conflicts": pc_serial,
        "per_lora_norm_sq": {str(k): float(v) for k, v in per_lora_norm_sq.items()},
        "magnitude_samples_unscaled": mag_unscaled,
        "ranks": {str(s[0]): s[1] for s in partial_stats},
        "target_key": tk_serial,
        "is_clip": is_clip,
        "raw_n": raw_n,
        "skip_count": skip_count,
        "strength_signs": {str(i): (1 if active_loras[i]["strength"] >= 0 else -1)
                           for i in lora_indices},
    }
```

**Step 4: Run tests**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add _extract_for_analysis_cache on LoRAOptimizer"
```

---

### Task 4: Cache reconstruction helper (on `LoRAOptimizer`)

**Files:**
- Modify: `lora_optimizer.py` (add static method on `LoRAOptimizer`, after `_extract_for_analysis_cache`)
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing tests**

Add to `AnalysisCacheTests`:

```python
def test_reconstruct_rescales_by_new_strength(self):
    """Reconstruction rescales magnitude samples and display_l2 to new strength."""
    cached_prefix = {
        "pair_conflicts": {
            "0,1": {"overlap": 50, "conflict": 10, "dot": 0.4,
                    "norm_a_sq": 2.25, "norm_b_sq": 0.81,
                    "weighted_total": 0.6, "weighted_conflict": 0.1,
                    "expected_conflict": 0.12, "excess_conflict": 0.0,
                    "subspace_overlap": 0.2, "subspace_weight": 1.35}
        },
        "per_lora_norm_sq": {"0": 2.25, "1": 0.81},
        "magnitude_samples_unscaled": {"0": [1.0, 2.0], "1": [1.0]},
        "ranks": {"0": 16, "1": 32},
        "target_key": "layer.weight",
        "is_clip": False,
        "raw_n": 2,
        "skip_count": 0,
        "strength_signs": {"0": 1, "1": 1},
    }
    active_loras = [
        {"name": "a.safetensors", "strength": 2.0, "clip_strength": None},
        {"name": "b.safetensors", "strength": 0.5, "clip_strength": None},
    ]
    result = lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
        "prefix_x", cached_prefix, active_loras)
    self.assertIsNotNone(result)
    prefix, partial_stats, pair_conflicts, mag_samples, target_info, skip_count, raw_n, norm_sq = result

    self.assertEqual(prefix, "prefix_x")
    # display_l2 = sqrt(norm_sq) * abs(strength)
    self.assertAlmostEqual(partial_stats[0][2], math.sqrt(2.25) * 2.0, places=4)
    self.assertAlmostEqual(partial_stats[1][2], math.sqrt(0.81) * 0.5, places=4)
    # pair_conflicts keys are int tuples
    self.assertIn((0, 1), pair_conflicts)
    # magnitude_samples rescaled by abs(new_strength)
    self.assertAlmostEqual(mag_samples[0][0].item(), 1.0 * 2.0, places=4)
    self.assertAlmostEqual(mag_samples[1][0].item(), 1.0 * 0.5, places=4)

def test_reconstruct_returns_none_on_sign_flip(self):
    """Sign flip triggers None so caller falls back to full analysis."""
    cached_prefix = {
        "pair_conflicts": {},
        "per_lora_norm_sq": {"0": 1.0},
        "magnitude_samples_unscaled": {"0": [1.0]},
        "ranks": {"0": 16},
        "target_key": "layer.weight",
        "is_clip": False,
        "raw_n": 1,
        "skip_count": 0,
        "strength_signs": {"0": 1},  # was positive
    }
    active_loras = [{"name": "a.safetensors", "strength": -1.0, "clip_strength": None}]
    result = lora_optimizer.LoRAOptimizer._reconstruct_from_analysis_cache(
        "prefix_x", cached_prefix, active_loras)
    self.assertIsNone(result)
```

**Step 2: Run to verify failure**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests::test_reconstruct_rescales_by_new_strength -v
```
Expected: `AttributeError: type object 'LoRAOptimizer' has no attribute '_reconstruct_from_analysis_cache'`

**Step 3: Implement on `LoRAOptimizer`**

Add after `_extract_for_analysis_cache` in `lora_optimizer.py`:

```python
@staticmethod
def _reconstruct_from_analysis_cache(prefix, cached_prefix, active_loras):
    """
    Reconstruct the 8-tuple expected by _collect_analysis_result from cached
    data and current active_loras. Returns None if any strength sign has
    flipped vs the cached signs (triggers full re-analysis for this prefix).
    """
    cached_signs = cached_prefix.get("strength_signs", {})
    per_lora_norm_sq_raw = cached_prefix["per_lora_norm_sq"]
    lora_indices = sorted(int(k) for k in per_lora_norm_sq_raw.keys())

    for i in lora_indices:
        current_sign = 1 if active_loras[i]["strength"] >= 0 else -1
        if current_sign != cached_signs.get(str(i), 1):
            logging.info(
                f"[AutoTuner Analysis Cache] Sign flip on LoRA {i} "
                f"for {prefix!r}, falling back to full analysis")
            return None

    ranks = cached_prefix["ranks"]
    partial_stats = []
    for i in lora_indices:
        norm_sq = float(per_lora_norm_sq_raw[str(i)])
        display_l2 = math.sqrt(norm_sq) * abs(active_loras[i]["strength"])
        partial_stats.append((i, int(ranks.get(str(i), 0)), display_l2, norm_sq))

    pair_conflicts = {}
    for key_str, metrics in cached_prefix["pair_conflicts"].items():
        a, b = key_str.split(",")
        pair_conflicts[(int(a), int(b))] = metrics

    mag_unscaled = cached_prefix["magnitude_samples_unscaled"]
    magnitude_samples = []
    for i in lora_indices:
        raw = mag_unscaled.get(str(i), [])
        t = torch.tensor(raw, dtype=torch.float32) * abs(active_loras[i]["strength"])
        magnitude_samples.append(t)

    tk = cached_prefix["target_key"]
    target_key = tuple(tk) if isinstance(tk, list) else tk

    per_lora_norm_sq = {int(k): float(v) for k, v in per_lora_norm_sq_raw.items()}

    return (
        prefix,
        partial_stats,
        pair_conflicts,
        magnitude_samples,
        (target_key, cached_prefix["is_clip"]),
        int(cached_prefix.get("skip_count", 0)),
        int(cached_prefix.get("raw_n", len(lora_indices))),
        per_lora_norm_sq,
    )
```

**Step 4: Run tests**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add _reconstruct_from_analysis_cache on LoRAOptimizer"
```

---

### Task 5: Integrate cache into `_run_group_analysis`

`_run_group_analysis` checks the per-prefix cache hit/miss, uses reconstruction or falls back to `_analyze_target_group`, and returns new entries in its return dict. Cache save stays in `auto_tune` (Task 6). CPU path populates `new_analysis_entries` the same way as GPU path.

**Files:**
- Modify: `lora_optimizer.py:4637` (`_run_group_analysis` signature, body, and return dict)
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing test**

Add to `AnalysisCacheTests`:

```python
def test_run_group_analysis_skips_analyze_on_cache_hit(self):
    """When cached_analysis covers a prefix, _analyze_target_group is not called."""
    optimizer = lora_optimizer.LoRAOptimizer()
    active_loras = [
        _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
        _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
    ]
    model = _make_model()
    target_groups = optimizer._build_target_groups(
        ["prefix_a"], {"prefix_a": "layer.weight"}, {})

    fake_cached = {
        "prefix_a": {
            "pair_conflicts": {
                "0,1": {"overlap": 10, "conflict": 2, "dot": 0.1,
                        "norm_a_sq": 1.0, "norm_b_sq": 0.25,
                        "weighted_total": 0.3, "weighted_conflict": 0.05,
                        "expected_conflict": 0.1, "excess_conflict": 0.0,
                        "subspace_overlap": 0.1, "subspace_weight": 0.5}
            },
            "per_lora_norm_sq": {"0": 1.0, "1": 0.25},
            "magnitude_samples_unscaled": {"0": [0.5], "1": [0.3]},
            "ranks": {"0": 1, "1": 1},
            "target_key": "layer.weight",
            "is_clip": False,
            "raw_n": 2,
            "skip_count": 0,
            "strength_signs": {"0": 1, "1": 1},
        }
    }

    call_count = {"n": 0}
    orig = optimizer._analyze_target_group
    def counting_analyze(*args, **kwargs):
        call_count["n"] += 1
        return orig(*args, **kwargs)

    with mock.patch.object(optimizer, "_analyze_target_group",
                           side_effect=counting_analyze):
        device = torch.device("cpu")
        result = optimizer._run_group_analysis(
            target_groups, active_loras, model, None, device,
            cached_analysis=fake_cached,
        )

    self.assertEqual(call_count["n"], 0)
    # new_analysis_entries should be empty (full cache hit)
    self.assertEqual(result["new_analysis_entries"], {})

def test_run_group_analysis_populates_new_entries_on_miss(self):
    """Cache miss populates new_analysis_entries in the return dict."""
    optimizer = lora_optimizer.LoRAOptimizer()
    active_loras = [
        _make_lora_entry({"prefix_a": 1.0}, strength=1.0, name="a.safetensors"),
        _make_lora_entry({"prefix_a": 0.5}, strength=1.0, name="b.safetensors"),
    ]
    model = _make_model()
    target_groups = optimizer._build_target_groups(
        ["prefix_a"], {"prefix_a": "layer.weight"}, {})

    device = torch.device("cpu")
    result = optimizer._run_group_analysis(
        target_groups, active_loras, model, None, device,
        cached_analysis={},  # empty cache = full miss
    )
    # prefix_a should have been analyzed and added to new entries
    self.assertIn("prefix_a", result["new_analysis_entries"])
```

**Step 2: Run to verify failure**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests::test_run_group_analysis_skips_analyze_on_cache_hit -v
```
Expected: `TypeError: _run_group_analysis() got an unexpected keyword argument 'cached_analysis'`

**Step 3: Modify `_run_group_analysis` signature at `lora_optimizer.py:4637`**

Add two new optional parameters:

```python
def _run_group_analysis(self, target_groups, active_loras, model, clip,
                        compute_device, clip_strength_multiplier=1.0,
                        merge_refinement="none",
                        decision_smoothing=0.0, progress_cb=None,
                        cached_analysis=None):
```

**Step 4: Add `new_analysis_entries` tracking before the group loop**

After the existing `pairs`, `pair_accum`, `all_magnitude_samples`, etc. initializations and before `group_items = list(target_groups.values())` (~line 4769), add:

```python
        new_analysis_entries = {}
```

**Step 5: Replace the GPU loop body**

Replace the existing GPU loop (lines 4770–4779):

```python
        if use_gpu:
            for target_group in group_items:
                result = self._analyze_target_group(
                    target_group, active_loras, model, clip, compute_device,
                    clip_strength_multiplier=clip_strength_multiplier,
                    merge_refinement=merge_refinement,
                )
                _collect_analysis_result(result)
                if progress_cb is not None:
                    progress_cb()
```

With:

```python
        if use_gpu:
            for target_group in group_items:
                prefix = target_group["label_prefix"]
                result = None
                if cached_analysis is not None and prefix in cached_analysis:
                    result = self._reconstruct_from_analysis_cache(
                        prefix, cached_analysis[prefix], active_loras)
                if result is None:
                    result = self._analyze_target_group(
                        target_group, active_loras, model, clip, compute_device,
                        clip_strength_multiplier=clip_strength_multiplier,
                        merge_refinement=merge_refinement,
                    )
                    if result is not None and cached_analysis is not None:
                        new_analysis_entries[prefix] = \
                            self._extract_for_analysis_cache(result, active_loras)
                _collect_analysis_result(result)
                if progress_cb is not None:
                    progress_cb()
```

**Step 6: Replace the CPU loop body**

Replace the existing CPU block (lines 4780–4794):

```python
        else:
            max_workers = min(4, max(1, len(group_items)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._analyze_target_group, target_group, active_loras,
                        model, clip, compute_device, clip_strength_multiplier,
                        merge_refinement
                    ): target_group["label_prefix"]
                    for target_group in group_items
                }
                for future in concurrent.futures.as_completed(futures):
                    _collect_analysis_result(future.result())
                    if progress_cb is not None:
                        progress_cb()
```

With:

```python
        else:
            max_workers = min(4, max(1, len(group_items)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._analyze_target_group, target_group, active_loras,
                        model, clip, compute_device, clip_strength_multiplier,
                        merge_refinement
                    ): target_group["label_prefix"]
                    for target_group in group_items
                }
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None and cached_analysis is not None:
                        prefix = result[0]
                        if prefix not in (cached_analysis or {}):
                            new_analysis_entries[prefix] = \
                                self._extract_for_analysis_cache(result, active_loras)
                    _collect_analysis_result(result)
                    if progress_cb is not None:
                        progress_cb()
```

**Step 7: Add `new_analysis_entries` to the return dict**

At the return statement (~line 4798), add `"new_analysis_entries": new_analysis_entries` to the existing dict:

```python
        return {
            "all_key_targets": all_key_targets,
            "target_groups": dict(target_groups),
            "prefix_stats": prefix_stats,
            "per_lora_stats": per_lora_stats,
            "pair_accum": pair_accum,
            "branch_energy": branch_energy,
            "all_magnitude_samples": all_magnitude_samples,
            "prefix_count": prefix_count,
            "skipped_keys": skipped_keys,
            "pairs": pairs,
            "new_analysis_entries": new_analysis_entries,
        }
```

**Step 8: Run tests**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests -v
```
Expected: All PASS.

**Step 9: Run full suite**

```
pytest tests/ -v
```
Expected: All existing tests still PASS.

**Step 10: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: integrate analysis cache into _run_group_analysis"
```

---

### Task 6: Wire hash through `auto_tune`

**Files:**
- Modify: `lora_optimizer.py:7674` (add hash computation)
- Modify: `lora_optimizer.py:7721` (extend `clear_and_run` to delete analysis cache)
- Modify: `lora_optimizer.py:7808` (pass `cached_analysis` to `_run_group_analysis`)
- Modify: `lora_optimizer.py:7829` (extract `new_analysis_entries` from `analysis_data`)
- Modify: `lora_optimizer.py:7808` area (add save after analysis)

No new tests needed — correctness is covered by Tasks 1–5. Run full suite to catch regressions.

**Step 1: Compute `names_only_hash` in `auto_tune`**

After the existing `lora_hash` computation at line ~7677, add:

```python
        names_only_hash, _strength_signs = self._compute_names_only_hash(active_loras)
        cached_analysis = self._analysis_cache_load(names_only_hash)
        if cached_analysis is not None:
            logging.info(
                f"[AutoTuner Analysis Cache] HIT — {len(cached_analysis)} prefixes cached")
        else:
            logging.info("[AutoTuner Analysis Cache] MISS — will run full analysis")
```

**Step 2: Clear analysis cache on `clear_and_run`**

In the `memory_mode == "clear_and_run"` block at line ~7721, after the existing `_memory_clear` call:

```python
            analysis_path = self._analysis_cache_path(names_only_hash)
            if os.path.exists(analysis_path):
                os.unlink(analysis_path)
            cached_analysis = None
```

**Step 3: Pass `cached_analysis` to `_run_group_analysis`**

At line ~7808, add `cached_analysis=cached_analysis` to the existing call:

```python
        analysis_data = self._run_group_analysis(
            target_groups, active_loras, model, clip, compute_device,
            clip_strength_multiplier=clip_strength_multiplier,
            merge_refinement="none",
            decision_smoothing=decision_smoothing,
            progress_cb=lambda: pbar.update(1),
            cached_analysis=cached_analysis,
        )
```

**Step 4: Save new entries after analysis**

After `analysis_data = self._run_group_analysis(...)` and its unpacking block (~line 7824), add:

```python
        new_analysis_entries = analysis_data.get("new_analysis_entries", {})
        if new_analysis_entries:
            merged = dict(cached_analysis or {})
            merged.update(new_analysis_entries)
            source_loras = [{"name": item["name"]} for item in active_loras]
            self._analysis_cache_save(names_only_hash, merged, source_loras)
```

**Step 5: Run full suite**

```
pytest tests/ -v
```
Expected: All PASS.

**Step 6: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat: wire analysis cache hash and save through auto_tune"
```

---

### Task 7: Extend dataset entry with raw analysis

**Files:**
- Modify: `lora_optimizer.py:8494` (`_save_tuner_dataset_entry` signature + body)
- Modify: `lora_optimizer.py:8299` (call site — add `new_analysis_entries` arg)
- Modify: `tests/test_lora_optimizer.py`

**Step 1: Write failing test**

Add to `AnalysisCacheTests`:

```python
def test_dataset_entry_includes_raw_analysis(self):
    """Dataset entries include raw_analysis field when new_analysis_entries provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("lora_optimizer.folder_paths.get_user_directory",
                        return_value=tmpdir):
            tuner = lora_optimizer.LoRAAutoTuner()
            tuner_data = {
                "top_n": [],
                "analysis_summary": {
                    "n_loras": 2, "prefix_count": 1,
                    "avg_conflict_ratio": 0.3, "avg_excess_conflict": 0.1,
                    "avg_subspace_overlap": 0.2, "avg_cosine_sim": 0.5,
                    "magnitude_ratio": 1.2, "decision_smoothing": 0.25,
                },
                "architecture_preset": "auto",
                "lora_hash": "abc",
            }
            prefix_stats = {
                "prefix_a": {
                    "n_loras": 2, "conflict_ratio": 0.3,
                    "excess_conflict": 0.1, "avg_cos_sim": 0.5,
                    "magnitude_ratio": 1.2, "avg_subspace_overlap": 0.2,
                    "magnitude_samples": [],
                    "per_lora_norm_sq": {0: 1.0, 1: 0.5},
                    "pairwise_dots": {},
                    "raw_n_loras": 2,
                }
            }
            new_analysis_entries = {
                "prefix_a": {
                    "pair_conflicts": {"0,1": {"overlap": 10}},
                    "per_lora_norm_sq": {"0": 1.0, "1": 0.5},
                    "magnitude_samples_unscaled": {"0": [0.5], "1": [0.3]},
                    "ranks": {"0": 1, "1": 1},
                    "target_key": "layer.weight",
                    "is_clip": False,
                    "raw_n": 2,
                    "skip_count": 0,
                    "strength_signs": {"0": 1, "1": 1},
                }
            }
            active_loras = [
                {"name": "a.safetensors", "strength": 1.0},
                {"name": "b.safetensors", "strength": 0.5},
            ]
            tuner._save_tuner_dataset_entry(
                tuner_data, active_loras, prefix_stats, "wan_video",
                names_only_hash="testhash",
                new_analysis_entries=new_analysis_entries)

            dataset_path = os.path.join(
                tmpdir, "lora_optimizer_reports", "autotuner_dataset.jsonl")
            with open(dataset_path) as f:
                entry = json.loads(f.readline())
            self.assertIn("raw_analysis", entry)
            self.assertEqual(entry["raw_analysis"]["names_only_hash"], "testhash")
            self.assertIn("prefix_a", entry["raw_analysis"]["per_prefix"])
            self.assertIn("pair_conflicts",
                          entry["raw_analysis"]["per_prefix"]["prefix_a"])
```

**Step 2: Run to verify failure**

```
pytest tests/test_lora_optimizer.py::AnalysisCacheTests::test_dataset_entry_includes_raw_analysis -v
```
Expected: `TypeError: _save_tuner_dataset_entry() got an unexpected keyword argument 'names_only_hash'`

**Step 3: Modify `_save_tuner_dataset_entry` at `lora_optimizer.py:8494`**

Change signature:

```python
def _save_tuner_dataset_entry(self, tuner_data, active_loras, prefix_stats,
                              detected_arch, names_only_hash=None,
                              new_analysis_entries=None):
```

Before the `with open(dataset_path, "a")` line, add:

```python
            raw_analysis = None
            if names_only_hash is not None and new_analysis_entries:
                raw_analysis = {
                    "names_only_hash": names_only_hash,
                    "per_prefix": new_analysis_entries,
                }
            entry["raw_analysis"] = raw_analysis
```

**Step 4: Update the call site at `lora_optimizer.py:8299`**

Change:

```python
            self._save_tuner_dataset_entry(
                tuner_data, active_loras, prefix_stats,
                getattr(self, '_detected_arch', None))
```

To:

```python
            self._save_tuner_dataset_entry(
                tuner_data, active_loras, prefix_stats,
                getattr(self, '_detected_arch', None),
                names_only_hash=names_only_hash,
                new_analysis_entries=new_analysis_entries)
```

**Step 5: Run full suite**

```
pytest tests/ -v
```
Expected: All PASS.

**Step 6: Commit**

```bash
git add lora_optimizer.py tests/test_lora_optimizer.py
git commit -m "feat: add raw_analysis field to AutoTuner dataset entries"
```
