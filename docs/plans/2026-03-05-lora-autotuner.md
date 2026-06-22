# LoRA AutoTuner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two new ComfyUI nodes — LoRAAutoTuner and LoRAMergeSelector — that sweep all merge parameters, score configurations via a hybrid heuristic+measurement approach, and output the best merge with the option to select alternatives.

**Architecture:** AutoTuner inherits from `LoRAOptimizer` (which inherits `_LoRAMergeBase`). It reuses Pass 1 analysis, adds a parameter grid generator, heuristic scorer, merge-and-measure scorer, and a two-pass workflow (explore → apply). MergeSelector also inherits `LoRAOptimizer` and takes cached `TUNER_DATA` to replay a specific config. Both nodes live in `lora_optimizer.py`.

**Tech Stack:** Python 3.10+, PyTorch, ComfyUI node API

---

### Task 1: Add parameter grid generator

**Files:**
- Modify: `lora_optimizer.py` (add method to `_LoRAMergeBase` or as standalone function near line ~1928)

**Step 1: Write `_generate_param_grid` function**

Add before the `LoRAOptimizer` class definition (line 1930). This generates all valid parameter combinations, pruning nonsensical combos:

```python
def _generate_param_grid():
    """Generate all valid merge parameter combinations for AutoTuner sweep."""
    grid = []
    merge_modes = ["weighted_average", "slerp", "consensus", "ties"]
    sparsifications = ["disabled", "dare", "della"]
    densities = [0.5, 0.7, 0.9]
    dampenings = [0.0, 0.3, 0.6]
    qualities = ["standard", "enhanced", "maximum"]
    auto_strengths = ["enabled", "disabled"]
    opt_modes = ["per_prefix", "global"]

    for mode in merge_modes:
        for spars in sparsifications:
            density_vals = densities if spars != "disabled" else [0.7]
            for density in density_vals:
                damp_vals = dampenings if spars == "dare" else [0.0]
                for dampening in damp_vals:
                    for quality in qualities:
                        for auto_str in auto_strengths:
                            for opt_mode in opt_modes:
                                grid.append({
                                    "merge_mode": mode,
                                    "sparsification": spars,
                                    "sparsification_density": density,
                                    "dare_dampening": dampening,
                                    "merge_quality": quality,
                                    "auto_strength": auto_str,
                                    "optimization_mode": opt_mode,
                                })
    return grid
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): add parameter grid generator"
```

---

### Task 2: Add heuristic scoring function

**Files:**
- Modify: `lora_optimizer.py` (add method after `_generate_param_grid`, before `LoRAOptimizer` class)

**Step 1: Write `_score_config_heuristic` function**

This scores a parameter config against Pass 1 analysis metrics WITHOUT merging. It predicts how well each config matches the data profile:

```python
def _score_config_heuristic(config, avg_conflict_ratio, avg_cos_sim,
                            magnitude_ratio, prefix_stats):
    """
    Score a merge config against analysis metrics (no merge needed).
    Returns float score in [0, 1] where higher = better predicted quality.
    """
    mode = config["merge_mode"]
    spars = config["sparsification"]
    density = config["sparsification_density"]
    quality = config["merge_quality"]
    auto_str = config["auto_strength"]
    opt_mode = config["optimization_mode"]

    score = 0.0

    # --- Mode fit score (0-0.4) ---
    # How well does the chosen mode match the data profile?
    if mode == "consensus":
        # Consensus excels with high similarity, low conflict
        if avg_cos_sim > 0.5 and avg_conflict_ratio < 0.15:
            score += 0.4
        elif avg_cos_sim > 0.3 and avg_conflict_ratio < 0.25:
            score += 0.25
        else:
            score += 0.05  # poor fit
    elif mode == "slerp":
        # SLERP excels with low-to-moderate conflict, any alignment
        if avg_conflict_ratio < 0.30:
            score += 0.35
        elif avg_conflict_ratio < 0.50:
            score += 0.20
        else:
            score += 0.10
    elif mode == "weighted_average":
        # Weighted average: decent baseline, best for orthogonal LoRAs
        if abs(avg_cos_sim) < 0.25 and avg_conflict_ratio < 0.60:
            score += 0.30
        elif avg_conflict_ratio < 0.40:
            score += 0.20
        else:
            score += 0.10
    elif mode == "ties":
        # TIES excels with high conflict
        if avg_conflict_ratio > 0.25:
            score += 0.35
        elif avg_conflict_ratio > 0.15:
            score += 0.20
        else:
            score += 0.10  # overkill for low conflict

    # --- Sparsification fit (0-0.15) ---
    if spars != "disabled":
        # Sparsification helps when there's conflict to reduce
        conflict_benefit = min(avg_conflict_ratio / 0.5, 1.0) * 0.10
        score += conflict_benefit
        # Density: lower keeps less, higher preserves more
        # Moderate density (0.7) is generally safest
        density_penalty = abs(density - 0.7) * 0.05
        score += 0.05 - density_penalty
    else:
        # No sparsification: slight bonus for low-conflict stacks
        if avg_conflict_ratio < 0.15:
            score += 0.10

    # --- Quality fit (0-0.15) ---
    if quality == "maximum":
        # Maximum quality always helps but diminishing returns for easy merges
        conflict_benefit = min(avg_conflict_ratio / 0.3, 1.0) * 0.10
        score += 0.05 + conflict_benefit
    elif quality == "enhanced":
        score += 0.08 + min(avg_conflict_ratio / 0.3, 1.0) * 0.07
    else:
        # Standard: slight bonus for very clean merges (no overhead needed)
        if avg_conflict_ratio < 0.10:
            score += 0.10
        else:
            score += 0.05

    # --- Auto-strength fit (0-0.15) ---
    if auto_str == "enabled":
        # Auto-strength helps with unbalanced magnitudes
        if magnitude_ratio > 2.0:
            score += 0.15
        elif magnitude_ratio > 1.5:
            score += 0.10
        else:
            score += 0.05
    else:
        # Manual strength: slight bonus for balanced stacks
        if magnitude_ratio < 1.5:
            score += 0.10
        else:
            score += 0.03

    # --- Optimization mode fit (0-0.15) ---
    if opt_mode == "per_prefix":
        # Per-prefix is generally better when prefixes vary
        if prefix_stats:
            conflict_ratios = [ps["conflict_ratio"] for ps in prefix_stats.values()
                               if ps.get("n_loras", 0) > 1]
            if conflict_ratios:
                conflict_variance = max(conflict_ratios) - min(conflict_ratios)
                if conflict_variance > 0.2:
                    score += 0.15  # high variance = per-prefix really helps
                else:
                    score += 0.10
            else:
                score += 0.10
        else:
            score += 0.10
    else:
        # Global: simpler, slight bonus for uniform stacks
        score += 0.07

    return score
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): add heuristic scoring function"
```

---

### Task 3: Add merge-and-measure scoring function

**Files:**
- Modify: `lora_optimizer.py` (add method after `_score_config_heuristic`)

**Step 1: Write `_score_merge_result` function**

This measures quality metrics on an actual merged result (patches dict). Called only for top-N candidates:

```python
def _score_merge_result(model_patches, clip_patches):
    """
    Score an actual merge result by measuring output quality metrics.
    Returns dict with individual metrics and composite score in [0, 1].
    """
    import torch

    norms = []
    effective_ranks = []
    sparsities = []

    all_patches = list(model_patches.values()) + list(clip_patches.values())
    for patch in all_patches:
        # Extract the tensor from patch format
        if patch is None:
            continue
        if isinstance(patch, tuple) and len(patch) >= 2:
            # ("diff", (tensor,)) format
            tensor = patch[1][0] if isinstance(patch[1], tuple) else patch[1]
        elif hasattr(patch, 'diff'):
            # LoRAAdapter format — get effective diff
            tensor = None  # low-rank, skip full analysis
        else:
            continue

        if tensor is None:
            continue

        t = tensor.float()

        # Frobenius norm
        norms.append(t.norm().item())

        # Effective rank via spectral analysis (sample for speed)
        if t.dim() == 2 and min(t.shape) > 1:
            try:
                # Use a small number of singular values for speed
                k = min(min(t.shape), 64)
                s = torch.linalg.svdvals(t)[:k]
                s_norm = s / (s.sum() + 1e-10)
                # Shannon entropy -> effective rank
                entropy = -(s_norm * (s_norm + 1e-10).log()).sum().item()
                eff_rank = min(math.exp(entropy), min(t.shape))
                effective_ranks.append(eff_rank)
            except Exception:
                pass

        # Sparsity (fraction of near-zero weights)
        threshold = t.abs().max().item() * 0.01
        if threshold > 0:
            sparsity = (t.abs() < threshold).float().mean().item()
            sparsities.append(sparsity)

    metrics = {}

    # Norm stats
    if norms:
        metrics["norm_mean"] = sum(norms) / len(norms)
        metrics["norm_std"] = (sum((n - metrics["norm_mean"])**2 for n in norms)
                               / len(norms)) ** 0.5
        # Lower coefficient of variation = more uniform = better
        metrics["norm_cv"] = metrics["norm_std"] / (metrics["norm_mean"] + 1e-10)
    else:
        metrics["norm_mean"] = 0.0
        metrics["norm_cv"] = 1.0

    # Effective rank (higher = more expressive)
    if effective_ranks:
        metrics["effective_rank_mean"] = sum(effective_ranks) / len(effective_ranks)
    else:
        metrics["effective_rank_mean"] = 0.0

    # Sparsity (moderate is ideal — too sparse = lost info, too dense = noisy)
    if sparsities:
        avg_sparsity = sum(sparsities) / len(sparsities)
        metrics["sparsity_mean"] = avg_sparsity
        # Ideal sparsity around 0.3-0.5
        metrics["sparsity_fit"] = 1.0 - abs(avg_sparsity - 0.4) * 2.0
        metrics["sparsity_fit"] = max(0.0, metrics["sparsity_fit"])
    else:
        metrics["sparsity_mean"] = 0.0
        metrics["sparsity_fit"] = 0.5

    # Composite score
    score = 0.0
    # Higher effective rank = more expressive (0-0.4)
    if metrics["effective_rank_mean"] > 0:
        # Normalize: rank 10-50 is typical range
        rank_score = min(metrics["effective_rank_mean"] / 40.0, 1.0)
        score += rank_score * 0.4

    # Lower norm CV = more uniform patches (0-0.3)
    cv_score = max(0.0, 1.0 - metrics["norm_cv"])
    score += cv_score * 0.3

    # Sparsity fitness (0-0.3)
    score += metrics["sparsity_fit"] * 0.3

    metrics["composite_score"] = score
    return metrics
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): add merge-and-measure scoring function"
```

---

### Task 4: Add LoRAAutoTuner node class

**Files:**
- Modify: `lora_optimizer.py` (add new class after `LoRAOptimizerSimple`, before `WanVideoLoRAOptimizer` at line ~3691)

**Step 1: Write the LoRAAutoTuner class**

This is the main node. It inherits from `LoRAOptimizer` and overrides `optimize_merge` to add the sweep logic:

```python
class LoRAAutoTuner(LoRAOptimizer):
    """
    Automatic parameter sweep that finds the best merge configuration for
    a given LoRA stack.  Runs Pass 1 analysis once, scores all parameter
    combinations via heuristic, then merges top-N candidates and measures
    output quality.  Outputs the #1 result as MODEL/CLIP, plus a ranked
    report and TUNER_DATA for optional override via a Merge Selector node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model to merge LoRAs into."
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack from a LoRA Stack node."
                }),
                "output_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Master volume for the merged result."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP model for text-encoder LoRA keys."
                }),
                "clip_strength_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for CLIP LoRA strengths."
                }),
                "top_n": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of top configurations to evaluate via actual merge. Higher = slower but explores more options."
                }),
                "normalize_keys": (["disabled", "enabled"], {
                    "default": "disabled",
                    "tooltip": "Makes LoRAs from different training tools compatible."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "TUNER_DATA")
    RETURN_NAMES = ("model", "clip", "report", "tuner_data")
    FUNCTION = "auto_tune"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Automatically sweeps all merge parameters and finds the best "
        "configuration for your LoRA stack. Outputs the best merge directly. "
        "Connect TUNER_DATA to a Merge Selector node to try alternatives."
    )

    def auto_tune(self, model, lora_stack, output_strength, clip=None,
                  clip_strength_multiplier=1.0, top_n=3, normalize_keys="disabled"):
        import hashlib, json

        # --- Normalize & validate stack ---
        active_loras = self._normalize_stack(lora_stack, normalize_keys=normalize_keys)
        if not active_loras:
            return (model, clip, "No active LoRAs in stack.", None)

        # Compute lora_hash for cache validation
        hash_input = json.dumps([(l["name"], l["strength"]) for l in active_loras],
                                sort_keys=True)
        lora_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # --- Pass 1: Analysis (run once, reuse for all configs) ---
        # We need the raw analysis data. Run the parent's Pass 1 logic
        # by calling optimize_merge with a neutral config and extracting
        # the analysis.  Instead, we replicate the analysis accumulation
        # here to avoid a full merge.

        model_keys = self._get_model_keys(model)
        clip_keys = self._get_model_keys(clip) if clip is not None else set()

        # Detect architecture
        first_lora_sd = self._load_lora(active_loras[0]["name"])
        detected_arch = self._detect_architecture(first_lora_sd)
        if normalize_keys == "enabled":
            for item in active_loras:
                lora_sd = self._load_lora(item["name"])
                self._normalize_keys(lora_sd, detected_arch)

        # Collect all prefixes
        all_lora_prefixes = set()
        for item in active_loras:
            lora_sd = self._load_lora(item["name"])
            for key in lora_sd:
                if key.endswith(".lora_up.weight") or key.endswith(".lora_B.weight"):
                    prefix = key.rsplit(".", 2)[0]
                    all_lora_prefixes.add(prefix)
        all_lora_prefixes = sorted(all_lora_prefixes)

        if not all_lora_prefixes:
            return (model, clip, "No LoRA prefixes found.", None)

        # Determine compute device
        compute_device = self._get_compute_device()
        use_gpu = compute_device.type == "cuda"

        # Run Pass 1 analysis (same as parent)
        all_key_targets = {}
        skipped_keys = 0
        per_lora_stats = [{
            "name": item["name"], "strength": item["strength"],
            "ranks": [], "key_count": 0, "l2_norms": [],
        } for item in active_loras]

        pairs = [(i, j) for i in range(len(active_loras))
                 for j in range(i + 1, len(active_loras))]
        pair_accum = {(i, j): [0, 0, 0.0, 0.0, 0.0] for i, j in pairs}
        all_magnitude_samples = []
        prefix_count = 0
        prefix_stats = {}

        def _collect_analysis(result):
            nonlocal skipped_keys, prefix_count
            if result is None:
                return
            prefix, partial_stats, pair_conflicts, mag_samples, target_info, skips = result
            skipped_keys += skips
            if len(partial_stats) > 0:
                all_key_targets[prefix] = target_info
                prefix_count += 1
            for (idx, rank, l2) in partial_stats:
                per_lora_stats[idx]["ranks"].append(rank)
                per_lora_stats[idx]["key_count"] += 1
                per_lora_stats[idx]["l2_norms"].append(l2)
            for (i, j), (ov, conf, dot, na_sq, nb_sq) in pair_conflicts.items():
                pair_accum[(i, j)][0] += ov
                pair_accum[(i, j)][1] += conf
                pair_accum[(i, j)][2] += dot
                pair_accum[(i, j)][3] += na_sq
                pair_accum[(i, j)][4] += nb_sq
            all_magnitude_samples.extend(mag_samples)
            if len(partial_stats) > 0:
                n_contributing = len(partial_stats)
                pf_overlap = sum(ov for ov, conf, dot, na_sq, nb_sq in pair_conflicts.values())
                pf_conflict = sum(conf for ov, conf, dot, na_sq, nb_sq in pair_conflicts.values())
                pf_conflict_ratio = pf_conflict / pf_overlap if pf_overlap > 0 else 0.0
                pf_l2s = [l2 for _, _, l2 in partial_stats if l2 > 0]
                pf_mag_ratio = max(pf_l2s) / min(pf_l2s) if len(pf_l2s) >= 2 else 1.0
                pf_cos_sims = []
                for (ov, conf, dot, na_sq, nb_sq) in pair_conflicts.values():
                    denom = (na_sq ** 0.5) * (nb_sq ** 0.5)
                    if denom > 0:
                        pf_cos_sims.append(dot / denom)
                avg_cs = sum(pf_cos_sims) / len(pf_cos_sims) if pf_cos_sims else 0.0
                prefix_stats[prefix] = {
                    "n_loras": n_contributing,
                    "conflict_ratio": pf_conflict_ratio,
                    "magnitude_ratio": pf_mag_ratio,
                    "magnitude_samples": list(mag_samples),
                    "avg_cos_sim": avg_cs,
                }

        logging.info(f"[LoRA AutoTuner] Pass 1: Analyzing {len(all_lora_prefixes)} prefixes...")
        t_start = time.time()
        if use_gpu:
            for lora_prefix in all_lora_prefixes:
                result = self._analyze_prefix(lora_prefix, active_loras,
                                              model_keys, clip_keys, model, clip, compute_device)
                _collect_analysis(result)
        else:
            max_workers = min(4, max(1, len(all_lora_prefixes)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_prefix, lora_prefix, active_loras,
                                    model_keys, clip_keys, model, clip, compute_device): lora_prefix
                    for lora_prefix in all_lora_prefixes
                }
                for future in concurrent.futures.as_completed(futures):
                    _collect_analysis(future.result())

        if prefix_count == 0:
            return (model, clip, "No compatible LoRA keys found.", None)

        t_analysis = time.time() - t_start
        logging.info(f"[LoRA AutoTuner] Analysis complete: {prefix_count} prefixes ({t_analysis:.1f}s)")

        # Finalize global stats
        lora_stats = []
        l2_means = []
        for i, stat in enumerate(per_lora_stats):
            avg_rank = sum(stat["ranks"]) / len(stat["ranks"]) if stat["ranks"] else 0
            l2_mean = sum(stat["l2_norms"]) / len(stat["l2_norms"]) if stat["l2_norms"] else 0
            l2_means.append(l2_mean)
            lora_stats.append({
                "name": stat["name"], "strength": stat["strength"],
                "key_count": stat["key_count"], "avg_rank": avg_rank,
                "l2_mean": l2_mean,
                "conflict_mode": active_loras[i].get("conflict_mode", "all"),
                "key_filter": active_loras[i].get("key_filter", "all"),
            })

        total_overlap = sum(pair_accum[p][0] for p in pairs)
        total_conflict = sum(pair_accum[p][1] for p in pairs)
        avg_conflict_ratio = total_conflict / total_overlap if total_overlap > 0 else 0

        pairwise_similarities = {}
        for i, j in pairs:
            ov, conf, dot, na_sq, nb_sq = pair_accum[(i, j)]
            denom = math.sqrt(na_sq) * math.sqrt(nb_sq)
            pairwise_similarities[(i, j)] = dot / denom if denom > 0 else 0.0

        valid_l2 = [m for m in l2_means if m > 0]
        magnitude_ratio = max(valid_l2) / min(valid_l2) if len(valid_l2) >= 2 else 1.0

        avg_cos_sim = (sum(pairwise_similarities.values())
                       / len(pairwise_similarities)) if pairwise_similarities else 0.0

        del all_magnitude_samples

        # --- Phase 1: Score all parameter combos (heuristic, fast) ---
        logging.info("[LoRA AutoTuner] Phase 1: Scoring parameter grid...")
        grid = _generate_param_grid()
        scored = []
        for config in grid:
            h_score = _score_config_heuristic(
                config, avg_conflict_ratio, avg_cos_sim,
                magnitude_ratio, prefix_stats)
            scored.append((h_score, config))
        scored.sort(key=lambda x: x[0], reverse=True)
        logging.info(f"[LoRA AutoTuner] Scored {len(grid)} combos, top heuristic: {scored[0][0]:.3f}")

        # --- Phase 2: Merge top-N and measure ---
        top_candidates = scored[:top_n]
        results = []

        for rank_idx, (h_score, config) in enumerate(top_candidates):
            logging.info(f"[LoRA AutoTuner] Phase 2: Merging candidate #{rank_idx + 1} "
                         f"({config['merge_mode']}, {config['merge_quality']})...")
            t_merge = time.time()

            merged_model, merged_clip, report, lora_data = super().optimize_merge(
                model, lora_stack, output_strength,
                clip=clip,
                clip_strength_multiplier=clip_strength_multiplier,
                auto_strength=config["auto_strength"],
                optimization_mode=config["optimization_mode"],
                sparsification=config["sparsification"],
                sparsification_density=config["sparsification_density"],
                dare_dampening=config["dare_dampening"],
                merge_quality=config["merge_quality"],
                merge_strategy_override=config["merge_mode"],
                # Fixed defaults for sweep
                free_vram_between_passes="disabled",
                cache_patches="disabled",
                compress_patches="non_ties",
                svd_device="gpu",
                normalize_keys=normalize_keys,
                behavior_profile="v1.2",
            )

            # Measure output quality
            m_patches = lora_data["model_patches"] if lora_data else {}
            c_patches = lora_data["clip_patches"] if lora_data else {}
            measured = _score_merge_result(m_patches, c_patches)

            t_elapsed = time.time() - t_merge
            logging.info(f"[LoRA AutoTuner]   Candidate #{rank_idx + 1}: "
                         f"measured={measured['composite_score']:.3f} ({t_elapsed:.1f}s)")

            # Extract auto-adjusted strengths from report if available
            adjusted_strengths = None
            if config["auto_strength"] == "enabled" and lora_data:
                adjusted_strengths = [ls["strength"] for ls in lora_stats]

            results.append({
                "rank": rank_idx + 1,
                "score_heuristic": h_score,
                "score_measured": measured["composite_score"],
                "config": config,
                "metrics": {
                    "norm_preservation": measured.get("norm_mean", 0.0),
                    "effective_rank_mean": measured.get("effective_rank_mean", 0.0),
                    "sparsity_mean": measured.get("sparsity_mean", 0.0),
                    "norm_cv": measured.get("norm_cv", 0.0),
                },
                "merged_model": merged_model,
                "merged_clip": merged_clip,
                "lora_data": lora_data,
            })

        # Sort by measured score (Phase 2 replaces Phase 1 ranking)
        results.sort(key=lambda x: x["score_measured"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        # Best result
        best = results[0]

        # Build TUNER_DATA (exclude model/clip objects — just configs and scores)
        tuner_data = {
            "version": 1,
            "lora_hash": lora_hash,
            "analysis_summary": {
                "n_loras": len(active_loras),
                "prefix_count": prefix_count,
                "avg_conflict_ratio": avg_conflict_ratio,
                "avg_cosine_sim": avg_cos_sim,
                "magnitude_ratio": magnitude_ratio,
            },
            "top_n": [{
                "rank": r["rank"],
                "score_heuristic": r["score_heuristic"],
                "score_measured": r["score_measured"],
                "config": r["config"],
                "metrics": r["metrics"],
            } for r in results],
        }

        # Build report
        report = self._build_autotuner_report(
            results, tuner_data["analysis_summary"], output_strength)

        return (best["merged_model"], best["merged_clip"], report, tuner_data)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): add LoRAAutoTuner node class"
```

---

### Task 5: Add AutoTuner report builder

**Files:**
- Modify: `lora_optimizer.py` (add `_build_autotuner_report` method to `LoRAAutoTuner` class)

**Step 1: Write the report builder method**

Add inside the `LoRAAutoTuner` class, after `auto_tune`:

```python
    def _build_autotuner_report(self, results, analysis_summary, output_strength):
        """Build the ranked report for AutoTuner results."""
        lines = []
        lines.append("=" * 54)
        lines.append("  LoRA AutoTuner Results")
        lines.append("=" * 54)
        lines.append("")
        lines.append("  Analysis Summary:")
        s = analysis_summary
        lines.append(f"    LoRAs: {s['n_loras']} | Prefixes: {s['prefix_count']} "
                     f"| Avg conflict: {s['avg_conflict_ratio']:.1%}")
        lines.append(f"    Avg cosine similarity: {s['avg_cosine_sim']:.2f} "
                     f"| Magnitude ratio: {s['magnitude_ratio']:.1f}x")
        lines.append("")
        lines.append("  " + "-" * 38)
        lines.append(f"  Top {len(results)} Configurations")
        lines.append("  " + "-" * 38)

        for r in results:
            lines.append("")
            c = r["config"]
            m = r["metrics"]
            marker = " (applied to output)" if r["rank"] == 1 else ""
            star = " \u2605" if r["rank"] == 1 else ""
            lines.append(f"  #{r['rank']}{star}{marker}"
                         f"          Score: {r['score_measured']:.2f}")
            lines.append(f"    Mode: {c['merge_mode']} | Quality: {c['merge_quality']}")
            if c["sparsification"] != "disabled":
                spars_info = f"{c['sparsification']} (density={c['sparsification_density']}"
                if c["dare_dampening"] > 0:
                    spars_info += f", dampening={c['dare_dampening']}"
                spars_info += ")"
                lines.append(f"    Sparsification: {spars_info}")
            else:
                lines.append(f"    Sparsification: disabled")
            lines.append(f"    Auto-strength: {c['auto_strength']} "
                         f"| Optimization: {c['optimization_mode']}")
            if m.get("effective_rank_mean", 0) > 0:
                lines.append(f"    Effective rank: {m['effective_rank_mean']:.1f} "
                             f"| Sparsity: {m.get('sparsity_mean', 0):.1%}")

        lines.append("")
        lines.append("  To use a different config: connect TUNER_DATA")
        lines.append("  to a Merge Selector node and set selection=N")
        lines.append("=" * 54)
        return "\n".join(lines)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): add report builder for AutoTuner"
```

---

### Task 6: Add LoRAMergeSelector node class

**Files:**
- Modify: `lora_optimizer.py` (add new class after `LoRAAutoTuner`)

**Step 1: Write the LoRAMergeSelector class**

```python
class LoRAMergeSelector(LoRAOptimizer):
    """
    Applies a specific merge configuration from AutoTuner results.
    Connect TUNER_DATA from a LoRA AutoTuner node and set the selection
    index to choose which ranked configuration to apply.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Base model (same one connected to the AutoTuner)."
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "LoRA stack (same one connected to the AutoTuner)."
                }),
                "tuner_data": ("TUNER_DATA", {
                    "tooltip": "Connect from the LoRA AutoTuner's tuner_data output."
                }),
                "selection": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Which ranked configuration to apply (1 = best, 2 = second best, etc.)."
                }),
                "output_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Master volume for the merged result."
                }),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional CLIP model for text-encoder LoRA keys."
                }),
                "clip_strength_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                    "tooltip": "Multiplier for CLIP LoRA strengths."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "report")
    FUNCTION = "select_merge"
    CATEGORY = "LoRA Optimizer"
    DESCRIPTION = (
        "Applies a specific merge configuration from LoRA AutoTuner results. "
        "Set selection to choose which ranked configuration to use."
    )

    def select_merge(self, model, lora_stack, tuner_data, selection,
                     output_strength, clip=None, clip_strength_multiplier=1.0):
        import hashlib, json

        if tuner_data is None or "top_n" not in tuner_data:
            return (model, clip, "Error: No valid TUNER_DATA provided.")

        # Validate lora_hash
        active_loras = self._normalize_stack(lora_stack)
        hash_input = json.dumps([(l["name"], l["strength"]) for l in active_loras],
                                sort_keys=True)
        current_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        if current_hash != tuner_data.get("lora_hash", ""):
            logging.warning("[Merge Selector] LoRA stack has changed since AutoTuner ran. "
                            "Results may not match. Re-run AutoTuner for accurate results.")

        # Get selected config
        top_n = tuner_data["top_n"]
        if selection < 1 or selection > len(top_n):
            return (model, clip,
                    f"Error: selection={selection} out of range (1-{len(top_n)}).")

        entry = top_n[selection - 1]
        config = entry["config"]

        logging.info(f"[Merge Selector] Applying config #{selection}: "
                     f"{config['merge_mode']}, {config['merge_quality']}")

        # Run merge with selected config
        merged_model, merged_clip, _report, _lora_data = super().optimize_merge(
            model, lora_stack, output_strength,
            clip=clip,
            clip_strength_multiplier=clip_strength_multiplier,
            auto_strength=config["auto_strength"],
            optimization_mode=config["optimization_mode"],
            sparsification=config["sparsification"],
            sparsification_density=config["sparsification_density"],
            dare_dampening=config["dare_dampening"],
            merge_quality=config["merge_quality"],
            merge_strategy_override=config["merge_mode"],
            free_vram_between_passes="disabled",
            cache_patches="enabled",
            compress_patches="non_ties",
            svd_device="gpu",
            normalize_keys="disabled",
            behavior_profile="v1.2",
        )

        # Build report for this selection
        lines = []
        lines.append(f"Merge Selector — Applied config #{selection}")
        lines.append(f"  Mode: {config['merge_mode']} | Quality: {config['merge_quality']}")
        if config["sparsification"] != "disabled":
            lines.append(f"  Sparsification: {config['sparsification']} "
                         f"(density={config['sparsification_density']})")
        lines.append(f"  Auto-strength: {config['auto_strength']} "
                     f"| Optimization: {config['optimization_mode']}")
        lines.append(f"  Heuristic score: {entry['score_heuristic']:.3f} "
                     f"| Measured score: {entry['score_measured']:.3f}")
        report = "\n".join(lines)

        return (merged_model, merged_clip, report)

    @classmethod
    def IS_CHANGED(cls, model, lora_stack, tuner_data, selection,
                   output_strength, clip=None, clip_strength_multiplier=1.0):
        return (id(tuner_data), selection, output_strength, clip_strength_multiplier)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): add LoRAMergeSelector node class"
```

---

### Task 7: Register nodes in mappings

**Files:**
- Modify: `lora_optimizer.py` (lines 4400-4422 — NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS)

**Step 1: Add both nodes to mappings**

Add to `NODE_CLASS_MAPPINGS` (after the `WanVideoLoRAOptimizer` entry):
```python
    "LoRAAutoTuner": LoRAAutoTuner,
    "LoRAMergeSelector": LoRAMergeSelector,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS`:
```python
    "LoRAAutoTuner": "LoRA AutoTuner",
    "LoRAMergeSelector": "Merge Selector",
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): register AutoTuner and MergeSelector nodes"
```

---

### Task 8: Final verification and cleanup

**Step 1: Full syntax check**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: OK

**Step 2: Check imports**

Verify that `math`, `time`, `logging`, `concurrent.futures`, and `hashlib` are already imported at the top of `lora_optimizer.py`. If `hashlib` or `json` are missing from top-level imports, they're imported inline in the methods (via `import hashlib, json`), which is fine.

**Step 3: Smoke test import**

Run: `python -c "from lora_optimizer import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"`
Expected: List including `LoRAAutoTuner` and `LoRAMergeSelector`

**Step 4: Commit**

```bash
git add lora_optimizer.py
git commit -m "feat(autotuner): final verification pass"
```
