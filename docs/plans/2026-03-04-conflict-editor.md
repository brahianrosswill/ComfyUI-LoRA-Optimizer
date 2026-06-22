# LoRA Conflict Editor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a LoRAConflictEditor node that analyzes pairwise LoRA conflicts, auto-suggests per-LoRA conflict modes, and lets users manually override conflict modes and merge strategy before the optimizer runs.

**Architecture:** Pass-through node between stacker and optimizer. Takes LORA_STACK in, runs pairwise conflict analysis (reusing optimizer internals), resolves conflict_mode per LoRA (auto-suggest or manual), and outputs an enriched LORA_STACK. Inherits from `_LoRAMergeBase` to reuse LoRA loading, normalization, and conflict analysis methods.

**Tech Stack:** Python (ComfyUI node), JS (widget visibility)

---

### Task 1: Define LoRAConflictEditor class skeleton

**Files:**
- Modify: `lora_optimizer.py:3013-3025` (NODE_CLASS_MAPPINGS registration)
- Modify: `lora_optimizer.py` (new class, insert before NODE_CLASS_MAPPINGS)

**Step 1: Add the class with INPUT_TYPES**

Insert before `NODE_CLASS_MAPPINGS` (line 3013):

```python
class LoRAConflictEditor(_LoRAMergeBase):
    """
    Analyzes pairwise conflicts between LoRAs and lets users control
    per-LoRA conflict modes and merge strategy. Sits between a stacker
    and the optimizer.
    """

    MAX_LORAS = 10

    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "Connect a LoRA Stack node here. The editor will analyze conflicts between these LoRAs."
                }),
                "merge_strategy": (["auto", "ties", "weighted_average", "weighted_sum"], {
                    "default": "auto",
                    "tooltip": "Merge strategy to pass to the optimizer. "
                               "'auto': let the optimizer decide based on conflict analysis. "
                               "'ties': force TIES merging (good for high-conflict stacks). "
                               "'weighted_average': force simple averaging (good for compatible LoRAs). "
                               "'weighted_sum': force direct addition (preserves all weights exactly)."
                }),
            }
        }
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["required"][f"conflict_mode_{i}"] = (["auto", "all", "low_conflict", "high_conflict"], {
                "default": "auto",
                "tooltip": f"LoRA #{i} conflict filter. "
                           f"'auto': node suggests based on analysis. "
                           f"'all': apply everywhere. "
                           f"'low_conflict': only where this LoRA agrees with the majority. "
                           f"'high_conflict': only where this LoRA disagrees."
            })
        return inputs

    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "analysis_report", "merge_strategy")
    FUNCTION = "analyze_and_enrich"
    CATEGORY = "loaders/lora"
    DESCRIPTION = "Analyzes LoRA conflicts and lets you control per-LoRA conflict modes and merge strategy. Connect between a LoRA Stack and the LoRA Optimizer."
```

**Step 2: Register in NODE_CLASS_MAPPINGS**

Add to `NODE_CLASS_MAPPINGS` (line 3013):
```python
"LoRAConflictEditor": LoRAConflictEditor,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS` (line 3020):
```python
"LoRAConflictEditor": "LoRA Conflict Editor",
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**
```
feat: add LoRAConflictEditor class skeleton
```

---

### Task 2: Implement analyze_and_enrich method

**Files:**
- Modify: `lora_optimizer.py` (LoRAConflictEditor class)

**Step 1: Add the analysis and enrichment logic**

Add to LoRAConflictEditor class:

```python
    def analyze_and_enrich(self, lora_stack, merge_strategy="auto", **kwargs):
        """
        Analyze pairwise conflicts, resolve conflict modes, enrich the stack.
        """
        if not lora_stack or len(lora_stack) == 0:
            return (lora_stack, "No LoRAs in stack.", merge_strategy)

        # Normalize to consistent dict format
        normalized = self._normalize_stack(lora_stack)
        active_loras = [item for item in normalized if item["strength"] != 0]

        if len(active_loras) == 0:
            return (lora_stack, "No active LoRAs (all zero strength).", merge_strategy)

        n_loras = len(active_loras)

        # --- Pairwise conflict analysis ---
        compute_device = self._get_compute_device()
        use_gpu = (compute_device is not None and compute_device.type == "cuda")

        # Collect all unique prefixes across all LoRAs
        all_prefixes = set()
        for item in active_loras:
            for key in item["lora"].keys():
                prefix = key.rsplit(".", 1)[0] if "." in key else key
                # Only track LoRA weight prefixes (contain lora_up or lora_down)
                if "lora_up" in key or "lora_down" in key:
                    # Strip the lora_up/lora_down suffix to get the base prefix
                    base = key
                    for suffix in (".lora_up.weight", ".lora_down.weight",
                                   ".lora_A.weight", ".lora_B.weight",
                                   ".lora_up", ".lora_down"):
                        if base.endswith(suffix):
                            base = base[:-len(suffix)]
                            break
                    all_prefixes.add(base)

        # Per-LoRA pairwise accumulators
        pairs = [(i, j) for i in range(n_loras) for j in range(i + 1, n_loras)]
        pair_accum = {(i, j): [0, 0, 0.0, 0.0, 0.0] for i, j in pairs}
        per_lora_key_count = [0] * n_loras

        for lora_prefix in all_prefixes:
            # Compute diffs for each LoRA at this prefix
            diffs = []
            indices = []
            for idx, item in enumerate(active_loras):
                lora_info = self._get_lora_key_info(item["lora"], lora_prefix)
                if lora_info is None:
                    continue
                mat_up, mat_down, alpha, mid = lora_info
                rank = mat_down.shape[0]

                if use_gpu:
                    mat_up = mat_up.to(compute_device)
                    mat_down = mat_down.to(compute_device)
                    if mid is not None:
                        mid = mid.to(compute_device)

                if mid is not None:
                    final_shape = [mat_down.shape[1], mat_down.shape[0],
                                   mid.shape[2], mid.shape[3]]
                    mat_down = (
                        torch.mm(
                            mat_down.transpose(0, 1).flatten(start_dim=1).float(),
                            mid.transpose(0, 1).flatten(start_dim=1).float(),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )

                diff = torch.mm(
                    mat_up.flatten(start_dim=1).float(),
                    mat_down.flatten(start_dim=1).float()
                )
                del mat_up, mat_down
                diff = diff * (alpha / rank)
                diffs.append(diff)
                indices.append(idx)
                per_lora_key_count[idx] += 1

            # Pairwise conflicts at this prefix
            for a in range(len(diffs)):
                for b in range(a + 1, len(diffs)):
                    i, j = indices[a], indices[b]
                    ov, conf, dot, na_sq, nb_sq = self._sample_conflict(diffs[a], diffs[b], device=compute_device)
                    pair_accum[(i, j)][0] += ov
                    pair_accum[(i, j)][1] += conf
                    pair_accum[(i, j)][2] += dot
                    pair_accum[(i, j)][3] += na_sq
                    pair_accum[(i, j)][4] += nb_sq

            del diffs  # free GPU memory

        # --- Compute per-LoRA avg conflict ratio ---
        per_lora_conflict = [0.0] * n_loras
        per_lora_pair_count = [0] * n_loras
        pairwise_results = []
        total_overlap = 0
        total_conflict = 0

        for i, j in pairs:
            ov, conf, dot, na_sq, nb_sq = pair_accum[(i, j)]
            ratio = conf / ov if ov > 0 else 0.0
            total_overlap += ov
            total_conflict += conf

            # Cosine similarity
            cos_sim = None
            if na_sq > 0 and nb_sq > 0:
                cos_sim = dot / (math.sqrt(na_sq) * math.sqrt(nb_sq))

            pairwise_results.append({
                "i": i, "j": j,
                "name_a": active_loras[i]["name"],
                "name_b": active_loras[j]["name"],
                "overlap": ov,
                "conflicts": conf,
                "ratio": ratio,
                "cosine_sim": cos_sim,
            })

            per_lora_conflict[i] += ratio
            per_lora_conflict[j] += ratio
            per_lora_pair_count[i] += 1
            per_lora_pair_count[j] += 1

        avg_conflict = total_conflict / total_overlap if total_overlap > 0 else 0.0

        per_lora_avg_conflict = []
        for i in range(n_loras):
            if per_lora_pair_count[i] > 0:
                per_lora_avg_conflict.append(per_lora_conflict[i] / per_lora_pair_count[i])
            else:
                per_lora_avg_conflict.append(0.0)

        # --- Resolve conflict modes ---
        resolved_modes = []
        for i in range(n_loras):
            cm = kwargs.get(f"conflict_mode_{i+1}", "auto")
            if cm != "auto":
                resolved_modes.append((cm, "manual"))
            else:
                # Auto-suggest based on avg pairwise conflict
                avg_c = per_lora_avg_conflict[i]
                if avg_c > 0.40:
                    resolved_modes.append(("high_conflict", f"auto — high avg conflict {avg_c:.1%}"))
                elif avg_c > 0.15:
                    resolved_modes.append(("low_conflict", f"auto — moderate avg conflict {avg_c:.1%}"))
                else:
                    resolved_modes.append(("all", f"auto — low avg conflict {avg_c:.1%}"))

        # --- Resolve merge strategy ---
        if merge_strategy == "auto":
            if avg_conflict > 0.25:
                resolved_strategy = "ties"
                strategy_reason = f"auto — avg conflict {avg_conflict:.1%} > 25%"
            else:
                resolved_strategy = "weighted_average"
                strategy_reason = f"auto — avg conflict {avg_conflict:.1%} <= 25%"
        else:
            resolved_strategy = merge_strategy
            strategy_reason = "manual"

        # --- Build enriched output stack ---
        # Output in the same format as input, with conflict_mode baked in
        enriched = []
        first = lora_stack[0]
        if isinstance(first, dict):
            for i, item in enumerate(active_loras):
                enriched_item = dict(item)
                enriched_item["conflict_mode"] = resolved_modes[i][0]
                enriched.append(enriched_item)
        else:
            for i, item in enumerate(active_loras):
                enriched.append((
                    item["name"],
                    item["strength"],
                    item["clip_strength"] if item["clip_strength"] is not None else item["strength"],
                    resolved_modes[i][0],
                ))

        # --- Build analysis report ---
        report = self._build_conflict_report(
            active_loras, pairwise_results, per_lora_avg_conflict,
            per_lora_key_count, resolved_modes, resolved_strategy, strategy_reason
        )

        return (enriched, report, resolved_strategy)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**
```
feat: implement LoRAConflictEditor analyze_and_enrich method
```

---

### Task 3: Implement _build_conflict_report

**Files:**
- Modify: `lora_optimizer.py` (LoRAConflictEditor class)

**Step 1: Add the report builder**

Add to LoRAConflictEditor class:

```python
    @staticmethod
    def _build_conflict_report(active_loras, pairwise_results, per_lora_avg_conflict,
                                per_lora_key_count, resolved_modes, resolved_strategy, strategy_reason):
        """Format conflict analysis as a readable report."""
        n = len(active_loras)
        lines = []
        lines.append("=" * 46)
        lines.append("LORA CONFLICT EDITOR - ANALYSIS REPORT")
        lines.append("=" * 46)

        # Stack overview
        lines.append("")
        lines.append(f"--- Stack ({n} LoRAs) ---")
        for i, item in enumerate(active_loras):
            lines.append(f"  {i+1}. {item['name']} (strength: {item['strength']})")

        # Pairwise conflicts
        if pairwise_results:
            lines.append("")
            lines.append("--- Pairwise Conflicts ---")
            for pc in pairwise_results:
                # Short names for readability
                name_a = os.path.splitext(os.path.basename(pc["name_a"]))[0]
                name_b = os.path.splitext(os.path.basename(pc["name_b"]))[0]
                lines.append(f"  {name_a} \u00d7 {name_b}:")
                lines.append(f"    Overlapping positions: {pc['overlap']:,}")
                lines.append(f"    Sign conflicts: {pc['conflicts']:,} ({pc['ratio']:.1%})")
                if pc["cosine_sim"] is not None:
                    lines.append(f"    Cosine similarity: {pc['cosine_sim']:.3f}")

        # Applied conflict modes
        lines.append("")
        lines.append("--- Applied Conflict Modes ---")
        for i, item in enumerate(active_loras):
            name = os.path.splitext(os.path.basename(item["name"]))[0]
            mode, reason = resolved_modes[i]
            lines.append(f"  {i+1}. {name}: {mode} ({reason})")

        # Merge strategy
        lines.append("")
        lines.append("--- Merge Strategy ---")
        lines.append(f"  Applied: {resolved_strategy} ({strategy_reason})")

        return "\n".join(lines)
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**
```
feat: add conflict report builder for LoRAConflictEditor
```

---

### Task 4: Add merge_strategy_override input to LoRAOptimizer

**Files:**
- Modify: `lora_optimizer.py:1422-1467` (LoRAOptimizer.INPUT_TYPES optional inputs)
- Modify: `lora_optimizer.py:2249` (optimize_merge signature)
- Modify: `lora_optimizer.py:2509-2510` (where _auto_select_params is called)
- Modify: `lora_optimizer.py:1497-1509` (IS_CHANGED and _compute_cache_key)

**Step 1: Add optional input**

In `INPUT_TYPES` optional dict (after `sparsification_density`, line ~1466):

```python
"merge_strategy_override": ("STRING", {
    "default": "",
    "forceInput": True,
    "tooltip": "Connect the merge_strategy output from a LoRA Conflict Editor to override the optimizer's auto-detected strategy."
}),
```

Note: `forceInput: True` hides the widget and only shows the input socket — the user can't type in it, only connect it.

**Step 2: Update optimize_merge signature**

Add `merge_strategy_override=""` parameter to the signature (line 2249).

**Step 3: Apply the override**

After `_auto_select_params` is called (around line 2509-2510), add:

```python
# Apply merge strategy override from Conflict Editor
if merge_strategy_override and merge_strategy_override in ("ties", "weighted_average", "weighted_sum"):
    mode = merge_strategy_override
    reasoning.append(f"Merge mode overridden to '{mode}' by Conflict Editor")
```

For per_prefix mode, also apply in the per-prefix decision (around line 2622-2625): when override is set, use it instead of per-prefix auto-selection.

**Step 4: Update cache key and IS_CHANGED**

Add `merge_strategy_override` to `_compute_cache_key` hash string and to `IS_CHANGED` parameter list.

**Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Expected: `OK`

**Step 6: Commit**
```
feat: add merge_strategy_override input to LoRAOptimizer
```

---

### Task 5: Add JS widget visibility for conflict editor

**Files:**
- Modify: `js/lora_stack_dynamic.js` (add extension for LoRAConflictEditor)

**Step 1: Add the extension**

Append to the JS file:

```javascript
app.registerExtension({
    name: "LoRAOptimizer.LoRAConflictEditor",
    nodeCreated(node) {
        if (node.comfyClass !== "LoRAConflictEditor") return;

        // After execution, hide conflict_mode slots beyond the actual LoRA count.
        // Since we don't know the count until execution, show all 10 initially.
        // The node's output report tells the user which slots are active.

        // No dynamic hiding needed for now — all 10 slots are always visible.
        // Unused slots (beyond the stack size) are simply ignored.
        // Future: use PromptServer messages to hide extras after first run.
    },
});
```

**Step 2: Verify syntax**

Run: `node --check js/lora_stack_dynamic.js`
Expected: no output (success)

**Step 3: Commit**
```
feat: add JS extension stub for LoRAConflictEditor
```

---

### Task 6: Integration test and final verification

**Files:**
- Verify: `lora_optimizer.py` (full syntax check)
- Verify: `js/lora_stack_dynamic.js` (syntax check)

**Step 1: Full syntax verification**

Run: `python -c "import ast; ast.parse(open('lora_optimizer.py').read()); print('OK')"`
Run: `node --check js/lora_stack_dynamic.js`

**Step 2: Verify NODE_CLASS_MAPPINGS includes all nodes**

Run: `grep -A 10 "NODE_CLASS_MAPPINGS" lora_optimizer.py`

Expected: Should include `LoRAConflictEditor`.

**Step 3: Verify data flow**

Check that:
- LoRAConflictEditor accepts LORA_STACK and outputs LORA_STACK (pass-through with enrichment)
- Output LORA_STACK contains `conflict_mode` in each entry
- `merge_strategy` STRING output can connect to optimizer's `merge_strategy_override` STRING input
- Optimizer's `forceInput: True` means the widget is hidden unless connected

**Step 4: Commit**
```
chore: verify LoRAConflictEditor integration
```

---

## Verification Checklist

- [ ] LoRAConflictEditor appears in ComfyUI node list under "loaders/lora"
- [ ] Node accepts LORA_STACK from any stacker (dict or tuple format)
- [ ] Analysis report shows pairwise conflicts and applied modes
- [ ] `auto` conflict modes produce sensible suggestions
- [ ] Manual conflict modes (`all`/`low_conflict`/`high_conflict`) pass through unchanged and don't get overwritten on re-run
- [ ] `merge_strategy` output connects to optimizer's `merge_strategy_override` input
- [ ] Optimizer respects the override when connected
- [ ] Node works with 1 LoRA (no pairwise analysis, all defaults)
- [ ] Old workflows without the editor still work (backward compatible)
