# Maintainer Summary: upstream sync + merge correctness pass

## Short version

This PR does two jobs at once:

1. it keeps the newer upstream behavior that landed through `dfd3920`, and
2. it carries forward the local merge-correctness refactor that was built from a long external review of the optimizer.

The practical goal is simple: keep the current node/workflow surface intact, but make the merge logic operate on the right unit of work and stop losing accuracy in places where the code was using avoidable heuristics.

---

## The original brief, reduced to the actionable parts

The review we worked from made a few concrete points:

- the old overwrite bug was not the main remaining problem; current upstream already fixed late overwrite-by-collision,
- the deeper issue was that analysis was still organized by raw LoRA prefix instead of the resolved target model weight,
- linear LoRA merges were being expanded to dense diffs and recompressed, even when they can stay exact in low-rank form,
- auto-strength was described as exact but was using proxy norms rather than exact streamed energy,
- some small implementation bugs were real:
  - `LoRAConflictEditor` dropped `key_filter`,
  - some key-normalization claims were stronger than the code,
  - saved alias selection was not deterministic,
  - analysis and final merge could disagree about what actually overlapped.

The review also pushed on documentation quality:

- the README sometimes overstated what the code was actually doing,
- paper references were sometimes closer to “inspired by” than faithful implementations,
- AutoTuner should be described as proxy-ranked unless a real evaluator is connected.

That was the base brief. Then upstream moved and added more behavior that also had to be preserved.

---

## What upstream changed while this work was in flight

Upstream added several behavior changes after the earlier local branch point:

- the AutoTuner ↔ Optimizer bridge (`tuner_data`, `settings_source`, `output_mode`),
- full-rank aware merge safeguards,
- `LoRACompatibilityAnalyzer`,
- Z-Image fused-QKV save fixes,
- folder-aware `SaveTunerData`,
- newer workflow / UI expectations around those nodes.

So the correct job was not “replace upstream with the refactor.”
It was “merge the refactor onto current upstream without regressing either side.”

---

## What actually changed in code

### 1. Analysis and merge now use resolved target groups

This is the main correctness change.

Before:

- Pass 1 analyzed raw prefixes,
- Pass 2 merged raw prefixes,
- only at collection time were collisions accumulated if two aliases resolved to the same target weight.

That prevented outright overwrite loss, but it still meant analysis and AutoTuner could misread real overlap.

Now:

- aliases are grouped by resolved `(is_clip, target_key)` before analysis,
- per-LoRA contributions inside a group are aggregated before conflict/compatibility logic runs,
- Pass 1 and Pass 2 operate on the same real merge unit,
- late collision accumulation remains only as a safety guard.

This is the change that addresses the “mixed trainer aliases still blind Pass 1” problem.

Relevant area:

- `/Users/sarav/Downloads/play/ComfyUI-LoRA-Optimizer/lora_optimizer.py`

### 2. Linear merge paths stay exact when they can

For linear merge modes (`weighted_sum`, `weighted_average`, `normalize`), the code now tries to keep the result in exact low-rank form by concatenating factors rather than:

- expanding to dense diff,
- merging dense tensors,
- recompressing with SVD.

The dense path still exists as a fallback when exact low-rank composition is not valid for a given patch shape or parameterization.

This is both cleaner mathematically and cheaper operationally.

### 3. Auto-strength uses exact streamed branch energy

The earlier path used norm proxies that were not exact for the whole merged object.

Now the optimizer accumulates streamed branch-level quantities and computes model and CLIP scaling from:

- per-LoRA Frobenius norm squares,
- pairwise dot products.

This keeps the logic streaming-friendly while making the math consistent with the reported behavior.

### 4. Conflict metrics are less naive

The refactor already moved conflict handling beyond raw sign counts by adding:

- weighted conflict,
- expected conflict baseline,
- excess conflict,
- subspace overlap,
- optional activation-aware importance,
- optional decision smoothing.

Those changes remain in place on top of upstream.

This does not turn the optimizer into a paper-faithful research implementation, but it does remove some avoidable noise from the older decision logic.

### 5. Save behavior is more deterministic and more defensive

`SaveMergedLoRA` keeps the earlier local fixes:

- canonical prefix selection for alias-collapsed targets,
- adaptive rank estimation for diff compression,
- exact-low-rank-aware save handling,
- deterministic mapping when multiple aliases resolve to one target.

At the same time it keeps newer upstream save-side behavior:

- Z-Image fused-QKV handling,
- dtype preservation during re-fusion,
- current upstream save diagnostics and output layout.

### 6. Small real bugs are fixed

The branch also keeps/fixes the smaller concrete defects from the review:

- `LoRAConflictEditor` preserves `key_filter`,
- canonical alias selection is deterministic,
- bridge return contracts remain consistent with current upstream expectations,
- the compatibility analyzer node is present and registered,
- `SaveTunerData` matches the newer folder-aware upstream flow.

---

## What was intentionally preserved from upstream

This PR does **not** roll back the newer upstream UX/runtime work.

It keeps:

- `LoRAOptimizer` `tuner_data` / `settings_source`,
- `LoRAAutoTuner` `output_mode`,
- the bridge JS behavior,
- full-rank gating logic,
- `LoRACompatibilityAnalyzer`,
- current save-folder behavior for tuner data,
- current workflow compatibility.

That was a deliberate constraint the whole time.

---

## What was intentionally *not* claimed

The original review was right to call out overclaiming, so the docs were adjusted accordingly.

This branch does **not** claim that:

- every cited paper is implemented faithfully,
- AutoTuner finds the objective best merge in a strong sense,
- all merge decisions are mathematically optimal,
- memory is constant regardless of LoRA count or active layer size.

The code is still a practical optimizer with heuristics. The difference is that some of the biggest avoidable inaccuracies are removed.

---

## Why these choices were made

The rule used for this branch was:

- fix correctness first where the merge unit was wrong,
- keep exact math where the representation allows it,
- preserve current upstream UX contracts,
- do not make the PR bigger than necessary by trying to replace the entire optimizer philosophy.

That is why the branch focuses on:

- target-group correctness,
- exact linear merge handling,
- exact streamed auto-strength,
- deterministic saving,
- preserving upstream node and workflow behavior.

---

## Validation that was actually run

The branch was validated with:

- `python3 -m py_compile lora_optimizer.py tests/test_lora_optimizer.py`
- `.venv-tests/bin/python -m py_compile lora_optimizer.py tests/test_lora_optimizer.py`
- `.venv-tests/bin/python -m unittest discover -s tests -v`
- `git diff --check`

Current result:

- full test suite passed: `19 passed`

The test coverage added here is targeted, not decorative. It covers:

- target-group alias behavior,
- exact low-rank linear merges,
- exact auto-strength math,
- excess-conflict / subspace metrics,
- bridge/workflow compatibility,
- deterministic save prefixes,
- save-path safety,
- tuner-data output exposure,
- compatibility-analyzer node registration.

---

## How to review this PR efficiently

Recommended order:

1. `lora_optimizer.py`
   - target-group build and Pass 1 / Pass 2 flow
   - exact linear merge path
   - auto-strength branch-energy logic
   - full-rank gates
   - saver behavior
   - bridge return contract
2. `tests/test_lora_optimizer.py`
3. `js/lora_optimizer_bridge.js`
4. `README.md` and `docs/wiki/Nodes.md`

If a maintainer wants only the highest-value behavior changes, the key question is:

> Does the optimizer now analyze and merge the same real target weight when multiple aliases map to that weight?

If that answer is yes, most of the important correctness work in this branch is already in the right place.
