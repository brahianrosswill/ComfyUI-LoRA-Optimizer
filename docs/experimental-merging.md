# Experimental NP-LoRA and CT-Merging

Status: opt-in implementation; small CPU tensors and stock-loader round trips are tested. This is **not evidence of better H3 video/audio generation**. Existing stable mode dropdowns and disconnected behavior remain unchanged. No new dependencies, sampling, training, checkpoint downloads, or automatic release.

## Wiring

Connect **LoRA Experimental Options → experimental_options** on either:

- **LoRA AutoTuner**, or
- **LoRA AutoTuner Settings → settings** on **LoRA Optimizer** / **LoRA Optimizer (Inline Chain)**.

Enable NP-LoRA, CT-Merging, or both. Both enabled means separate competing candidates, not sequential transformations. Disconnecting the options node, disabling its master switch, or disabling both methods restores the stable path. New optional sockets are appended, preserving existing socket order.

Use **Merge Selector** with the returned TUNER_DATA to try any ranked candidate. Saved tuner data retains exact method settings; it can also be applied through the existing Simple/Legacy tuner-data bridges. A bare experimental mode name in a manual override is rejected: replay requires its versioned parameters.

## Controls

| Control | Default | Meaning |
| --- | --- | --- |
| `enabled` | true | Master switch; takes effect only when connected. |
| `np_lora`, `ct_merge` | true | Independent candidate switches. |
| `np_subject_slot`, `np_style_slot` | 1, 2 | Positions among **active, non-preserved adapters**, in stack order. NP requires exactly two. Disabled/zero-strength entries and preserve overlays are not counted. For Inline, this is the resulting captured merge stack, not necessarily the original loader numbering; check the report's source names. |
| `np_strength` | 0.5 | Soft-projection parameter μ; zero is additive. Larger values suppress more subject contribution in the style subspace. |
| `np_rank` | 0 | Maximum style-subspace rank; zero uses all numerically supported directions. |
| `np_energy` | 1.0 | Energy fraction retained **within the rank-capped** style subspace. |
| `ct_common_rank` | 4 | Maximum consensus rank per layer, bounded by numerical support. |
| `ct_residual_rank` | 16 | Maximum retained SVD directions per adapter/layer. |
| `ct_scale` | 1.0 | Scale of shared CT matrix merges; not of unique targets, vectors, or preserve overlays. |
| `trials_per_method` | 1 | 1–3 additional trials per eligible method. Trials use the requested μ/scale, half, then double, within widget bounds; duplicate settings are removed. |

## Trial cost and ranking

The original `top_n` stable-trial budget is retained. Experiments add up to `enabled eligible methods × trials_per_method` trials, **plus one additive baseline**. Default options therefore add three merges: additive, NP-LoRA, and CT-Merging. With stable `top_n=10` and three trials per method, at most 17 configurations are measured. Selection controls accept all 17. Final winner application can require one more merge, as with ordinary multi-candidate tuning.

The **unreleased search-admission prototype** additionally ignores density/sign-voting placeholders for plain weighted sum and weighted average when experiments are enabled, because those fields do not affect the merge. Their shared-group cache keys use the same rule. Auto-strength on/off candidates are also deduplicated only when the actual adjustment is **exactly 1 on both model and CLIP branches**; external evaluators retain separate strength trials. Real TIES voting/density, sparsification, refinement and nontrivial strength changes remain distinct. This can admit a different next candidate or winner; it does not guarantee faster searches, better generation quality, or unique output tensors for every remaining configuration. The completed forty-trial H3 comparison has **11–21% higher pair-median search times with unchanged selected outputs**; the recommendation is to withhold this prototype as-is, not ship it as a speed improvement. Timing variation and exact scope are preserved in the [research checkpoints](research/2026-09-08-h3-autotuner-render-study.md). An experimental-only search version invalidates in-node and persistent tuning results without changing stable cache identities. Disconnected or disabled experimental options retain the original search. The installed node pack has not been replaced by this prototype.

Experimental runs currently score **all target groups for every candidate**, bypassing the `scoring_speed` subsampling setting. This checks compatibility before ranking, including layers that a fast subset could miss. Expect extra time on large models. Disconnected runs retain existing speed behavior.

The legacy heuristic does not select or rank these extra trials. They enter with a neutral placeholder heuristic score and receive the same measured scoring / external evaluator as stable candidates. Reports label experiments and skipped candidates. Weight-space scores measure mathematical properties, not prompt fidelity, motion, sound, or synchronization.

## Math and numerical policy

The implementations are independently written from the papers, not copied from third-party node packs.

**NP-LoRA** implements soft subject/style projection (Equation 12): `style + content - μ/(1+μ) × content × V × V.T`, where V spans selected right singular directions of the style update. It is asymmetric; exchanging roles changes the result. Rank/energy caps are explicit controls beyond the paper's full-adapter-subspace default. [NP-LoRA, revision 3](https://arxiv.org/html/2511.11051v3)

**CT-Merging** follows Algorithm 1's projector-based common subspace, paired common response, projected residuals, polar alignment, and per-task RMS coefficients. Common directions are computed from concatenated thin bases, equivalent to diagonalizing their mean projector without allocating a full square projector. [CT-Merging](https://arxiv.org/html/2607.20561v1)

This implementation adds explicit safeguards for heterogeneous/degenerate adapters: retain only numerically supported directions; allow different retained ranks per task; use the partial polar factor without arbitrary nullspace completion; omit fully projected-out residual columns on both sides; and reduce a common-rank cutoff that would split a tied consensus eigenspace. These safeguards define this experimental variant and are tested separately, not claimed as reproduced paper benchmark results.

Signed strengths and alpha/rank scaling are applied before decomposition. Eligible plain low-rank shared targets now produce **native FP32 factors**, without constructing the dense update or recompressing it in Pass 2. NP applies the projection to the content down-factor; CT evaluates its common response through the input factors and emits the polar recomposition directly. For a fully supported style rank with no energy truncation, NP uses a canonical QR basis depending only on the style down-factor. This preserves exact factor sharing across H3 Q/K/V slices; rank-deficient/truncated cases retain the SVD-selected subspace.

Native output is used only when all contributors have eligible 2D factors and the result is smaller than a dense patch. An explicit aggressive rank limit is still respected. Dense, spatial, cleaned, conflict-masked, preserved-overlay, and unsupported virtual/sliced cases keep their existing fallback; unsupported shared spatial experiments are still rejected. Analysis still materializes per-group deltas, so this does **not** eliminate all dense-diff memory use. CPU experimental group merging is serial. Full-adapter H3 measurements are recorded in the [render study](research/2026-09-08-h3-autotuner-render-study.md); audiovisual quality remains unvalidated.

Native experimental factors retain FP32, including when patch compression is disabled. When a dense fallback is compressed, its merged delta and compressed factors also retain FP32. Rounding to BF16 before SVD can overwhelm small projection corrections. Arbitrary rank truncation remains lossy, and uncompressed **dense fallback** patches keep their existing storage policy. Stable modes retain their existing precision behavior.

## Payload safety and limitations

- Shared 2D linear/dense updates are experimental merge targets. Bias/norm vectors, unique targets, and explicit `preserve` overlays stay additive. These side-payload rules also apply to the CLIP branch. NP roles remain associated with their original active indices when other entries are preserved or absent from a target.
- Shared spatial convolutions are unsupported by the new modes: that candidate is skipped with a reason, not partially applied. Stable candidates continue. Invalid settings fail clearly rather than being silently ignored.
- Use `preserve` for compatible Turbo/distillation adapters, or apply them separately. They are not automatically identifiable from arbitrary filenames; do not assign them subject/style roles. Existing full/pruned basis, H3 partition, ambiguity, DoRA, and PDD rejection checks remain in force.
- Experimental hierarchical merge formulas are rejected for now; role assignment to intermediate virtual adapters needs a separate design.
- Parameters and algorithm version are included in ranking and patch cache identities. Experimental runs do not use/upload community rankings; `auto_ignore_strength` uses strength-sensitive local memory while experiments are enabled. Analysis can still reuse valid context-aware statistics.
- If `record_dataset` is enabled, experiments write `autotuner_experimental_dataset.jsonl`, not the stable research dataset. Legacy bridges do not copy experimental parameters into manual widgets that cannot represent them.
- Experimental merges bypass lossy diff caches and the stable cross-candidate patch caches. This keeps factor/dense inputs consistent with final replay. Disk/RAM diff caches can still help stable trials in the same run.
- Save Merged LoRA embeds the mode, versioned experimental parameters, and participant names in metadata. Normal loader-compatible patches are exported; no runtime routing hooks are required. A separately requested lossy export rank remains lossy.

## Verification

Run `python -m pytest -q`, `node tests/js/dynamic_migration.test.cjs`, and the separate `python tests/integration/comfy_roundtrip.py /path/to/ComfyUI` check. The experimental tests include independent matrix references, additive limits, signed/mixed-rank updates, role reversal, degenerate/tied subspaces, factor/dense agreement, preserved overlays, disabled-path equivalence, bounded trials, failure reports, and fresh/in-memory/persistent/Selector/bridge replay. Real ComfyUI round trips cover H3 partial QKV, AdaLN/bias/norm, signed CLIP contributions, and metadata for both modes.

Before making H3 quality claims, compare fixed-seed FL2VA/Ref2VA renders and audio against additive and stable winners, on matching bases. The existing full-to-pruned affine conversion and PDD runtime work remain separate, unimplemented projects.
