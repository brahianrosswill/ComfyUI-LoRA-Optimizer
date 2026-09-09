# H3 1.8.5 release scope

This patch release includes the correctness and native-export fixes in
`cb92bd4f6e297fc16fe41889851801dcff792767`, plus the offline benchmark,
identity-review, waveform and evaluator-calibration tools and their research
records. It does not install or restart a running ComfyUI instance.

## What changes for the autotuner

- Conflict-aware sparsification really skips when its guard rejects it, and
  scoring discounts sparsity only when sparsification was actually applied.
- External evaluators receive complete candidate merges, reject non-finite
  scores, and fail explicitly in external-only mode instead of silently
  substituting internal weight statistics.
- H3 mixed diffusion/CLIP detection, experimental compression precision and
  eligible native-factor NP/CT exports are corrected. Cache revisions ensure
  the affected cached results are recomputed.

The scoring formulas' weights and default selection policy are **unchanged**.
Correcting scoring inputs can change a selected candidate; that is not a new
perceptual-ranking formula. NP-LoRA and CT-Merging remain opt-in. The measured
export-size savings do not establish a universal speed or visual-quality win.

## Research boundary

The 144-clip character study found seed-dependent identity, wardrobe and action
trade-offs, not a consistent NP/CT winner. One person's ratings are individual
preferences, not population ground truth. The local Qwen, Synchformer and FATE
probes did not qualify an automatic AV quality ranker; remaining AV ratings
are still missing. None of these evaluators is imported or enabled by nodes.

The rejected F08 candidate-admission/deduplication implementation and its
prototype-only test are deliberately excluded from this release. Its timing
records and generic comparison/audit tools are retained: the measured prototype
was slower with unchanged selected outputs. Running the benchmark against this
release creates a new comparison, not a reproduction of that excluded source.

The only additional production-code hook is `_new_patch_store`: ordinary nodes
still get fresh dictionaries. An explicit offline subclass can use bounded,
file-backed storage for the research TIES export path, with model application
and QKV refusion disabled. This is not a new automatically enabled node feature.

Historical reports and hashes describe the exact sources used at each checkpoint,
including the then-dirty prototype. They have not been rewritten to claim those
runs tested this release tree. Model weights, generated media, environments and
private artifact directories are not included; some replay/audit commands require
those pinned local artifacts and the separately documented environments.

See the [merge study](2026-09-08-h3-autotuner-render-study.md),
[local AV evaluator report](2026-09-09-h3-local-av-evaluator.md),
[Synchformer report](2026-09-09-h3-synchformer.md), and
[FATE calibration](2026-09-09-h3-fate-calibration.md) for evidence and limitations.

## Release-tree validation

The staged tree was exported to a fresh temporary directory and tested without
the local F08 implementation or its prototype-only test. Production optimizer
SHA-256: `d1a9f9657580229c30aa4a7e43d700a2308c800f4afe0ebee81a199ebae00360`.

- Python 3.13, user `13_env_py313`, CUDA hidden: **1,052 passed, 3 CUDA-only
  skipped, 69 subtests passed**. The existing pynvml deprecation warning remains.
- JavaScript workflow migrations: **3 passed**.
- Real ComfyUI CPU loader/export round trips: **passed**, including signed
  model/CLIP contributions, partial H3 QKV, dense/native NP/CT and metadata.
- Offline mapped TIES versus ordinary export: **passed** with identical tensor
  payloads and real ComfyUI application for FP32 and BF16 fixtures.
- Staged whitespace check passed; no model weights or generated media staged.

These are fresh release-tree correctness checks, not additional render-quality
evidence. This validation did not modify the live ComfyUI installation.

## Commit identity and historical records

GitHub rejected the first push under its private-email protection. Both
unpublished commits were recreated using the user's verified GitHub no-reply
identity; the original commits remain on a local backup branch, not published.
The historical fix commit `27bedd7621e8e53b5d9c4d87ba594c2b7aa76008` and
published `cb92bd4f6e297fc16fe41889851801dcff792767` have identical Git trees.
Frozen research records retain their historical commit IDs and source hashes.
The search benchmark's baseline reference now uses the published equivalent,
with its optimizer content-hash guard unchanged, so a clean checkout can resolve
the baseline. This one-line reference update does not change measured outcomes;
historical plan replay still requires the exact historical, hash-pinned runner.
