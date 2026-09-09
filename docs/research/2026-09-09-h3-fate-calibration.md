# H3 FATE calibration: insufficient agreement for autotuner ranking

The [audited control stage](2026-09-09-h3-fate.md) qualified a bounded natural-clip calibration, not an automatic quality judge. This follow-up completes the [prospectively frozen original-only sweep](data/2026-09-09-h3-fate-calibration-policy.json) and keeps its negative findings. It does not retune the model or add a production reward.

## What was evaluated

All **12 unique already-rated seed03 originals** completed, in A01–A12 order: 36 two-second forward passes with the unchanged strict-loaded FATE-derived model, adapter, processor, audio resampling, temporal alignment, precision and three-window mean. Both previously evaluated originals were included as repeatability anchors. No held-out clips, new render jobs, augmented variants, new weights, window searches or alternative scoring formulas were used.

The model received only video/audio tensors. R1's fourteen contextual rating entries were joined afterward by the original media SHA-256. One person's ratings remain one person's perception, not consensus. Shared controls remain separate entries within their original pairings; their labels were not averaged or reconciled.

## Agreement with R1 synchronization preferences

| Pairing | Concordant | Discordant | Human-tied | Non-tied agreement |
| --- | ---: | ---: | ---: | ---: |
| Combat + Cinema | 8 | 6 | 7 | 8/14 = 57.1% |
| Combat + Repair | 9 | 7 | 5 | 9/16 = 56.25% |

Each pairing contains all 21 unordered comparisons of its seven entries. There were no model ties or missing values. These fractions are descriptive for this small, correlated, single-reviewer calibration; they are **not statistical significance, population accuracy or held-out performance**. The two pairings are not pooled as independent samples.

Raw model scores are cosine averages, not 0–4 ratings:

| Arm | Cinema model score | R1 sync | Repair model score | R1 sync |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.169194 | 2 | 0.169194 | 1 |
| Combat only | 0.160499 | 4 | 0.160499 | 4 |
| Additive | 0.201844 | 3 | 0.173192 | 1 |
| NP | 0.209897 | 3 | 0.188891 | 4 |
| CT | 0.248579 | 3 | 0.179054 | 4 |
| Tuner winner | 0.190967 | 2 | 0.184301 | 3 |
| Second candidate | 0.207037 | 3 | 0.192921 | 3 |

The strongest contrary example persists: **Combat-only is last by the model in both groups despite R1 synchronization 4 in both**. Cinema CT is first by the model but R1 rates its synchronization 3; Repair's second candidate leads the model while several lower-ranked arms have higher R1 synchronization ratings. This does not justify an NP/CT preference bonus or establish that any arm is perceptually best.

The result is consistent with the earlier caution that gross-offset/tonal sensitivity is not sufficient for ranking naturally generated contact/sound quality. We do not infer why the model misorders clips solely from these scores, invert its ranking, or select a better-looking window after seeing the labels.

**Disposition: do not use this FATE-derived score as the current autotuner reward.** Preserve it as an audited research diagnostic. The frozen one-sweep stopping rule applies; this result is not a reason to search prompts, seeds, windows or thresholds until agreement looks better. No automatic held-out admission follows.

## Independent verification

The separate CPU auditor imports neither the inference runner nor its comparison function. It rechecks actual source streams and frame PTS; independently reconstructs decoding, downmix/resampling, window boundaries, resize/normalization and masks; and matches **all 108 input-tensor records** exactly. All 36 window scores reproduce from saved normalized embeddings using a separate double-precision contraction. Every one of the fourteen label mappings and 42 pairwise comparisons reproduces independently.

All six original A05/A09 anchor windows exactly match the earlier control run's input records, normalized-embedding hashes, diagonal sequences and scores. There is no model/input drift explaining the negative ranking result.

Focused evaluator tests: **159 passed**, including seventeen new scope/label-join tests. Peak model allocation remained **2,762,175,488 bytes** (about 2.57 GiB). Inference and auditing exited successfully. Artifacts and implementation identities are pinned in the [calibration checkpoint](data/2026-09-09-h3-fate-calibration-checkpoint.json).

## Direct node verification added

With the evaluator finished and the existing generation queue still empty, the three formerly skipped GPU regressions ran in the user's `/media/p5/miniforge3/envs/13_env_py313` environment on the RTX 5090 (PyTorch 2.11.0+cu130):

- Virtual LoRA expansion runs on the GPU and matches the CPU expansion.
- Z-Image QKV deferral/refusion registers the fused tensor's scoring statistics correctly and returns CPU patches.
- The synthetic end-to-end QKV tuner sweep preserves GPU-versus-CPU rankings and final scores within the test's declared tolerance.

**3 passed, 550 deselected, no skips**; the [receipt](data/2026-09-09-h3-cuda-regression-followup.json) preserves the exact command, output and source hashes. These are the three specific missing CUDA checks, not a new full-model H3 quality benchmark or a reason to approve F08.

## Goal and shipping status

The existing [requirement audit](data/2026-09-08-h3-goal-audit.json) remains the scope reference. R7's three previously skipped tests now have direct CUDA evidence. R8 gains the audited evaluator calibration and preserved negative result. **R9 remains incomplete**: the other AV2 calibration seed and two held-out seeds still need real audiovisual judgments; these model scores do not replace them. Scoped checks found only the original downloaded R1 rating file, and the three existing review pages are present with recorded hashes.

Keep the prior shipping recommendation: retain committed F01–F07 correctness/native-export changes; leave NP/CT opt-in and default ranking unchanged; withhold the slower uncommitted F08 admission/backfill prototype. Do not release the whole dirty worktree. This turn makes no production changes, installs no node pack, starts/stops/unloads no ComfyUI session, and does not commit, push or bump version. HEAD remains `27bedd7`, version 1.8.4.

The remaining AV2 pages are ready when the reviewer is available: [seed04](../../.h3-study-artifacts/20260908/av2-review-seed04/review.html), [seed11](../../.h3-study-artifacts/20260908/av2-review-seed11/review.html), [seed12](../../.h3-study-artifacts/20260908/av2-review-seed12/review.html). They contain 42 contextual entries for 36 unique clips. No new ratings or broader AV winner is claimed. The overall goal remains active; lack of adequate AV evidence is not redefined as completion.
