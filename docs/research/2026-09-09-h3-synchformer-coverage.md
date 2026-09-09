# H3 evaluator coverage: the flagged contacts are not cropped out

The preceding [Synchformer calibration](2026-09-09-h3-synchformer.md) was substantive progress: the model detected imposed delay, but did not demonstrate useful ranking of the twelve original clips. This follow-up investigates that limitation without rerunning or tuning the evaluator.

The [prospective diagnostic policy](data/2026-09-09-h3-synchformer-coverage-policy.json) selected exactly the three sources and five contiguous ranges anchored by the earlier [frame/waveform observations](data/2026-09-08-h3-av2-sync-seed03.json): base A09 frames 6–12, and Repair additive A11 / Repair CT A07 frames 12–18 and 32–38. Those earlier observations were already unblinded and informed by the waveform. This is not a new independent reviewer, listening judgment or unbiased method comparison.

## What the actual inputs retain

The original media, prior diagnostics and selected model outputs were rehashed. Reconstructed RGB/time mapping matches the evaluated input records. Ten comparison panels show every selected frame with the exact crop boundary and the untouched crop beside it. All **35 selected frames** were inspected.

The model's overlapping segments represent 120 target time slots but only **115 unique source frames**, numbered 2–116. Frames 0–1 and 117–123 are excluded. The spatial crop retains **46.01%** of the resized image area. These figures are coverage facts, not quality scores.

| Selected sequence | Relevant content in the crop |
| --- | --- |
| A09, frames 6–12 | Approach, glove reaching the bag and following contact remain visible |
| A11, frames 12–18 | Additional glove extension and contact region remain visible |
| A11, frames 32–38 | Later reach, contact and follow-through remain visible |
| A07, frames 12–18 | Guard/no additional extension remains visible; its difference from A11 is retained |
| A07, frames 32–38 | Later reach and contact region remain visible |

No selected frame is temporally excluded. Peripheral feet, floor and background are partly cropped, but the relevant hand/bag region is retained throughout these ranges. **Crop/time-window exclusion is not supported as the explanation for these particular flagged events.** This does not prove that the model attends to the contacts or resolves their details, and it says nothing about uninspected events.

## Why a gross-offset score remains insufficient

The pinned [official class-grid and quantizer](https://github.com/v-iashin/Synchformer/blob/b66668a1521d7567cc760e5544b2b5b53179b687/dataset/transforms.py) were exercised on synthetic scalar offsets, without loading model weights. The 200 ms grid maps ±80 ms to the zero class. It maps ±125 ms to ±200 ms; both of those classes are also included in the previously frozen three-class near-zero mass.

Thus, that aggregate cannot distinguish these scalar offsets by class membership alone. This is quantization arithmetic, not a measured perceptual tolerance, proof of learned-model accuracy or permission to retune the diagnostic's thresholds. The previous energy/contact observations also suggest different local relationships within one clip; their energy maxima are **not identified sound onsets** and cannot justify a single corrective stream shift.

Keep the natural-merge ranking rejection. The evidence does not justify changing Synchformer crops or adopting confidence as an autotuner reward.

## Newer-method preflight

[FATE, August 2026](https://arxiv.org/html/2608.01310v1), retains temporally aligned embeddings rather than only an offset prediction. Its reported average per-clip human correlation is **0.1724**, not 17.24 as an unscaled correlation, and its tested generation families do not include H3. Its two-second training windows and 0.5-second stride do not establish sub-100-ms sensitivity. It is a candidate to calibrate, not a validated replacement.

The [official implementation](https://github.com/guankaisi/FATE) and public model metadata were inspected, with exact revisions and file hashes saved in the [preflight record](data/2026-09-09-h3-fate-preflight.json). Base plus adapter safetensors total approximately 3.40 GB. No FATE weights or packages were downloaded or installed, and no model inference was run.

The source audit found three relevant integration risks:

- Failed video decoding can become black frames, and empty audio can become silence. An evaluation harness must preserve these as failures.
- Failed duration probing can become an assumed ten-second clip. Actual source clocks must remain authoritative.
- A padded-length alignment branch resizes the feature dimension instead of time. Exercising only that reviewed method on a synthetic `[1,6,4]` video tensor produced `[1,6,3]`, where `[1,3,4]` was required. It is not established to occur for an unpadded two-second input; no upstream source was edited.

Next is a separately frozen FATE setup/control protocol with strict weight/adapter loading, verified feature clocks and explicit failures. Any later admission still needs natural-clip evidence; it cannot follow merely from successful installation or artificial controls.

## Checkpoint

The [completed observation record](data/2026-09-09-h3-synchformer-coverage-observations.json) pins every panel, reused diagnostic and synthetic quantizer result. Combined evaluator/coverage tests: **105 passed in 0.24 seconds**. No full merger-suite rerun, GPU work, ComfyUI contact, new human labels, held-out evaluation, production change, commit, push or version bump occurred. The overall goal remains active and the remaining audiovisual gate remains open.
