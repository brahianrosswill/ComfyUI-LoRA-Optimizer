# H3 Synchformer calibration: detects imposed delay, not admitted for merge ranking

This follows the [Qwen2.5 joint-evaluator rejection](2026-09-09-h3-local-av-evaluator.md). The user authorized isolated local evaluator setup and testing on previously rated clips. The previous goal turn made progress: its failed delay control changed the next method, rather than supplying usable automatic scores. The overall H3 goal remains incomplete.

## Method and isolation

[Synchformer](https://github.com/v-iashin/Synchformer) is a dedicated audio/video offset estimator. [AVGen-Bench](https://github.com/microsoft/AVGen-Bench) uses it separately from audio and visual quality measures; neither source establishes validity for these H3 merges. The official AudioSet checkpoint is `24-01-04T16-39-21`; source revision is `b66668a1521d7567cc760e5544b2b5b53179b687`.

The separate `.h3-study-artifacts/20260909/synchformer/venv` installed 71 resolved packages from [pinned direct dependencies](h3-synchformer-requirements.txt), reusing cached CUDA wheels. It does not alter ComfyUI's conda environment or the Qwen evaluator. The source/configuration download and 1,131,153,989-byte checkpoint download completed. The checkpoint matches the publisher's MD5 `54037d261253aad8bd8ba82cdd7e71a7`; recorded SHA256 is `5b4b3557fbd96b61aaffa8bc70b28f9ff53f8fa98edc202655c5d94ab3c719ee`. All 513 model tensors, containing 237,460,245 elements, were finite. All 67 tracked source/configuration files are pinned in the setup receipt.

An initial bare `weights_only=True` load correctly refused the checkpoint's OmegaConf training metadata. Static checkpoint inspection enumerated its required globals. The pinned OmegaConf container restoration methods were inspected; the subsequent load allowed only those configuration containers, primitive containers and numeric scalar/dtype types. It still used `weights_only=True`, never unrestricted pickle. No checkpoint configuration interpolation, training objects or optimizer state were used to construct the model. The downloaded YAML supplies the reviewed architecture; both feature-extractor checkpoint paths are set to null because their weights are already present. All model keys/shapes must match exactly before strict loading. No positional truncation, training entrypoint, telemetry, model-side downloads, external media upload or credentials.

## Frozen preprocessing and controls

The [prospective control policy](data/2026-09-09-h3-synchformer-policy.json) was saved before inference. Its implementation uses the official deterministic test transforms, FP32 parameters and CUDA FP16 autocast, batch one and four CPU threads. The existing ComfyUI queue must be empty before model loading and each case; loading requires 20 GiB free GPU memory. These checks are not a GPU reservation, but no unrelated process was stopped or unloaded.

The [official example](https://github.com/v-iashin/Synchformer/blob/main/example.py) re-encodes media, and its repository warns that PyAV versions change results. This study instead uses FFmpeg raw RGB decode with actual frame PTS, explicit nearest-frame mapping from 24 to 25 fps and a bicubic 426×256 resize. No source media are rewritten. Maximum measured frame-time error is about 20 ms. This is an audited alternative input route, not reproduction of the publisher's example probabilities.

The fixed 224-square center crop begins at resized x=101, y=16. After the first-five-second crop, the official 14 overlapping 16-frame segments cover target frames 2–121, visual sample times 0.08–4.84 seconds and audio [0.08,4.88). Original frames near the beginning/end and peripheral image content are excluded. This is not a full-frame or whole-clip judgment. Audio uses the existing audited native float decode, arithmetic stereo mean and 16 kHz `librosa/soxr_hq` resampling, without gain normalization; stereo fidelity is not assessed.

Controls retain original, exact-zero and +750 ms delayed waveforms. Delay is zero-filled and tail-trimmed, never wrapped; transform offset is zero to avoid shifting twice. The estimator's sign expresses audio advance, so delayed audio should move toward a negative offset. Only already-rated A05 Combat-only and A09 base were used, followed by all three A05 conditions again. The classifier receives no paths, merge names, labels or human notes.

## Control results

All nine cases completed, and an independent CPU auditor reconstructed every actual model input tensor from source media without invoking the official composed transform sequence.

| Source | Original modal offset | After +750 ms delay | Near-zero mass, original → delayed |
| --- | --- | --- | --- |
| A05 Combat-only | 0 s | −0.8 s | 0.9982298 → 0.00003677 |
| A09 base | 0 s | −0.8 s | 0.9997329 → 0.00002401 |

The three repeated A05 distributions matched exactly. Muted distributions were diffuse, but the offset-only model still emitted classes; an explicit waveform-silence guard reports null instead. **This is harness abstention, not learned silence recognition or general synchronizability detection.** All necessary frozen control gates pass, without automatically granting audiovisual-quality or held-out qualification.

## Natural merge differences versus the personal ratings

The [second prospective policy](data/2026-09-09-h3-synchformer-calibration-policy.json) permitted one original-only sweep of all twelve already-rated seed03 clips after the controls passed. It recorded the known counterexample first: A05 has human synchronization 4, while the repeated A09 base has ratings 1 and 2, despite both being assigned near-zero model offsets. No thresholds, crops, model choices or score mappings were fitted.

All twelve inferences completed. **Every original had modal offset zero.** Near-zero class mass ranged from 0.9982298 to 0.9998597. The secondary measure, negative absolute modal offset, therefore tied every pair. The primary confidence-based comparison retained all 14 human entries, including shared-control disagreement:

| Pairing | Concordant human-untied comparisons | Discordant | Human-tied comparisons retained |
| --- | --- | --- | --- |
| Combat–Cinema | 6/14 (42.9%) | 8 | 7 |
| Combat–Repair | 9/16 (56.3%) | 7 | 5 |

These are descriptive counts against one person's perception, not population estimates, significance tests or independent samples. The shared clips are correlated. The model's confidence would order the poorly rated base above Combat-only; the latter actually receives the lowest near-zero mass among all twelve. No general audio, action, appearance or overall-preference score is inferred from synchronization confidence.

A second CPU auditor independently reconstructed all twelve model tensors, verified the full human-label mapping/comparisons and confirmed that A05/A09 exactly reproduce their earlier control-run inputs and outputs. No GPU inference was rerun for either audit.

## Disposition and next gate

Keep this implementation as a **gross imposed-delay diagnostic on the tested sources only**. Do not use its near-zero confidence or modal offset as an autotuner reward: it has not demonstrated discrimination of the natural synchronization differences in these personal ratings. No held-out AV2 clips were evaluated. The earlier Qwen joint-quality rejection is unchanged.

The next methodological question is event-level coverage: whether the evaluated region and local contact/impact timing represent the specific faults a reviewer notices. That requires a separately frozen diagnostic and evidence, not crop/threshold searching to improve these scores. Audio-content plausibility, event-level synchronization, general audiovisual preference and the original held-out AV gate remain open. No machine results were imported as human labels or used to fit NP/CT bonuses.

Combined isolated evaluator tests: **91 passed in 0.22 seconds**. The full merger suite was not rerun because production merger code was untouched. Peak recorded CUDA allocation was about 2.02 GiB; timings include preprocessing/warmup and are not a controlled throughput benchmark. At 07:54:36 UTC, the GPU was idle with 29,399 MiB free and the existing ComfyUI queue empty. Both download/install jobs, both GPU runs and both CPU auditors are terminal. No optimizer/default/installed-node changes, commit, push or version bump.

The [machine-readable checkpoint](data/2026-09-09-h3-synchformer-checkpoint.json) pins policies, setup, actual runners, raw outcomes and independent audits. Research storage remains ignored; durable findings and scripts are uncommitted in the working tree.
