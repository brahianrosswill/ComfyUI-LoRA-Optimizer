# H3 autotuner: render-backed evaluation

Status: follow-up improvement work in progress. Correctness/native-export fixes verified; **60 local INT8/FP4 renders completed** (12 cup-pilot clips plus all 48 frozen AV2 clips). One reviewer supplied first-seed audiovisual ratings; these are subjective calibration evidence, not consensus or a general winner. Frame/waveform diagnostics completed on three already-rated clips. Second-seed and held-out ratings remain open; all remaining blind review pages are ready. Latest suite: **686 passed, 3 skipped, 69 subtests passed**.

## Scope and protocol

User authorized local H3 use, a small selection of installed LoRA sets, merge/render experiments, iterative optimizer fixes, and this progress record. Use `/media/p5/Comfyui` with Conda environment `13_env_py313` for merge execution; the community research API is `http://192.168.1.12:8188`. Do not interrupt unrelated jobs, modify existing workflows, publish community results, or commit/push without a further request.

Update: user explicitly authorized generation on the local machine, specifying INT8 convrot diffusion weights, FP4 text encoding, and limited resolution. An existing idle local ComfyUI was discovered on `http://127.0.0.1:8189` (same environment, RTX 5090). No new server was started. Only our pending remote R01 was removed; the unrelated remote upscale continued. The pending-only deletion returned an empty HTTP body, then a read-only queue check confirmed removal. Its watcher was stopped.

1. Resolve exact installed LoRA files, hashes, training partition/basis, source settings, and cached non-explicit examples. Keep source material separate from our benchmark prompts.
2. Verify the execution environment and mathematical/evaluator prerequisites. Reproduce each suspected defect before fixing it; run regression tests afterward.
3. Start with compatible two-LoRA sets and fixed prompts/seeds. Compare base, individual adapters, additive, stable autotuner selection, NP-LoRA and CT-Merging. Keep compatible acceleration adapters and sampling settings fixed. Do not conflate FL2VA and Ref2VA or full/pruned/convrot bases.
4. Run small calibration cases before expanding. Record errors, time, memory, exact configuration and artifact locations. Keep actual audiovisual observations separate from tensor statistics and automated proxy measurements.
5. Change one variable at a time, retain baselines, and reserve unseen prompts/seeds and LoRA sets for validation. Do not claim general improvements from a pilot.

## Evaluation and acceptance

- Check both adapters' intended contributions, prompt/reference adherence, temporal stability, audio quality and event synchronization.
- Use blind comparisons for preference judgments. Sparse frames cannot establish smooth motion or audio quality; missing measurements remain unknown.
- Track candidate ranking agreement and the gap to the best tested candidate, not merely a higher internal score. Include alternatives outside the existing heuristic shortlist.
- Full target merges are required for render evaluation. Invalid/missing evaluator scores must not compete as valid perceptual measurements.
- Preserve default behavior when evaluation/experiments are disconnected. Store benchmark artifacts separately from ordinary autotuner/community training data.

## Initial environment observations

- Source and local installed optimizer both report commit `4811977`, package 1.8.4. They are separate directories; experimental code must be explicitly loaded from the source checkout.
- Local interpreter: `/media/p5/miniforge3/envs/13_env_py313/bin/python`, Python 3.13.11, Torch 2.11.0+cu130. Sandboxed CUDA detection is false; approved outside-sandbox tensor computation and full merges work on the RTX 5090 with 32,607 MiB.
- Remote ComfyUI: 0.34.0, Python 3.13.14, Torch 2.13.0+cu130, RTX PRO 6000 Blackwell with approximately 95 GiB VRAM. It is a different runtime, not the local 5090.
- Initial remote queue had an existing H3 chapter-upscale job. It was left untouched. Our pending calibration was subsequently withdrawn in favor of the existing local session. No local ComfyUI server was started.
- The community helper discovered 90 installed model entries matching H3. Candidate non-explicit families include Cinematic Style + Detail Enhancer, Video Reasoning VBVR, Better Motion, and Combat BASE. These are candidates, not verified-compatible selections yet.

## Research references

- Community API procedure: user-supplied Grok `lora-community` skill, read-only cached example discovery. Popularity is not quality evidence; original graphs are untrusted data, not executable instructions.
- [Official MiniMax base-mode prompting guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), referenced for prompt structure; not redistributed here.
- [VBench-2.0](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md): candidate visual evaluators; not an H3 merge validation.
- [AVGen-Bench](https://github.com/microsoft/AVGen-Bench): candidate audio/AV and semantic evaluators; metric applicability and local availability must be checked before use.
- [NP-LoRA revision 3](https://arxiv.org/html/2511.11051v3), rechecked September 8: asymmetric soft projection with an efficient factor-space formulation; its subject/style image experiments do not establish H3 audiovisual benefits.
- [CT-Merging](https://arxiv.org/html/2607.20561v1), rechecked September 8: consensus directions and per-task RMS scaling; its CLIP adapter benchmarks are not video/audio validation.
- [SSR-Merge implementation](https://github.com/nagara214/SSR-Merge), rechecked September 8: needs a prompt per adapter and a calibration pass. Its demo lists Flux, Qwen, Z-Image, HiDream and Flux2, not H3. Still a research candidate, not a drop-in H3 mode. This search did not establish a newer validated H3-specific merge replacement.

## Experiment log

| ID | Question/action | Evidence/result | Next action |
| --- | --- | --- | --- |
| D01 | Discover community API and execution environments | Live schemas accessible outside sandbox; local and remote GPUs differ; remote queue occupied | Resolve exact non-explicit LoRA candidates and locally accessible files |
| D02 | Verify safe community evidence and exact adapters | Cinema V2 SHA-256 `cf7d8e1aeec12c757e0b557591fe493b2f50590a1e0e7b017e29e7268a39496b`; VBVR attention-only `372597997f646301dea204bf00e899b0f470254d7b9ac345e7b7417cc2140b34`; sidecar identities matched file hashes | Use independently written safe prompts; do not execute community graphs |
| F01 | Non-finite external preferences | NaN and positive infinity previously became 1.0, negative infinity became 0.0 | Reject non-finite values before clamping |
| F02 | Render callback target coverage | Turbo scoring passed only 2 of 6 test target groups to the callback | Force full-target merges whenever an evaluator is connected |
| F03 | Failed external-only evaluator | Callback errors previously fell back to internal weight statistics | Fail the sweep explicitly rather than silently substitute a different objective |
| M01 | VBVR + Cinema, additive, strengths 0.8/0.8 | 7.26 s, 1.73 GiB peak CUDA allocation, 375,794,656-byte export | Render matched baseline |
| M02 | Same pair, NP-LoRA, subject=VBVR/style=Cinema, mu=0.5 | 11.91 s, 1.73 GiB peak allocation, 1,335,963,728-byte export | Quantify compression error before interpreting render differences |
| M03 | Same pair, CT-Merging, common rank 4 / residual rank 16 / scale 1 | 14.29 s, 1.73 GiB peak allocation, 1,335,963,696-byte export | Same numerical validation |
| T01 | Stable autotuner, top 3, full targets, SVD scoring disabled | 1,050 heuristic combinations; 3 actual merges; 36.29 s; 16.56 GiB peak allocation; best internal score 0.6882 | Replay exact winner through Merge Selector; this is not a perceptual score |
| R01 | Base-only cup interaction, seed 2026090801 | Remote accepted graph with no node errors; prompt ID `4068c083-568f-4688-8404-b934a006297d`; never executed | Subsequently withdrawn; see local R02 |
| F04 | BF16 rounding before experimental compression | NP/CT regression fixture had about 0.3% reconstruction error; NP error exceeded the intended method change on some real H3 targets | Preserve FP32 through experimental compression only; compressed patch cache identity invalidated |
| M04 | Exact stable T01 winner via Merge Selector | 6.67 s, 375,794,656-byte export; not a manually approximated per-prefix default | Matched render |
| M05 | Combat + Cinema additive, 0.8/0.8 | 11.76 s, 2.31 GiB peak allocation, 620,309,608-byte export | Further pair comparisons after calibration |
| M06 | Combat + Motion Repair additive, 0.8/0.6 | 10.44 s, 2.31 GiB peak allocation, 620,309,600-byte export | Same |
| M07 | NP after F04, same M02 parameters | 10.85 s; all 312 normalized groups finite; max relative export error 2.8732e-6 | Use corrected export, retain pre-fix artifact as evidence |
| M08 | CT after F04, same M03 parameters | 12.36 s; all 312 normalized groups checked; max relative export error 4.1013e-6 | Use corrected export |
| T02 | Combat + Motion Repair, full-target stable top 2 | 89.00 s; top internal scores 0.561 and 0.485; peak allocation 16.38 GiB | Not an audiovisual preference result |
| R01 withdrawn | Switch generation to authorized local session | Exact pending prompt removed, no running remote job interrupted | Local R02 supersedes remote pilot; do not compare TE profiles as if identical |
| R02 | Local base cup, seed 2026090801 | Success in 92.38 s; 640x384, 124 frames/24 fps, 5.167 s, 32 kHz stereo AAC; prompt `7bee8549-83aa-4494-9036-40a1e39b0d61` | Individual-adapter and merged comparisons |

### Local generation profile and first observation

All local cup comparisons use `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, native video/audio VAEs, 640x384, 124 frames, 24 fps, `res_multistep` / `simple`, 20 steps, no turbo, the same prompt, and seed 2026090801. The existing server confirmed quantization metadata and mixed-precision text/model operations. No simultaneous headless merge jobs run during these renders.

R02 output: `/media/unraid/comfyui/output/h3_autotuner_study_20260908/local-cup-base-01_00001_.mp4`. Inspection at 4 fps shows a coherent red-cup lift, contact with the lips, and return to the saucer; the left hand stays near the table. These sampled observations do not establish full-frame smoothness, finger correctness throughout the clip, or audio synchronization. Stream inspection confirms stereo audio exists, not that it sounds correct.

Four generated comparison adapters were copied (without overwriting files) into the existing local LoRA search path under `/media/p5/model_temps/h3-autotuner-study-20260908/`. No model-path configuration or installed optimizer code was changed. The source adapters remain untouched.

### Precision measurements

`scripts/h3_export_check.py` expands exported factors and compares them with uncompressed FP32 reference deltas, splitting native QKV into the same three components used by normalization. This checks export/compression fidelity, not an independent validation of the research algorithms (separate matrix-reference tests cover those).

- NP before F04: maximum relative error 0.00245384 across 24 sampled normalized groups; maximum error / intended NP change 3.9545.
- NP after F04: maximum error 0.00000151878 on those same 24 groups (about 1,616x smaller); maximum 0.00000287320 across all 312 groups. Maximum error / intended change across all groups 0.001624.
- CT before F04: maximum error 0.00240730 on 24 sampled groups; after F04, maximum 0.00000410128 across all 312.
- Corrected NP/CT exports are 1,419,456,448 / 1,419,456,416 bytes, about 6.25% larger than their pre-fix counterparts. The experimental compressed factors remain FP32; uncompressed storage and stable-mode precision policies are unchanged.
- This does not remove arbitrary rank-truncation error or prove a perceptual improvement. Native-factor NP/CT output is a promising follow-up to avoid dense materialization and rank-64 padding, not implemented in this iteration.

### First-seed render comparison (2026090801)

All seven runs completed successfully, produced 124 video frames and stereo audio, and decoded without errors. The following observations come from **unblinded 4-fps frame inspection** plus whole-clip descriptive metrics. They are not full-motion/audio preference scores.

| Variant | Execution time, seconds | Sampled visual observation | Audio RMS / peak |
| --- | ---: | --- | --- |
| Base | 92.38 | Cup lifted early, held at lips, returned to saucer | 0.004958 / 0.203718 |
| VBVR 0.8 | 63.39 | Very close to base framing and action sequence | 0.004079 / 0.166469 |
| Cinema 0.8 | 41.16 | More frontal framing; later cup lift and return | 0.004598 / 0.210396 |
| Additive 0.8/0.8 | 40.50 | Cinema-like framing; complete lift/contact/return sequence | 0.004863 / 0.278024 |
| Stable winner | 54.07 | Also Cinema-like; complete sampled action sequence | 0.004299 / 0.211131 |
| NP, corrected FP32 | 42.13 | Close to additive framing and timing | 0.004757 / 0.251990 |
| CT, corrected FP32 | 32.26 | Earlier return and longer still ending than additive | 0.001325 / 0.029036 |

No decoded audio samples clipped at absolute amplitude 1. CT's RMS is approximately 11.3 dB below additive in this seed. That flags an audio difference for listening; quieter is not automatically worse, and no event-sync score has been assigned. Shared H3 transformer edits can affect generated audio even when the source adapters lack explicit audio-only keys.

Execution time is not a controlled speed benchmark: base paid cold-loading costs, later runs reused conditioning/model caches, and GPU state varied. CT completing faster is not evidence that its merge mode accelerates H3.

The tiny NP-vs-additive visual difference is consistent with the small tensor-level correction measured for this near-orthogonal pair, but this is an inference, not a general result. No internal ranking formula or default merge recommendation has been changed from these observations. VBVR is a capability adapter, not a subject-identity adapter: assigning it NP's content role is exploratory, not a reproduction of the paper's subject/style task.

### Second-seed check (2026090802)

All five runs succeeded with unchanged parameters. Sampled frames again show cup lift, lip contact, and return for each variant. NP stays close to additive; CT changes the room/framing more noticeably. None of these observations establishes a perceptual winner.

| Variant | Execution seconds | Audio RMS | Audio peak |
| --- | ---: | ---: | ---: |
| Base | 35.18 | 0.003802 | 0.189000 |
| Additive | 30.91 | 0.003942 | 0.183837 |
| Stable winner | 36.74 | 0.002313 | 0.084247 |
| NP, corrected FP32 | 37.91 | 0.003922 | 0.182186 |
| CT, corrected FP32 | 40.77 | 0.004129 | 0.198678 |

CT's lower audio amplitude **did not repeat**: its RMS is slightly above additive in seed 2. Do not label seed 1's quietness a confirmed CT defect. No audio clipping was detected in either seed; intelligibility, unwanted sounds and event synchronization remain unevaluated. No learned video/AV judge was installed or run, and no human preference labels were collected. All twelve videos were decoded and their 4-fps frame grids inspected; these are not full-frame audiovisual reviews.

The local queue was empty after the last benchmark. No render server was restarted or stopped, no user canvas was changed, and no further generation is queued by this study.

### Durable evidence and next gates

The [machine-readable pilot record](data/2026-09-08-h3-pilot.json) preserves 10 local merge/tuner/replay run summaries, four precision checks, and all 12 render manifests, output hashes, timings and descriptive measurements. Original MP4 files remain under `/media/unraid/comfyui/output/h3_autotuner_study_20260908/`; raw logs, tuner data and diagnostic frame grids are in the temporary artifact root. The JSON record is durable in this worktree, but not committed or backed up remotely.

1. Extend to a harder action prompt and the Combat/Cinema and Combat/Repair pairs; add a genuinely overlapping subject/style pair. Keep unseen prompts/seeds for validation. Two seeds of one cup action are insufficient to calibrate the tuner.
2. Collect blinded full-video **and audio** comparisons against individual adapters, additive and the stable winner. Only then test whether external scores improve ranking agreement; do not use luma, motion magnitude or audio loudness as a substitute preference target.
3. Investigate exact native-factor NP/CT outputs to reduce dense memory and rank padding. Preserve signed strengths, alpha, QKV semantics and stock-loader equivalence. This is a numerical/performance improvement candidate, not a promised quality gain.
4. Investigate skipped-sparsification candidates still materializing dense patches and receiving a configuration-based scoring penalty. Reproduce equivalence before changing ranking policy; keep default behavior stable until justified.

### Follow-up goal: verified skip contract (F05)

User activated the broader improvement goal on September 8. Start with reproducible correctness/performance fixes, then exact experimental factor outputs and held-out audiovisual evaluation. Package version remains 1.8.4; no commit/push authorization is inferred.

**Correction to the pilot interpretation:** the old "sparsification skipped" log was misleading. On the >40% conflict-mask guard, both `dare_conflict` and `della_conflict` fell through to **unconditional DELLA**, modifying every input rather than skipping. Therefore the old 28-second candidate was not a verified no-op, and its score difference cannot be attributed solely to representation or penalties.

- Tiny-tensor tests reproduced all eight wrong outputs (two conflict-aware settings, four merge modes), plus two unwanted unconditional-DELLA/RNG calls. Replacing that fall-through makes the guard a true no-op; low-conflict cases still call the requested conflict-aware sparsifier. The existing threshold and TIES behavior are unchanged.
- Two additional integration tests reproduced unnecessary compression and a 0.05 measured-score difference even after the math fix. Verified skipped, plain linear groups now retain native factors after the exact dense guard check; no sampled-analysis shortcut is used. Cleaned/masked inputs, preserved overlays and lossy diff-cache paths retain their fallback.
- Per-group applied/skipped bookkeeping is collected on the result thread and preserved across candidate cache hits. The scorer retains the existing penalty for genuinely applied sparsification, not verified skipped groups. Reports and tuner metrics expose the counts.
- Ranking-cache revision is 1.13.1 (internal, not the package/release version); conflict-aware patch identities are invalidated. Existing saved rankings can be stale because the previously mislabeled candidate really changed the weights.
- Full suite after F05: 658 passed, 3 skipped, 49 subtests passed (21.34 s).
- Real VBVR/Cinema top-three rerun (`vbvr-cinema-tune-02-skip-fix`): 11.477 s total versus 36.289 s in the original pilot; peak CUDA allocation 1,859,297,280 bytes (1.73 GiB) versus 16.56 GiB. The affected candidate's merge phase fell from about 28 s to 1.7 s, all 208 shared groups genuinely skipped, all 312 normalized patches stayed factorized, and its measured score matched the equivalent disabled candidate (approximately 0.688164). This is a correctness/performance result, not a generation-quality win. Cross-run timing is observational rather than a controlled repeated speed benchmark; the candidate now performs different, corrected math.

### Native experimental output (F06)

Rechecked [NP-LoRA v3, Appendix C](https://arxiv.org/html/2511.11051v3#A3) and [CT-Merging Algorithm 1](https://arxiv.org/html/2607.20561v1). The new implementation remains independently written; the papers' image/classification quality claims are not transferred to H3.

- NP projects the content **down-factor**, concatenating it with the unchanged style factors. CT computes its common projected response as `(Uc.T @ B) @ A` and returns the scaled polar factors directly. Neither eligible Pass-2 path materializes the full output update. Analysis remains streaming-dense and still sets peak allocation in this pair.
- All contributors must be eligible plain 2D factors; unsupported/missing participants, masks, cleaning, preserve overlays and spatial/virtual-slice cases retain the existing fallback. Explicit aggressive compression still applies if a native result would exceed its rank limit. Native results retain FP32.
- Initial integration rejected file-based QKV slices using a restriction intended for virtual captures. Correcting that eligibility gate made all 208 shared normalized groups native; stock-loader partial-QKV tests cover the distinction.
- Initial fully native NP output was 644,164,968 bytes. Equivalent full-rank Q/K/V style subspaces produced slightly different floating-point down factors, preventing exact refusion sharing. Canonical QR of the shared style down-factor (only after confirming full numerical support, no energy truncation) restores that sharing without inventing rank-deficient directions.
- Final NP run `vbvr-cinema-np-05-canonical`: **375,794,896 bytes** versus 1,419,456,448 after F04, a 73.5% reduction and essentially additive's size. Pass 2 about 1.1 s; total including analysis/export 8.014 s; peak CUDA allocation 1.73 GiB. All 312 groups checked: maximum relative reconstruction error **8.3009e-8**, all finite. No new render or perceptual improvement is claimed yet.
- CT run `vbvr-cinema-ct-03-native`: **721,694,056 bytes**, a 49.2% reduction from the F04 export. Pass 2 about 2.2 s; total 9.539 s; peak allocation 1.73 GiB. All 312 groups checked, all finite, maximum relative reconstruction error **4.3588e-6** (comparable to the previous FP32-compressed CT error, 4.1013e-6). CT response reassociation and near-degenerate polar directions can amplify FP32 differences; this is not bitwise equivalence or a quality claim.
- CPU tests cover native/dense agreement, signed mixed ranks and alpha, zero/cancelling inputs, role reversal, truncation, canonical factor sharing, and no Pass-2 dense preparation/recompression on eligible fixtures. Real stock-loader tests cover native partial QKV and signed CLIP output.

### Mixed diffusion/CLIP architecture detection (F07)

The larger native roundtrip fixture exposed another existing bug: a partial H3 adapter with CLIP `self_attn.q_proj` keys was detected as ACE-Step, even with an H3 model hint, and its H3 Q target was renamed and skipped. Focused tests reproduced this and the analogous WAN misclassification. Diffusion architecture heuristics now exclude explicitly prefixed TE keys when diffusion keys are also present. TE-only legacy detection and the earlier SD1/SDXL bundle detection remain unchanged; an H3 hint resolves otherwise unknown partial keys. This does not assert that arbitrary ambiguous adapter names prove H3 compatibility.

Validation after F06/F07 and the durable-record test: **666 passed, 3 skipped, 69 subtests passed** (18.60 s), JavaScript migration **3 passed**, and the expanded actual-ComfyUI dense/native NP/CT + signed CLIP/partial-QKV roundtrip **PASS**. The [follow-up numerical record](data/2026-09-08-h3-followup-numerical.json) preserves eight baseline/intermediate/final run records, export hashes, exact integer timestamps, precision summaries and relevant timing logs. Code and research remain uncommitted.

Next active milestone: freeze these validated exports/code identities, expand to harder multi-pair/multi-prompt renders on the existing INT8 convrot/FP4 session, and collect genuinely audiovisual comparisons before changing candidate ranking beyond the verified skip correction. No new videos were rendered during F05–F07, and no perceptual preference labels were created.

### Expanded audiovisual benchmark: AV2 (in progress)

The [frozen AV2 plan](data/2026-09-08-h3-av2-plan.json), SHA-256 `4982928f36c1052dfc7092456d1d612a3277d94e549ef72d5eb618c27c1d7d20`, reserves 48 physical renders / 56 comparison entries. Shared base and Combat-only controls are reused within the same prompt/seed, not counted as independent evidence. This is a planned matrix, **not 48 completed renders**.

- Pairs: Combat/Cinema 0.8/0.8 and Combat/Motion Repair 0.8/0.6. Seven arms each: base, each individual adapter, additive, exact stable winner, NP (mu 0.5) and CT (common 4 / residual 16 / scale 1).
- Calibration: boxing, seeds 2026090803 and 2026090804. Held-out: two-person padwork, seeds 2026090811 and 2026090812. Both prompt texts, model profile and export identities were frozen before new render inspection. Held-out outputs must not inform candidate selection. The first bounded batch requests only the first calibration seed (12 physical clips).
- All renders retain the established local INT8 convrot / FP4 profile, 640x384, 124 frames, 24 fps, 20 steps, no turbo. `scripts/h3_benchmark.py` validates installed export hashes, prepared graphs and prompts; it submits serially, follows recorded prompt IDs over WebSocket, stops for existing queued work, and refuses automatic resubmission of failures.
- `scripts/h3_av_review.py` creates a self-contained local player with randomized method labels and a separate private key. Stream-copy remuxing removes workflow metadata; no loudness normalization, interpolation or frame selection changes the review media. Explicit full-motion and full-audio review plus all six ratings are required for each exported label. This is a human-label collection tool, not an automatic judge or proof someone actually listened.
- Runtime capability check: no local Ollama service responded on port 11434. An attempted audio tool input returned **"audio content omitted because you do not support audio input"**. Thus this session cannot independently listen to the clips. No learned AV judge, new server, installation or external clip upload was introduced. Full AV preferences remain a required external input; descriptive frame/audio statistics do not fill that gap.
- Read-only community discovery returned 90 name-matching H3 entries. A broader local file inventory found additional H3-named-directory adapters, but no verified non-explicit subject-identity candidate. Do not call the capability pairs a subject/style identity reproduction. No extra model was downloaded or included on filename evidence alone.

The [AV2 numerical record](data/2026-09-08-h3-av2-numerical.json) preserves ten additive/tuner/winner/experimental run identities. All four new experimental exports were checked against all 312 normalized target groups and are finite:

| Pair / mode | Export bytes | Total merge/export seconds | Max relative reconstruction error |
| --- | ---: | ---: | ---: |
| Combat/Cinema NP | 620,309,840 | 12.699 | 1.0127e-7 |
| Combat/Cinema CT | 1,100,257,904 | 15.066 | 4.1907e-5 |
| Combat/Repair NP | 620,309,824 | 12.089 | 4.9301e-7 |
| Combat/Repair CT | 1,100,394,080 | 15.465 | 6.1156e-6 |

CT's Combat/Cinema error is higher than the earlier pilot, about 0.0042%; numerical equivalence is approximate, not bitwise. These errors are still small relative to the intended CT change (maximum error/change 3.2368e-5). None of these numerical measurements establishes a perceptual benefit.

Stable Combat/Cinema top-three took 11.218 s. The conflict-aware candidate genuinely skipped all 312 groups and matched the disabled score (~0.670), independently extending F05's real-adapter evidence. Stable Combat/Repair top-two took 60.265 s and retained the earlier ~0.561 / ~0.485 scores; its SLERP-containing candidate still requires dense work and peaked at 16.29 GiB. No claim that F05 removes every tuner bottleneck is justified. Exact winner exports were replayed through Merge Selector, not approximated manually.

Storage incident: `combat-cinema-ct-01-native` and `combat-cinema-winner-02` failed during atomic save with `/tmp` disk-quota errors; neither has a usable export. Both run directories and logs were preserved. After all writers terminated, the full study tree was moved to ignored, disk-backed `.h3-study-artifacts/20260908/`; `/tmp/h3-autotuner-study-20260908` is now a symlink to it, preserving prior manifest paths. New directories `combat-cinema-ct-02-native` and `combat-cinema-winner-03` succeeded. No unrelated temporary files or model files were deleted. Eight verified AV2 exports were copied without overwriting into the existing dedicated LoRA search directory.

Validation at AV2 start: **671 passed, 3 skipped, 69 subtests passed** (30.91 s). New tests cover fixed-profile graphs, split/seed separation, shared-control accounting, failed-history rejection, explicit audiovisual-label requirements, exact review/media identity, and output-path containment. Production optimizer code remains at the F07 identity; no ranking change is justified by AV2 yet.

#### AV2 calibration seed 2026090803: completed

All **12 physical videos** succeeded; their executed graphs matched the prepared graphs, and whole-clip decoding verified 124 frames, 640x384 at 24 fps, with 32 kHz stereo audio. The [hash-backed render record](data/2026-09-08-h3-av2-calibration-seed03.json) contains no missing jobs and **no quality labels**. Timings range from 32.119 to 78.685 seconds, with model/conditioning caching uncontrolled; do not infer mode acceleration.

The Combat/Repair NP clip and Combat/Cinema CT clip each have **one** decoded channel sample beyond absolute amplitude 1 (peaks 1.00189 and 1.01456; fraction 3.02418e-6). The existing metric name `clipped_fraction` is a threshold diagnostic, not proof of source hard-clipping or an audible defect. The remaining ten clips have no such samples. Listening is needed before changing any method based on this flag. Quietness/brightness/motion magnitude have not been used to rank them.

The local review page is `.h3-study-artifacts/20260908/av2-review-seed03/review.html` (about 13 MiB, self-contained; open in a browser). It presents 14 comparison entries in two adapter-pair groups, including two reused controls. The private key remains separate. An independent FFmpeg verification checked **all 12 unique remuxes**, confirming identical decoded video hashes and float32 audio hashes versus the originals, and absence of workflow/prompt metadata. Proof: `av2-review-seed03/media-verification.json`, review ID `2298ae831788d764f8df`. No gain correction was applied.

A human review was requested asynchronously; no ratings had been received at this checkpoint (the later import is documented below). Method labels are hidden in the page, but the agent previously displayed one unblinded additive diagnostic frame grid. A reviewer who recognizes that grid has partial prior exposure; the next calibration seed must remain undisplayed before blind review. No held-out padwork clips had been rendered or inspected at this checkpoint.

The suite subsequently passed **673 tests, 3 skipped, 69 subtests** (32.13 s); the review-page construction test also syntax-checks its JavaScript and ensures private job/method names are absent from the public page. Tests are synthetic harness checks, not perceptual labels.

An attempted post-render experimental ranking run (`combat-cinema-experimental-tune-01`) failed in Pass 1 with CUDA OOM: the empty-queue ComfyUI process still retained about 25 GiB of model cache. After verifying both queue lists were empty, `/free` released **idle model cache only**; no jobs, history or server process were interrupted. Free VRAM increased from about 1.9 to 25.9 GiB. The headless runner now checks conservative free-memory thresholds (20 GiB for tuning, 6 GiB for merge/replay) before large allocations; it never unloads another process itself. This is an orchestration fix, not an NP/CT numerical defect or a general VRAM estimator. The failed attempt is preserved, and new run directories are used for retries.

#### Internal ranking predictions, before preference labels

Both retry sweeps succeeded and included three stable candidates plus additive, NP and CT, all full-target. The [internal-ranking record](data/2026-09-08-h3-av2-internal-rankings.json) retains exact scores/configurations and code identities. The native-factor production code is unchanged from the frozen exports.

| Pair | NP | Additive | Stable full-strategy winner | CT |
| --- | ---: | ---: | ---: | ---: |
| Combat/Cinema | 0.707735 | 0.707517 | 0.669984 | 0.619730 |
| Combat/Repair | 0.567505 | 0.553397 | 0.561444 | 0.520482 |

These are **predictions from tensor statistics**, not video/audio scores. Cinema NP's lead over additive is only 0.000218 and cannot justify a perceptual superiority claim. `scoring_svd=disabled` remains fixed; effective-rank fields of zero mean unmeasured, not zero-rank updates. No external evaluator or synthetic preference labels were supplied to these sweeps. At this checkpoint, ranking agreement/regret remained uncomputed pending actual AV labels.

#### First human calibration review: one person's preferences

The user supplied review `2298ae831788d764f8df`, explicitly cautioning that this is one person's perception. The [imported record](data/2026-09-08-h3-av2-ratings-seed03.json) preserves all six original integer grades, free-text notes, media identities and provenance. The original submitted JSON has SHA-256 `562cc2e5886ea5093f6a40ae0a797c9b839a7937f796484382da2a711b9964a2` and was not modified. The reviewer is recorded as R1, not pooled with machine judgments. All 14 entries affirm full video review and listening; they cover only 12 unique videos, one prompt/seed and two overlapping adapter pairs.

| Pair | Additive | Stable winner | NP | CT | Combat only | Second adapter only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combat/Cinema | 2 | 2 | 2 | 3 | 3 | 3 |
| Combat/Repair | 2 | 3 | 3 | 3 | 3 | 2 |

These are R1's **overall ordinal grades, 0–4**, not averaged dimensions or population-quality estimates. CT exceeds additive on this seed in both pairs; NP does so only with Repair. No merge exceeds Combat alone overall. For Cinema, CT trades higher appearance (3 versus Combat's 2) for lower action, temporal, audio and sync grades (3 versus 4). Overall ties do not imply identical strengths.

The same base and Combat videos were each presented twice. Their overall grades agree (base 2, Combat 3), while some component grades vary by one point. This variation is preserved as context-dependent judgment; repeated controls are not independent trials and are not silently averaged away. The prompt requests two straight punches but does not require alternating arms: the one-arm complaint is an additional natural-motion preference, not by itself an explicit prompt violation.

The tensor ranker places CT below additive for both pairs, opposite R1's overall ordering on this seed. This is a concrete calibration mismatch, but insufficient evidence to invert the ranker or prefer CT globally. Missing second-seed and held-out judgments remain missing. No fitted score, significance test, new preference default or automatic memory-training entry was created.

The internal top pick is NP for both pairs. On Cinema, R1 grades that pick 2 versus the best tested merged arm's 3 (CT); on Repair, the pick and best tested merged arms all grade 3. These are gaps of one ordinal grade and zero within this reviewed seed, not utility regret or independent accuracy estimates. Individual-adapter controls remain important: neither pick exceeds Combat alone overall.

Read-only timing checks on the first-seed base, Combat/Repair additive and Combat/Repair CT found video and audio stream starts all at zero (video 5.166667 s, audio 5.167 s). This rules out a detected **container start-time offset** in these clips, not a generated event-sync defect. Independent review-remux decode equality also excludes altered samples in review preparation. The user's request to compare frames against the waveform will be handled as event-timing diagnostics on already-reviewed calibration clips, not a substitute for listening or quality labels.

#### AV2 calibration seed 2026090804: completed, not yet rated

All 12 renders succeeded and their executed graphs match the prepared graphs. The [second-seed technical record](data/2026-09-08-h3-av2-calibration-seed04.json) verifies 124 frames at 640x384/24 fps and 32 kHz stereo. Nine clips contain 1–6 decoded channel samples beyond absolute amplitude 1; these remain threshold flags, not proof of audible clipping. Rendering times span 30.966–69.773 seconds with uncontrolled caching, not a speed comparison.

The self-contained page `.h3-study-artifacts/20260908/av2-review-seed04/review.html`, review ID `e85a514ebe54afe967cb`, is ready. All 12 unique remuxes passed full decoded-video and float32-audio identity checks, with workflow metadata absent. Neither these clips nor their diagnostic frame grids have been displayed or visually inspected. The user will be unavailable to rate for several hours; no reminders or invented labels are needed while independent checks continue.

Validation after rating import and reusable review verification: **678 passed, 3 skipped, 69 subtests passed** (20.48 s). Four synthetic preference-summary tests cover ties, missing labels, repeated-control variation and split/duplicate rejection. They are not study labels.

#### Held-out policy freeze, before generation or inspection

The [decision record](data/2026-09-08-h3-av2-heldout-policy.json) leaves every AV2 candidate/export and the tensor ranker unchanged. It carries CT-versus-additive as a hypothesis from the first reviewed seed, not a selected production default. Remaining calibration labels can test repeatability but will not silently alter this validation matrix. Both padwork seeds retain all seven arms, including individual-adapter controls. No extra strength search or selective rerender is added. Full-clip technical audits and blinded review preparation are permitted; perceptual judgments remain separate and held-out results must not be used to fit this policy.

#### Frame/waveform follow-up requested by the user

The user suggested comparing shots/frames against the audio waveform while human review is unavailable. `scripts/h3_av_sync.py` now generates all-frame stereo waveform sheets and a local full-resolution frame-step inspector from **already-rated calibration clips only**. It uses decoded frame presentation timestamps and checks audio sample-count/timestamp continuity, including any nonzero start; it does not assume the streams start together. Channel-preserving 5 ms min/max and RMS bins avoid cancellation from averaging opposite-phase channels. Original clips, loudness and timing remain untouched. The timing/probing approach uses [FFprobe's frame and stream inspection](https://ffmpeg.org/ffprobe.html); waveform visualization is diagnostic, not sound classification.

All 124 frames of base, Combat/Repair additive and Combat/Repair CT were inspected in timestamped sheets, with selected contact windows enlarged. The [hash-backed diagnostic record](data/2026-09-08-h3-av2-sync-seed03.json) explicitly separates assistant visual annotations from R1's listening/preferences. These three clips were selected after seeing ratings and method identities; this is unblinded diagnosis, not another independent trial.

- Additive: a large energy peak at 0.550–0.555 s occurs while the hands are retracting/near the head, before the next clear contact frames around 0.625–0.667 s. Another peak at 1.495–1.500 s follows contact already visible around 1.417 s. Sound identity is unknown, but the different local relationships argue against assuming one uniform stream delay.
- CT: the extra additive action/burst around 0.55–0.67 s is absent. A later contact around 1.417 s has a nearby energy peak at 1.450–1.455 s. This is a plausible local association, not proof the sound is a convincing punch or that CT is generally synchronized.
- Base: visible contact begins between frames 7 and 8 (0.292–0.333 s), while the first dominant energy peak is at 0.410–0.415 s. This is another concrete region for a listener to check, not an exact measured perceptual delay.
- The consecutive-frame inspection also suggests **four punch cycles in additive and three in CT**, versus the prompt's two. Both therefore retain an action-adherence issue on this seed. A higher internal score or the presence of nearby waveform peaks would miss that distinction.

Frame spacing is about 41.7 ms; motion blur/occlusion add annotation uncertainty. Energy peaks are not onsets, shoe/chain sounds can produce peaks, and a tail can be nearer to contact than the actual impact onset. A future event-aware evaluator needs independently identified contacts and sound onsets, one-to-one association, and unmatched-event reporting; nearest-peak or global motion/audio correlation alone is not a validated score. No audio shift, generated-audio repair, preference label or optimizer ranking change was made.

Local inspectors are under `.h3-study-artifacts/20260908/av2-sync-seed03-{base,repair-additive,repair-ct}/inspect.html`. They are separate from the blind review pages, with lossless extracted stills and waveform navigation. Eight synthetic tests cover timestamp order, audio gaps/sample counts, nonzero offsets, stereo phase preservation, partial bins, silence/non-finite inputs, peak semantics and browser controls. A real FFmpeg synthetic AV fixture with a known 125 ms audio offset verifies that the diagnostic preserves the offset instead of accidentally aligning stream starts.

Code reinspection found the experimental group/single-patch and lossy diff-cache paths already disabled on experimental merges; full-target scoring also remains enforced. These safeguards were not changed. No additional production defect has been established by the waveform work.

#### Held-out rendering complete; preference validation pending

Both frozen padwork seeds completed successfully: **24 physical videos / 28 comparison entries**, with no failed or missing jobs. The [seed 2026090811 record](data/2026-09-08-h3-av2-heldout-seed11.json) and [seed 2026090812 record](data/2026-09-08-h3-av2-heldout-seed12.json) verify exact prepared/executed graphs, whole-clip decoding, 124 frames at 640x384/24 fps, and 32 kHz stereo. Timing ranges were 29.015–48.570 s and 28.937–29.829 s, respectively; caching was not controlled, so these are operational timings only. Three first-seed and two second-seed clips have above-unit decoded sample flags; these have not been interpreted as audible clipping or used to select a method.

All 24 review remuxes have identical full decoded video and float32 audio hashes to their originals, with identifying workflow metadata absent. Neither held-out videos nor their generated frame grids have been visually inspected or listened to. No extra render, revised prompt, strength ablation or preference-driven policy change was added. The existing local server's queue was empty after completion; no server was started/restarted and no unrelated job was interrupted.

| Next review | Local page | Review ID | State |
| --- | --- | --- | --- |
| Boxing seed 2026090804 | [Calibration repeat](../../.h3-study-artifacts/20260908/av2-review-seed04/review.html) | `e85a514ebe54afe967cb` | Ready, not rated |
| Padwork seed 2026090811 | [Held-out first seed](../../.h3-study-artifacts/20260908/av2-review-seed11/review.html) | `47d7e83e8ed0056eeee9` | Ready, not rated |
| Padwork seed 2026090812 | [Held-out second seed](../../.h3-study-artifacts/20260908/av2-review-seed12/review.html) | `b4cf4e7a07569a6269f2` | Ready, not rated |

Prefer the calibration repeat before opening held-out reviews. Partial rating exports are supported; missing cases are not imputed as failures. The remaining 42 entries represent 36 unique clips, not independent repetitions. Private method keys remain outside these pages. There is no need to review while unavailable.

Provisional shipping recommendation: the tested correctness and numerical-efficiency fixes have technical support; keep NP/CT opt-in and leave preference ranking/defaults unchanged. R1's first seed motivates validation, not an automatic CT boost. Completing the audiovisual goal still requires the remaining real judgments and their matched, uncertainty-aware comparison, or an explicit user decision to narrow that scope. A waveform diagnostic does not close that evidence gap.

Final checks for this continuation: **686 passed, 3 skipped, 69 subtests passed** (18.20 s), `git diff --check` clean. The optimizer, experimental merge implementation, frozen render harness and plan hashes remain unchanged from the pre-held-out policy record. New work is confined to local diagnostics, tests and research artifacts. No commit, push, version bump or update of the separately installed optimizer was performed.

### Interpretation and reproducibility limits

Artifacts live under `/tmp/h3-autotuner-study-20260908/`; each local run has its own manifest, hashes, report and log. Temporary artifacts are not a durable backup. `scripts/h3_merge_study.py` imports this source checkout against the user's actual local ComfyUI runtime, not the separately installed optimizer checkout.

The local runner loads **complete real adapter tensors**, but only the actual checkpoint's safetensors header to build a meta-device shape model. It validates native ComfyUI mapping and export without loading another full H3 generation model. This is restricted to base-independent, non-AdaLN additive adapters, with magnitude taming disabled. The matching base is FL2VA pruned INT8 convrot, not Ref2VA. No inference or audiovisual evaluation occurs in this local shape-only process.

Initial pilot NP/CT exports used explicitly selected aggressive patch compression (rank 64 on 208 shared normalized groups); 104 unique groups retain original low-rank factors. The additive path stays factorized and does not need that compression. F04 quantified and reduced compression error; F06 subsequently replaced this representation on eligible targets. No arbitrary rank-truncation losslessness claim is made.

Pair analysis: raw sign disagreement 50.04%, excess conflict 0.12%, cosine similarity -0.00154, subspace overlap 0.00605. The near-orthogonal classification is appropriate evidence against treating the 50% sign base rate as destructive conflict; it is not evidence of better rendered motion. The stable shortlist's conflict-aware candidate **logged** skips for all 208 shared groups, but F05 later reproduced that it actually fell through into unconditional DELLA. The verified follow-up distinguishes that math defect from representation and scoring effects.

The first tuner run revealed that `memory_mode=disabled` does not disable analysis/pair caches: it wrote four cache JSONs into the local ComfyUI `models/autotuner_memory` directory. The study runner now redirects `folder_paths.models_dir` before optimizer import so subsequent analysis/patch caches stay inside the run directory. No community dataset was recorded. Existing unrelated cache files were not removed.

The cached [Cinema comparison example](https://civitai.com/images/141431746) is a 30.375-second concatenated comparison, not one 30-second H3 generation. Its three inspected frames are labeled Cinematic Style v2, No LoRA, and Cinematic Style v1; they show an adult ballroom couple with different framing. Its embedded graph confirms Cinema V2 on the FL2VA pruned INT8 convrot base with turbo disabled. Sparse frames cannot validate its motion or audio. Source prompt and graph were preserved separately and not submitted.

R01 uses our own single-shot cup interaction prompt, native `res_multistep` / `simple`, 20 steps, 832x480, 124 frames at 24 fps, and no turbo LoRA. Keeping the Cinema trigger in all cup variants controls prompt wording. These are pilot settings, not a recommendation established by this study.

## Changes and validation

- Added three regression tests reproducing F01–F03, confirmed failures before the fixes, then confirmed they pass. Finite out-of-range scores retain the established clamp behavior. Evaluator cache identity was incremented only on the evaluator path; disconnected default behavior and package version remain unchanged.
- Full Python suite including offline harness tests: **653 passed, 3 skipped, 35 subtests passed** (19.30 s). F04 was reproduced red for both NP and CT before the fix.
- JavaScript dynamic migration tests: **3 passed**.
- Actual local ComfyUI integration: **PASS**, including H3 partial QKV, signed alpha, dense AdaLN/bias/norm, LoCon, FP32 export, atomic rejection, signed CLIP, and NP/CT round trips.
- Source and research scripts are uncommitted. No version bump or push performed.

## User-authorized commit checkpoint

After the above experiments, the user requested committing the improvements. This snapshot packages the implemented F01–F07 fixes, regression/integration tests, local evaluation scripts and hash-backed research records. Earlier uncommitted-status statements describe their respective checkpoints. Large video/model/browser artifacts remain ignored and local; unrelated planning documents are excluded. Package version remains 1.8.4, no push or installed-node update is included, and the remaining audiovisual validation is still open. No preference-driven ranking change is claimed.
