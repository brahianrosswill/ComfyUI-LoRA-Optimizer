# H3 autotuner: render-backed evaluation

Status: technical shipping audit passed; overall audiovisual qualification remains open. F01–F07 correctness/native-export fixes are verified and committed; the research-only mapped-storage hook and F08 opt-in search prototype remain uncommitted. **All 144 character/effect benchmark clips / 17,856 frames are technically audited. All 72 held-out detailed frame/waveform reviews and matched character-only comparisons are saved**, covering four complete blocks. Each block was unblinded only after all eighteen detailed observations were frozen and verified. Numerical qualification covers all four pairings. **All 40 paired timing trials are complete and audited: F08's pair medians are 11.19–21.01% slower, with all twelve paired winner payloads unchanged. Withhold this prototype as-is.** Matched evidence shows seed-dependent identity/action/wardrobe trade-offs, not an overall NP/CT winner. NP's clearer post-smile hand placement in crossed seed 42 does not restore Sully's vest; Series30's character-only two-attempt boxing result becomes at least three attempts in every merged arm. No learned held-out evaluator, automatic NP/CT preference or default change is justified. Earlier material comprises 60 cup/AV2 clips, 16 character-only qualification clips and one dense-loader generation smoke. One reviewer supplied earlier AV ratings, not character-study labels or a consensus. Assistant audio listening, sound identity and perceived synchronization remain unassessed. Fresh final checks: **907 passed, 3 skipped, 155 subtests passed**, three JavaScript checks and real CPU ComfyUI loader round trips passed. The older AV2 study still has 36 unrated clips / 42 comparison entries; the completed character-frame review does not replace that listening gate. See the goal-wide audit below.

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

- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE), checked September 8: the published local-weight license excludes the EU, UK, US and South Korea and offers a separate licensing contact for excluded territories. This is a local-model licensing caveat, not legal advice or a claim about the separate paid API's terms. Model files remain local and are not redistributed by this study.
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

## Character-reference benchmark: authorized acquisition

The user proposed comparing merged character LoRAs against creator examples and matched character-only generations, and authorized downloading missing H3 character adapters. Two were installed through the existing LoRA Manager into the shared `MiniMax H3/character/h3-identity-study-20260908/` directory. Both subsequently appeared in the existing local ComfyUI (`8189`) loader. No service restart, render submission, credential extraction, production-code change, commit, push or version bump was performed.

The [acquisition record](data/2026-09-08-h3-identity-acquisition.json) pins exact versions, files, SHA-256 hashes, sizes, reference media and validation limits. Civitai/H3/model-registry skill checks guided exact-version selection, the shared-library destination, and compatibility inspection. The generic MCP downloader was not used for installation because it targeted the remote service without a local workspace; LoRA Manager's established shared-library route was verified against the known Cinema file instead.

| Adapter | Size | Format and coverage | Source example |
| --- | --- | --- | --- |
| [Sully v1.0](https://civitai.com/models/2886539?modelVersionId=3263006) | 148 MiB | Rank 16, AI-Toolkit BF16, 208 native targets including token refiner | [Doorway close-up](https://civitai.com/images/140718260); strength 1, 20 steps, res_multistep/simple, Turbo disabled |
| [Fictional Women Series30 v1.0](https://civitai.com/models/2845077?modelVersionId=3212165) | 284 MiB | Rank 16, Musubi-style FP32, 200 native targets, no token refiner | [Clothed adult cafe portrait](https://civitai.com/images/139185912); trigger `ai30`, strength 0.8, 8 steps, Turbo strength 0.7 |

All downloaded tensors are finite. Real ComfyUI's CPU loader consumed every native key; optimizer normalization also consumed every resulting key, addressing 312/300 split-QKV targets respectively. All factor output shapes match the selected FL2VA INT8-convrot checkpoint header. Neither adapter contains AdaLN or DoRA. **These are loader/shape checks, not an INT8 forward pass or evidence of identity quality.** Series30's training metadata explicitly names the pruned INT8-convrot base and `experimental_image_only`; Sully's published example uses the matching INT8/FP4 loaders.

Original videos, full API graphs, UI workflows and node-identified prompt fields were preserved under `.h3-study-artifacts/20260908/identity-acquisition/`. Three sparse frames per reference were actually viewed; no audio was listened to. Sully's prompt explicitly asks for green/purple fur and also contains `ltxsully`, while the trained trigger is `h3sully`; preserve those distinctions when creating a separate adapted prompt. Series30's embedded metadata records enhancement/sharpening, frame interpolation and resizing. Its video is therefore a qualitative identity reference, not a pixel-perfect raw-generation target.

Next: qualify reproducible character-only local baselines before comparing merges. Freeze prompt, seed, quantization and resolution within comparisons. Score unwanted identity drift separately from the intended action/style/transformation so an adapter that ignores the second concept cannot win by preserving identity alone. Include base, each adapter alone, additive, explicit stable methods, stable tuner winner, NP and CT as appropriate; reserve held-out prompts/characters. No new scoring weights or default method have been selected from these references.

### Character-only qualification: 16 clips completed

The [qualification plan](data/2026-09-08-h3-identity-qualification-plan.json), SHA-256 `566db7c77212267ee8dcf8e6f304123e9ac52996968036633d5a274390e57127`, was frozen before generation or inspection. Its matrix is two characters × two prompts × two seeds × base/character-only. Sully strength is 1; Series30 is 0.8. One prompt per character preserves the complete creator text, including trigger quirks. The other is a separate continuous head-turn probe. H3/core/prompt-writing skill guidance informed the fixed loader profile, structured probe text and existing-session execution; creator workflows were not imported or executed.

All jobs used the existing local `8189` session: **INT8 convrot diffusion, FP4 text encoder, 640×384, 124 frames at 24 fps, 20 steps, Turbo off**. Both seeds, 2026090821 and 2026090822, are explicitly calibration. Neither character can subsequently be described as unseen; future validation needs separately frozen unused prompts/seeds, and another character to test unseen-character generalization. The previous AV2 held-out material and frozen plan remain untouched.

The [seed 21](data/2026-09-08-h3-identity-qualification-seed21.json) and [seed 22](data/2026-09-08-h3-identity-qualification-seed22.json) technical records contain **16 successful executions, no missing jobs**, exact prepared/executed graph matches, full-clip decoding, and 32 kHz stereo audio. None has a decoded audio sample exceeding absolute amplitude 1; this does not establish sound quality or synchronization. Recorded execution intervals total 601.174 seconds, ranging 29.386–42.448 seconds. Caching is uncontrolled, so these are not merge-mode speed comparisons or end-to-end wall time. The local queue was empty afterward; no server restart, cancellation or unrelated-job interruption was used.

#### What the visual inspection supports

All 16 diagnostic sheets were inspected at 4 fps (21 actual samples per clip, four black padding tiles), along with three sampled frames from each creator reference. These are **unblinded assistant observations, not human preferences, continuous playback or listening**. The [observation record](data/2026-09-08-h3-identity-qualification-observations.json) links their exact media/audit identities and limitations.

- Both adapters visibly move character appearance closer to the creator reference on both seeds and both prompts. Sully's short curved horns, furry muzzle and broad nose distinguish it from the base's generic long-horned monster. Series30's facial appearance and long loose hair move toward the fictional reference. This supports using both as exploratory identity controls, not a validated numerical identity score.
- Both also alter framing, clothing or background. Sully's source seed 22 begins with the face largely hidden or outside the crop. Such visibility changes must be reported, not dropped to raise similarity. The preserved source explicitly asks for green/purple fur; canonical blue is not the target here.
- The turn probes expose baseline pose-adherence limitations: shoulder/body motion, stronger-than-requested profile, and imperfect frontal starts. Series30 seed 22 turns toward opposite screen sides in base and character variants. Same seed therefore does not guarantee corresponding poses or a valid frame-by-frame pixel comparison.

The self-contained local page `.h3-study-artifacts/20260908/identity-qualification-review/review.html` presents eight reference/base/character rows. All **18 unique original MP4s** were verified byte-for-byte against their embedded SHA-256 identities; JavaScript syntax was checked. Original timing and audio are retained, playback starts muted, and playing another panel pauses the previous one. Creator playback is independent because its timing/workflow differs. This is an unblinded qualification page with **no ratings**, not another blind AV2 study.

#### Newer metric research and next experiment

[MaSC (May 2026)](https://arxiv.org/html/2605.22469v1) separates concept preservation from background prompt following. Its preservation branch matches foreground-reference patch features against output patches; its other branch uses masked background pooling and a subject-stripped prompt. The published backbone is SigLIP2 SO400M-NaFlex, and the reported evaluation concerns single-concept images, not H3 audiovisual merging. It requires external masks; a cached smaller SigLIP2 model is not an equivalent implementation.

Our design inference is to investigate foreground-aware identity measurements, but **not use background-only prompt following to judge an action or body transformation**. Those intended changes occur on the subject itself. Visibility failures must remain explicit; identity, intended effect, temporal consistency and sound need separate outputs. No MaSC package, learned judge or new scalar score was installed or integrated at this checkpoint.

The next separate, frozen merge matrix should retain matched character-only and effect-only controls, additive, explicit stable methods such as SLERP/TIES, the exact stable tuner winner, NP and CT. Candidate selection must not reward simply suppressing the second adapter. These qualification observations do not justify new ranking weights or a preferred merge default.

#### Harness changes and verification

Added `scripts/h3_identity_study.py` for pinned-source preparation and serial existing-session execution, and `scripts/h3_identity_review.py` for the local comparison page. Submission intent is persisted before POST; ambiguous network failures stop for reconciliation rather than duplicate submission. Existing prompt receipts resume through the established WebSocket watcher. Frozen graph/source identities are checked, and queued unrelated work pauses new submissions.

Fourteen new tests cover matrix/profile preservation, source prompt fidelity, frozen-file checks, failed or mismatched history, queue protection, ambiguous submission, resume behavior, review completeness, media identity and HTML/JavaScript safety. Full suite: **700 passed, 3 skipped, 69 subtests passed** (23.53 seconds). Production optimizer/experimental-merge code and the old AV2 plan hashes are unchanged. These additions remain uncommitted; no push, installed-node replacement or version bump was performed.

### Crossed character/effect merges: numerical qualification

The next comparison crosses Sully and Series30 with both Cinema and Combat. Character strengths remain 1 and 0.8 respectively; each effect strength is 0.8. The [prospective protocol](data/2026-09-08-h3-character-merge-protocol.json), SHA-256 `b54c938768bc311b065479c6568700aa0b7ce303eadc37e0c8d9b3da535b9ecf`, records all four pairs, nine intended arms, separate unused calibration/held-out seeds, and eight structured prompt expansions. Its 144-clip matrix is **planned, not rendered**. Both identities are already calibration-exposed; held-out here does not mean unseen characters. No character-plus-effect video has been submitted or inspected at this checkpoint.

`scripts/h3_merge_study.py` now accepts these pairs, explicit global SLERP/TIES arms and a recorded compression choice. Existing pilot defaults are preserved. Twelve real additive/NP/CT exports completed with smart/native factor preservation; SLERP, TIES and stable-tuner replay have not yet been run on these character pairs. NP uses character slot 1, effect slot 2, mu 0.5; CT uses common rank 4, residual rank 16 and scale 1.

The [full numerical record](data/2026-09-08-h3-character-merge-numerical.json) pins all twelve exports, source identities, production hashes, timings and checks. The new `scripts/h3_native_export_check.py` loads original AI-Toolkit/Musubi files and exported adapters through **real ComfyUI's native mapping, without the optimizer's key normalizer**. It requires exact union coverage and all tensor keys consumed, then compares each Q/K/V component and other target against FP32 additive/experimental reconstruction. The experimental formula is the same research implementation used by the optimizer; this is an independent loading/export check, not an independent validation of the NP/CT algorithm itself.

All **3,744 target checks** passed finite/coverage checks (312 per export, 208 fused native targets). Series30 contributes 300 split targets and the effect alone contributes the remaining 12 token-refiner groups; these were retained, not excluded from scoring.

| Pair | Additive max relative export error | NP max relative export error | CT max relative export error |
| --- | ---: | ---: | ---: |
| Series30/Cinema | 1.372e-7 | 2.249e-7 | 5.909e-5 |
| Sully/Combat | 9.333e-8 | 5.353e-7 | 7.986e-5 |
| Sully/Cinema | 9.070e-8 | 2.248e-7 | 5.184e-5 |
| Series30/Combat | 1.404e-7 | 4.286e-7 | 7.608e-5 |

The largest CT discrepancy is about **0.008%**, not bitwise equality. All pass the provisional 0.05% numerical diagnostic gate documented in the protocol; that gate was selected after the first NP audit and before subsequent audits or any merge renders. Export error remains small relative to the intended method change (largest ratio 6.193e-5). None of this measures identity preservation or audiovisual preference.

The twelve exports total **9,248,478,208 bytes**. Additive/NP files are about 608–620 MB and CT files about 1.069–1.098 GB. Measured merge/export intervals total 164.800 seconds, with individual runs taking 10.558–17.591 seconds; import, input hashing and numerical auditing are outside those intervals. All report peak Torch CUDA allocation 2,476,961,792 bytes (~2.31 GiB). These are current measurements, not a before/after optimization claim.

Two orchestration details were resolved without changing merge math:

- The initial additive audit stopped because it read only `merge_mode=weighted_average`. Existing exports also record `merge_optimization_mode=additive`, which forces weighted sum; the text report correctly states this override. The checker now respects both fields and records the original metadata mode. A regression test covers the distinction. The same existing export subsequently passed; it was neither recomputed nor overwritten.
- The user authorized stopping the unused local ComfyUI session if necessary. A single idle `/free` request sufficed: free VRAM rose from 750 to 27,463 MiB and available RAM from 4,824 to 36,931 MiB. The generic request helper complained about `/free`'s empty response body after the operation; state checks confirmed it had succeeded, so the request was not repeated. The `8189` session remains running and idle; Resolve and the remote render server were untouched.

#### TIES memory constraint: investigate before running

The Series30-covered native matrices contain **19,267,584,000 elements**. Dense FP32 updates alone require **71.777 GiB** (35.889 GiB at two bytes per element), excluding working buffers, other processes and export copies. The dense path promotes contributor storage dtypes, so Series30's FP32 factors select FP32 storage on shared targets. Smart compression does not declare TIES low-rank-safe. Running this all-in-memory export against roughly 36 GiB available RAM is therefore not a sound next step.

This is a shape-derived storage requirement and source-code observation, **not a measured TIES OOM or a general claim that every TIES merge needs that memory**. No full-sized TIES run was attempted. Do not quietly label a rank-64 approximation as faithful TIES. A memory-safe execution/export path and its fidelity check remain required for that planned arm. The research runner currently has a GPU headroom guard, not a complete CPU/export-memory estimator; its new TIES CLI option is not evidence that this full-sized case is safe.

Validation after the new native checker, runner options and protocol checks: **709 passed, 3 skipped, 69 subtests passed** (18.59 seconds). Production optimizer and experimental code remain at the committed F07 identities, and the old AV2 and character-qualification plans are unchanged. The goal remains open: these are numerical exports, not improved ranking weights, new perceptual evidence, a completed nine-arm character benchmark or a shipping-quality verdict. No commit, push, installed-node update or version bump was performed.

### Memory-mapped TIES export: full-sized feasibility checkpoint

The preceding memory constraint is now addressed by a **research-only storage path**, not rank truncation or a new merge formula. [Safetensors documents zero-copy loading and partial tensor access](https://huggingface.co/docs/safetensors/index); local ComfyUI additionally has an Aimdo-backed file loader that retains mmap/file-slice references. Those capabilities motivated this implementation, but the measured results below—not the format documentation—establish feasibility for these specific exports.

`scripts/h3_mapped_export.py` reserves a complete native safetensors layout, receives each ordinary merged patch **after normal storage rounding**, and writes it directly into mapped file regions. Completed regions are flushed and their clean mapped pages released. Q/K/V rows are assembled directly into the final native matrix; unique-source targets retain their native factors. The writer refuses unsupported layouts, non-finite values, incomplete/overlapping coverage and existing destinations. It publishes only after coverage and real-reader validation, using a no-overwrite hard link; the temporary link is removed, not the exported data. A failed run retains an unpublished `.partial` for inspection.

The only new production-code change is `LoRAOptimizer._new_patch_store()`, an overridable private factory used for the two Pass-2 containers. It returns fresh plain dictionaries by default. The offline runner's subclass supplies the mapped container and disables model application and QKV refusion; normal UI inputs, merging, ranking and disconnected experimental defaults are unchanged. The current optimizer SHA-256 is `d1a9f9657580229c30aa4a7e43d700a2308c800f4afe0ebee81a199ebae00360`; the experimental implementation retains its F07 identity. Previously frozen plans and the twelve additive/NP/CT records have not been rewritten to pretend they used this new source identity.

#### Fidelity and first full-sized result

`tests/integration/h3_mapped_roundtrip.py` compares the normal SaveMergedLoRA path with mapped serialization using real ComfyUI and safetensors. Both FP32 and BF16 fixtures pass exact dtype/value equality and native patch-application equality. Coverage includes shared dense QKV, mixed flattened Musubi/native A/B keys, unique-source QKV factors and signed alpha/strength. This is a miniature CPU test, not H3 image-quality evidence.

The first full-sized run, `idm-series30-cinema-ties-01-mapped`, used strengths 0.8/0.8, normal smart compression, TIES density 0.5 and frequency sign election. It produced **77,082,394,664 bytes (71.789 GiB)** in **172.086 seconds**, with **3,973,509,120 bytes (3.701 GiB) peak process RSS** and **5,880,553,984 bytes (5.477 GiB) peak Torch CUDA allocation**. These are measured merge/export values; imports and input hashing precede the timer. Process RSS is not total system memory: the OS also uses reclaimable file cache. There is no measured all-in-memory baseline or speedup claim.

The new `scripts/h3_dense_export_check.py` reconstructs raw native source deltas independently of optimizer key normalization and uses the pinned ordinary TIES formula as its numerical reference. It applies exported patches through real ComfyUI, opening only one component at a time. All **312 components / 208 native targets** are finite and completely covered, including the 12 effect-only components absent from Series30. All **300 shared dense components match exactly**. The remaining native-factor reconstruction differences are at most **7.200e-8 relative error**. Shared FP32 inputs incur no normal storage rounding in this case. This validates serialization/loading, not an independent TIES implementation or perceptual benefit.

That audit took 147.170 seconds including import/input checks and the complete 77 GB SHA-256 pass; peak process RSS was 2.385 GiB. The export hash is `898e8b1fee5324a3dd6898b87ebec84eabf0a1c698e72a4cbe88febedae21d23`. All per-component results and the source/manifest/checker identities are preserved in the run's `dense_export_check.json`.

#### Actual full-file loader, separately tested

`tests/integration/h3_full_mapped_loader.py` loads the complete first export through the same local `comfy.utils.load_safetensors` Aimdo branch used by the existing server. All **224 tensors** retain mmap and file-slice references, and native mapping retains all **208 full-sized targets** with correct shapes. The measured loader/mapping interval was 0.020 seconds and process peak RSS 755,142,656 bytes (~0.703 GiB); this measures lazy mapping, **not reading all 77 GB**, which the prior numerical audit did. The probe checks stable inode/size/mtime and links the prior audit's payload hash without falsely claiming a second full-file rehash.

The first isolated CPU probe stopped because Aimdo's native library had not been initialized. Loading its native symbols before Comfy imports fixed the probe; no GPU device initialization, ComfyUI restart or export regeneration was required. This is a probe setup failure, not a failed H3 render. A full INT8 forward pass and its render-time memory/I/O cost are still untested for this dense export.

Validation: **716 passed, 3 skipped, 69 subtests passed** (20.47 seconds), including new coverage/no-overwrite/dtype/unique-factor and fresh-default-container regressions. The existing comprehensive real-Comfy round-trip integration also passes. The installed node pack remains untouched. No version bump, commit or push was performed. The second full-sized BF16 Sully/Combat run is in progress at this checkpoint; its measurements and storage-rounding audit must be recorded separately before qualification.

#### Second full-sized result: preserve BF16 behavior, report its rounding

`idm-sully-combat-ties-01-mapped` subsequently completed with strengths 1/0.8 and the same TIES density/sign policy. Its **80,153,280,520-byte (74.649 GiB)** export took **207.087 seconds**, with peak process RSS **3,676,205,056 bytes (3.424 GiB)** and the same 5.477 GiB peak Torch CUDA allocation. All 312 components are shared, so this case retains no unique-source factors.

The complete native-loader audit is finite and covers all 312 components / 208 targets. Every component matches the **ordinary storage-rounded result exactly: zero additional export error**. Normal BF16 rounding itself differs from the FP32 TIES calculation by up to **0.001665434 relative norm error (~0.167%)**. The research writer deliberately preserves that existing behavior; it must not be described as an FP32-precision TIES calculation merely because the saved payload is FP32. This is neither evidence of a visible defect nor a measured perceptual tolerance. The protocol's provisional 0.05% gate was explicitly for additive/NP/CT, not a retroactive requirement to silently change stable TIES rounding.

This second audit took 321.419 seconds including all checks and full-file hashing; peak process RSS was 2,414,854,144 bytes (~2.249 GiB). Disk caching and concurrent CPU-only tests/probes were not controlled, so audit timings are not a dtype-performance comparison. The export SHA-256 is `289f13fb277912637e3305c9d5c5a983e62c73d22f91db546b7153e0d3467828`.

The two new exports total **157,235,675,184 bytes**. They make faithful full-target TIES feasible on this host but remain far larger than the ~0.6–1.1 GB factor exports. A same-filesystem, verified hard link exposes only the first TIES export under `h3-autotuner-study-20260908/series30-cinema-ties-01-mapped.safetensors` in the existing loader root; it does not duplicate 77 GB, replace another file, or update installed code.

A separate technical forward-pass smoke test is prepared using the already-exposed cup prompt and seed 2026090801, still 640×384, 124 frames, 20 steps, INT8 diffusion, FP4 text encoder and Turbo off. It is **not a character identity comparison** and has no matched merge-quality controls. The smoke runner persists submission intent before POST, stops for unrelated queued work and refuses ambiguous retries. The character matrix's new prompts/seeds and old held-out AV2 outputs remain untouched. Generation has not yet been submitted at this checkpoint.

#### Full INT8/FP4 generation smoke: successful, expensive startup

The subsequent smoke test completed on the **existing local 8189 server**, prompt ID `2546da18-65a4-43fa-b376-a8dd3c22166a`. The server attached all 208 native patches to the INT8 H3 model. The [technical audit](data/2026-09-08-h3-dense-smoke-technical.json) verifies one successful execution, exact prepared/executed graph equality, complete decoding of **124 frames at 640×384 / 24 fps**, and **32 kHz stereo audio** with no decoded samples reaching absolute amplitude 1. No continuous playback, listening, identity rating or preference judgment was performed. The old cup prompt describes a man and lacks Series30's trigger; this is strictly a compatibility/feasibility probe.

The [original smoke plan](data/2026-09-08-h3-dense-smoke-plan.json) remains preserved. Its initial preflight stopped **before any network call or submission intent** because the unchanged preparation CLI records LoRA strength as text. A [pre-submission amendment](data/2026-09-08-h3-dense-smoke-amendment.json) pins the runner fix: normalize numeric strength on input, without changing the original manifest, graph, prompt or seed. Regression coverage now includes both string/numeric strengths and timeout-before-receipt behavior. Exactly one generation was submitted.

History reports **181.680 seconds end-to-end execution**. The server log marks dynamic model preparation at 16:44:10.756 and sampling completion at 16:46:37.340; the sampler's own progress timer reports about **23 seconds for 20 steps**. Thus much of the elapsed interval precedes that timer. Dense loading/preparation and cold initialization are plausible contributors, but this is **not a controlled attribution or a merge-mode speed ratio**. No concurrent export/hash process was running during generation.

The server's lifetime resident-memory high-water mark increased from 36,305,156 KiB before sampling to **38,022,620 KiB (36.261 GiB)** during this test. One sample found 3,472 MiB system available RAM; availability recovered after the job. This is a separate measurement from the **3.701 GiB offline export peak**, and it rules out claiming that the mapped exporter makes this full generation path low-RAM. The video nevertheless completed without OOM, cancellation, restart or unrelated-job interruption. Output SHA-256: `10d71f0d69d359e0bf8dbd2b3cb10d4ed206d959fb3569a0f165d1db5899c712`.

The [compact two-run numerical record](data/2026-09-08-h3-character-ties-mapped-numerical.json), SHA-256 `20f143be5c4b907d640098f47330834947565983577e7f483c5001acfb775eec`, now preserves both full TIES results, independent full-file rehashes, audit identities, worst-component details, and the first export's full-loader probe. The original per-component files remain in their artifact directories. Validation at this checkpoint is **721 passed, 3 skipped, 69 subtests passed** (19.56 seconds); both real-Comfy integration tests pass, and `git diff --check` is clean.

This does **not** complete the nine-arm character experiment: the other crossed TIES exports, character SLERP exports, exact stable-tuner winners, frozen execution matrix, matched identity/effect evaluation and held-out evidence remain. The next stage can now include faithful TIES without substituting a rank-64 approximation, while accounting for its disk and startup cost. No ranking weights/defaults were changed, no installed node code was replaced, and nothing was committed, pushed or version-bumped.

After confirming the local queue was empty, the completed test's model cache was released with `/free`. Verification found **27,477 MiB free VRAM and 38,681 MiB available system RAM**; the 8189 server remains running with an empty queue. The remote render server was never touched. A final identity check links the preserved original plan, preflight amendment, executed runner hash, unchanged prepared graph/manifest, submission intent and successful technical audit exactly.

### Character SLERP and exact stable-tuner replay

The two initial diagonal pairs now also have ordinary global SLERP exports, stable-only tuner sweeps, and exact rank-one selector replays. The [six-run numerical record](data/2026-09-08-h3-character-stable-numerical.json) retains source/export/manifest/checker hashes, full selected configurations, actual per-prefix decisions and measured resource use. None of these tensor scores are image-quality labels.

| Pair / export | Bytes | Merge/export seconds | Maximum relative error against FP32 formula |
| --- | ---: | ---: | ---: |
| Series30/Cinema SLERP | 1,892,888,568 | 20.350 | 1.98205e-5 |
| Sully/Combat SLERP | 1,514,877,256 | 17.094 | 0.00244414 |
| Series30/Cinema stable winner | 876,981,472 | 14.737 | 1.45971e-6 |
| Sully/Combat stable winner | 647,834,760 | 10.575 | 0.00242287 |

All four audits cover **312 finite components / 208 native targets**, require complete source/export key consumption, and load through real ComfyUI independently of optimizer key normalization. The checker invokes the pinned ordinary stable formulas; it does not independently establish SLERP's algorithmic merits. Both ordinary SLERP and the SLERP portions of a per-prefix winner use normal rank-64 compression. The BF16 Sully cases retain their existing dense-storage rounding and compressed-factor rounding, whereas Series30 promotes shared inputs to FP32. Sully's maximum normal storage-rounding error is about **0.166%** and its total discrepancy from the FP32 reference about **0.244%**. Error after that intermediate rounding is reported separately; the maxima refer to different components and are not additive. These are implementation baselines, **not lossless FP32 or perceptually equivalent results**, and the additive/NP/CT-only 0.05% gate was not repurposed to conceal them.

The Series30/Cinema stable-only sweep measured its top three candidates in **18.264 seconds**, with peak Torch CUDA allocation **13,605,939,712 bytes** and process RSS **17,044,779,008 bytes**. The winner selects weighted sum for 92 components, SLERP for 55, and weighted average for 165. Auto-strength is disabled. Its tensor score is 0.671603 and energy ratio 1.263637; neither establishes superior character retention. The Sully/Combat sweep took **8.930 seconds**, peaking at **4,142,279,168 bytes CUDA / 2,880,000,000 bytes RSS**. Its winner selects 183 sums, five SLERPs and 124 averages; enabled auto-strength actually returned model scale **1.0**, keeping source strengths 1/0.8. Its tensor score is 0.510329 and energy ratio 1.612480. These different source/mode mixes invalidate a universal claim that full-target tuning always stays under 2 GiB.

Both sweeps kept experimental options disconnected, measured all targets, disabled SVD scoring and persistent/community/dataset feedback, and did not fit to render labels. The replay runner now captures the selector's actual auto-strength result without altering it and requires exact equality between the replayed and selected per-prefix maps before exporting. Native audits apply that captured scale and the individual Q/K/V choices rather than treating every layer as a single global mode.

Reproducibility limitation: the two global SLERP files were created **before** this runner gained an explicit Torch compression seed. Their complete file hashes identify the actual evaluated artifacts, but bitwise regeneration from an unspecified initial RNG state is not promised. The subsequent two sweeps and two winner replays explicitly use research merge seed **2026090800**, distinct from video seeds, and record runner SHA-256 `ca87348b8787b523ff3b70a46454d5222a21e3f0c0ac1dcafbc72b8438d80f7f`. The old records were not rewritten to pretend they used this addition.

The new `scripts/h3_character_benchmark.py` expands the unchanged protocol into **72 execution cases for two pairs**, retaining all nine arms and all four predeclared seeds. Its first requested batch is only **18 calibration cases at seed 2026090831**; the remaining two crossed pairs are still required by the 144-case protocol, not silently dropped. Preparation rehashes every raw and merged adapter, validates numerical evidence and freezes the prompt/matrix/harness. Subsequent file guards check inode, size and nanosecond mtime instead of claiming to rehash 157 GB before each short clip. Installation uses exclusive, verified same-disk hard links in the existing temporary loader root. Execution preserves pre-POST intent, watches the same prompt after uncertain interruption, and stops for other queued work. It does not replace installed node code or edit the canvas.

Validation at this pre-render checkpoint: **739 passed, 3 skipped, 69 subtests passed** (19.37 seconds), with `git diff --check` clean. The 18 new scheduling/identity/rounding checks and existing tests cover matched conditions, disjoint hold-out, changed graphs/scales/exports, no-overwrite installation, explicit batch scope, queue protection and duplicate-submission prevention. No ranking changes, commit, push or version bump have occurred. The execution plan's full-file verification is in progress; no character-comparison generation has yet been submitted at this checkpoint.

#### Execution freeze and preliminary identity measurement

Full-file verification subsequently completed. The [execution plan](data/2026-09-08-h3-character-benchmark-plan.json) has SHA-256 `1f179b668839a4379a1743fd3c8f507b0202a254ccb5f47a3047f25a7a33ed40`. All twelve required merged adapters are exposed through same-inode hard links; the existing Series30 TIES alias was verified and reused. No duplicate 157 GB payload was created. The empty local queue was checked and the **18 calibration seed-31 cases** started serially; no seed-32 or held-out case was submitted.

In parallel, an offline CPU-only diagnostic was frozen and run on the **older sixteen qualification clips**, not the new merge outputs. The [probe plan](data/2026-09-08-h3-identity-probe-plan.json), SHA-256 `91403be30759ae3f446245e454208f323f806e70b455335d937116105b9ea30d`, fixes six source-frame indices (0, 24, 48, 72, 96, 120), a single first-frame creator reference per identity, and manually declared reference-head rectangles. Those rectangles were selected from already exposed creator-reference grids before any similarity scores were computed. They are approximate regions, not semantic segmentation. All 96 sampled output frames are retained; no visibility failures are silently filtered.

The diagnostic takes the mean best output-patch cosine for each selected reference-head patch. This is an **adaptation inspired by [MaSC's foreground-reference matching](https://arxiv.org/html/2605.22469v1), not MaSC itself**: it uses the cached [SigLIP2 base/512 backbone](https://huggingface.co/google/siglip2-base-patch16-512), not the paper's SO400M-NaFlex encoder, and rectangular head regions rather than supplied foreground masks. The [official processor configuration](https://huggingface.co/google/siglip2-base-patch16-512/blob/main/preprocessor_config.json) is explicit: 512×512 bilinear resizing, 1/255 rescaling and 0.5 mean/std normalization. This matters because the installed library's default processor is 224 pixels with a different resize filter. Whole-image pooled cosine is retained as a diagnostic comparator, not an identity oracle. No text model or background-prompt score is used to judge action.

The [results](data/2026-09-08-h3-identity-probe-results.json) preserve every sampled score and exact model/runner hashes. The cached full checkpoint hash is `fe0e601c625e69eed8e73500d39e9b6164403fe03db8048e87913c3cefbbb3fe`; the vision tower loaded completely, with only unused text/logit keys excluded. Scoring the 96 output frames took **106.982 seconds on CPU, four Torch threads**, excluding model loading/reference feature extraction. It ran concurrently with generation, so this is not a controlled throughput benchmark.

| Qualification prompt | Seed | Base median head-patch similarity | Character-only median |
| --- | ---: | ---: | ---: |
| Series30 source | 2026090821 | 0.756869 | 0.859614 |
| Series30 source | 2026090822 | 0.808304 | 0.882484 |
| Series30 turn | 2026090821 | 0.834189 | 0.848748 |
| Series30 turn | 2026090822 | 0.844724 | 0.859907 |
| Sully source | 2026090821 | 0.651597 | 0.717817 |
| Sully source | 2026090822 | 0.657475 | 0.719214 |
| Sully turn | 2026090821 | 0.595552 | 0.679234 |
| Sully turn | 2026090822 | 0.630950 | 0.698762 |

The character-only median is higher in **8/8 matched controls**. The human head-turn gains are only about 0.015, versus 0.062–0.084 for Sully. Whole-image cosine is higher in 7/8 controls and essentially tied/slightly reversed in Series30 turn seed 22 (0.904304 base versus 0.904288 character). This is a small calibration sanity check, **not eight independent subjects, a human-correlation study, an identity probability, or proof that head-patch matching is generally superior**. Hair, species, pose and lighting can still influence it. It does not determine whether a subject is actually visible, whether a punch is correct, or whether sound is synchronized. No threshold, ranking weight, learned judge or external-scoring hook was changed; applying this diagnostic to merge comparisons still requires separate effect/visibility review.

### First complete nine-arm character block: 18 videos, not a shipping verdict

All **18 calibration seed-2026090831 videos completed successfully**. The [whole-clip record](data/2026-09-08-h3-character-benchmark-seed31.json), SHA-256 `1680aac8c1c1222af8db727de9641ef0f73980ac55907327d731e452766c64fc`, confirms exact prepared/executed graph equality and full decoding of **2,232 video frames**: each clip is 640×384, 124 frames, 24 fps, with 32 kHz stereo audio and zero decoded samples at or beyond absolute amplitude 1. Successful execution and unclipped audio are not audiovisual preference or synchronization labels.

Recorded execution intervals total **799.066 seconds**, ranging from 28.869 to 105.807 seconds. The Series30 and Sully TIES cases took 105.807 and 98.711 seconds respectively. Cache state, conditioning reuse, cold model preparation and concurrent CPU-only diagnostics were not controlled, so these observations must not be reported as a merge-mode speed ranking. Every generation used the existing local 8189 server; no other job was cancelled, no session restarted, and the remote render server remained untouched.

After the batch, a fresh empty-queue check permitted one `/free` cache release (HTTP 200). Verification found **27,480 MiB free GPU memory and 38,393 MiB available system RAM**. ComfyUI remains running and idle. The completed media and merge files were retained.

#### Review fidelity and limited visual observations

`scripts/h3_character_review.py` builds a self-contained review at `.h3-study-artifacts/20260908/character-merge-review-seed31/review.html`, review ID **`a32c18bdfe6661c84b5d`**. It provides each pair's **own exact prompt**, an independently playable creator identity reference, and the matched local character-only anchor. Methods are hidden, but a character-only candidate can be recognized as a duplicate of the visible anchor; this is not a fully double-blind trial. Identity and intended effect are separate dimensions alongside temporal behavior, audio, synchronization and overall preference. Ratings remain empty. The script's character-label validator intentionally rejects the old AV-only score schema.

All **18 blinded candidate copies** pass independent full decoded-video and decoded-audio equality against their originals, with workflow-identifying metadata removed. No loudness normalization, interpolation, timing shift or audio replacement was used. Verification hash: `509862a6f3356a3342eeee218ba10a4ae595dba619919f29eb3486857ceed6f1`. Creator anchors were separately hash-checked and stream-copied; the 18-candidate equality report does not claim a new independent decode audit of those two anchors.

The assistant inspected 21 chronological 4-fps samples per candidate, **378 sampled images total**, under random C identifiers and public pair labels. The [blind observation record](data/2026-09-08-h3-character-seed31-blind-observations.json), SHA-256 `3c465168cb467a482d0bc28ca0f2ca1b9b4d7af514cf675773f8cb37c3824a2b`, was saved **before** reading the private method key. It reports visibility limits and observations for every candidate, not just favorable examples. This is assistant visual evidence from sparse frames: **no continuous playback, listening, human rating or exact contact timestamp was claimed**. The grid's unused black cells are padding, not black frames in the videos.

After preserving that record, unblinding showed:

- **Series30/Cinema:** C05 (base) and C10 (Cinema-only) had the clearest depiction/hair/collar differences from the character-bearing cluster. The character-only anchor C07, additive C11, NP C08, SLERP C01, TIES C04 and stable winner C16 remained close at this inspection scale, with visible head turn/return. CT C12 had some additional depiction differences, but the sheet did not justify a reliable identity preference. Lamp/background/blouse differences were not treated as identity errors or automatic style wins. There is no established winner among these merges.
- **Sully/Combat:** C17 (base) and C18 (Combat-only) had a distinct yellow-bearded, longer-horned depiction, unlike the green/purple creator anchor. Every character-bearing arm preserved the coarse intended features. The character-only control C15 already omitted the requested vest and used full-body framing; additive C14 also used a wide full-body shot. Thus those faults cannot automatically be attributed to merge-induced identity loss.
- **CT's specific follow-up signal:** C02 (CT) kept the green/purple reference-like depiction **and the requested navy vest**, while the other character-bearing arms omitted the vest in these samples. This is a prompt-compliance observation at one seed, not proof of a general CT advantage. SLERP C03, TIES C06, NP C13 and the stable winner C09 also showed repeated punch sequences apparently exceeding the requested two; several clips did not clearly achieve the prescribed lowered-glove ending. Exact action count/contact and sound require full-frame/event review. No arm was assigned an audiovisual score.

This block exposes a useful failure mode for future optimization: **identity alone can reward a result that loses wardrobe, framing or exact action compliance**, and those biases can already exist in the character-only adapter. A valid merge comparison must retain that control and the effect-only control. The CPU similarity probe has not yet been applied to these new merge outputs, and its calibration success does not supersede these separate checks.

Validation at this checkpoint: **749 passed, 3 skipped, 69 subtests passed** (21.61 seconds). Both real-Comfy CPU integration suites also pass, including exact ordinary-versus-mapped FP32/BF16 TIES payload/application equality. The only production change since F07 remains the private fresh-dictionary storage factory; the new execution, evaluation and review work is research tooling. No installed node replacement, ranker/default change, version bump, commit or push occurred.

The goal remains open. Next, test whether this seed's identity/wardrobe/action observations reproduce at the frozen second calibration seed, apply only explicitly qualified diagnostics, and inspect full-frame action/waveform timing where needed. Freeze any evaluation decisions before opening held-out outputs. The two other crossed character/effect pairs and the remaining seeds still belong to the full protocol; the 18 completed cases must not be presented as all 144. R1 remains one person's earlier AV perception, not a consensus or a label for any of these new character clips.

### Second character calibration seed: 36 videos complete, mixed replication

All 18 seed-2026090832 jobs completed successfully on the existing local 8189 instance, with no retries, cancellations or restarts. The [second whole-clip audit](data/2026-09-08-h3-character-benchmark-seed32.json), SHA-256 `e0a8399404095a1ec9bd7dc728a31b47c21c829da43c4cc64932c712e595f44b`, records another **2,232 decoded frames**, exact prepared/executed graphs, and the unchanged 640×384 / 124-frame / 24-fps INT8-convrot, FP4-TE profile. Recorded execution intervals total **815.014 seconds**, ranging from 29.522 to 121.659 seconds. These remain uncontrolled cache/loading timings, not a method-speed comparison.

One technical audio flag must not be hidden: Sully's **Combat-only** control has one decoded stereo-channel sample above absolute amplitude 1, peak **1.003684**, fraction **0.00000302418** across 330,668 scalar channel samples. The other 17 clips have no such samples. This is a decoded full-scale exceedance, not established audible clipping, its cause, or a merge defect. No normalization or repair was applied. All clips remain in the comparison.

After the completed batch, an empty-queue check permitted one successful cache release. Verification found **27,488 MiB GPU memory free, 0% GPU utilization and 38,838 MiB available RAM**. The local server remains running; the remote 192.168.1.12 server was not contacted. No model files were removed or duplicated.

#### Method-blinded frame observations before scores

The second self-contained review is `.h3-study-artifacts/20260908/character-merge-review-seed32/review.html`, ID **`b96d82d37dabd870d982`**. All 18 candidate copies pass full decoded-video/audio equality and metadata stripping; verification SHA-256 `fffa3bb78493eff062022892799c91f313797130b5f7a27538116b3f46754597`. As before, the creator anchors were separately hash-checked/stream-copied, not counted as two additional independently decoded candidate audits.

The assistant viewed all 18 chronological 4-fps sheets (**378 samples**) and saved [the observation record](data/2026-09-08-h3-character-seed32-blind-observations.json), SHA-256 `878158996a2d2ef11fe60293b8a5dd8cc8e278cf52586096405f653fbc8e0857`, **before reading this seed's private method key or similarity scores**. Prior seed-31 results were known and some repeated depictions could be recognizable; this is not an independent, fully double-blind trial. No full-motion review, listening, human ratings or preference scores were claimed. Public manifest SHA-256 `b4eb06a7f57b2a5d00818c025e3310d407c768ea2cc6c753f96788a3f3ba9a08`; private key SHA-256 `38ed897ad23ba3478ee7f288e07d158f4b99e8df8a4ff59ad9b3f2be1f57304b`.

Unblinding the preserved observations shows:

- **Sully's coarse character features reproduce:** all seven character-bearing arms retain the green/purple, short-horned, broad-muzzled depiction. Base C13 and Combat-only C04 instead have the distinct long yellow-green beard, purple spiky head and longer horns. Bag/arm occlusions are explicitly retained as visibility limitations.
- **CT's vest observation reproduces, but is no longer unique:** CT C15, SLERP C07 and TIES C17 have the navy vest. Character-only C08, additive C03, NP C16 and the stable winner C10 omit it. No character-bearing arm used the first seed's wide full-body framing at this inspection scale.
- **CT's ending does not reproduce:** C15's gloves remain raised near the end. SLERP C07, TIES C17, NP C16 and additive C03 show lowering in the late samples; the stable winner C10 lowers very late. Repeated extensions beyond the requested two remain visible across the character-bearing candidates. Sparse sheets do not establish exact counts or contact timing.
- **Series30 remains hard to rank visually:** TIES C01, NP C02, SLERP C05, additive C09 and stable winner C12 are close at this scale. Character-only C06 returns to frontal view earlier and has more frontal samples. CT C11 changes pose/framing; effect-only C14 and base C18 have more distinct depiction/lighting. None of these differences alone establishes a style or identity winner.

The stable winner therefore remains a **numerically selected baseline**, not an established perceptual optimum. Two seeds support investigating wardrobe/action tradeoffs, not an automatic CT bonus. These 36 clips cover only two character/effect pairs and calibration seeds; **108 of the full 144-case protocol remain unexecuted**.

### Full-frame action versus waveform: resolve occlusion before counting

`scripts/h3_av_sync.py` now supports a separate `--observations ... --review-key ...` path for already-exposed character calibration. It checks the frozen plan, public review manifest, observation identity, private key, seed, executed graph and original media hash before extraction. The original `--ratings` path remains intact. New outputs explicitly say `assistant_frame_observations`, with `human_ratings_supplied=false`; they do not fabricate a ratings digest or classify sound. Nine added regression cases cover provenance and rejection gates. Inspector SHA-256: `0db0ed94c6e97709ae774ade4944b768cc60586e3ef00117301f0d8adba5f0af`.

The assistant inspected **all 496 frames** across four selected seed-31 Sully clips, with unchanged stereo linear min/max and RMS envelopes in 5-ms bins. Selection was exploratory and **after unblinding**, motivated by the earlier CT wardrobe signal. This is not all-method coverage or real-time playback/listening. [The full-frame observation record](data/2026-09-08-h3-character-frame-waveform-seed31.json), SHA-256 `30f74a180eda203b3a49142236f69567badf5e41a32a0e6a85eba92566037118`, links all source/sync hashes, frame windows and anonymous energy peaks.

| Seed-31 arm | Visible action finding | Wardrobe / ending |
| --- | --- | --- |
| Character-only | Three distinct large punch/recovery cycles, not the requested two | Vest absent; remains at guard |
| Additive | Four clear large near-arm cycles plus multiple far-arm extensions; exceeds two | Vest absent; lowering finishes only in final frames |
| Stable winner | Five clear large near-arm cycles, with additional far-arm movement | Vest absent; no settled lowered-glove finish |
| CT | Two clear near-arm cycles **plus a third partly occluded far-arm punch** | Vest present; gloves lowered and relatively still for approximately the final second |

CT's middle event initially looked ambiguous in the thumbnail sheets. A separate original-resolution 16-frame grid (frames 40–55, SHA-256 `a54dd3005428189c6a8c40649e4dec73206028d1fb384fa06ab557eae83a8d22`) shows the far glove rising, extending behind the bag and retracting. That visual evidence supports a third attempted punch; exact first contact remains obscured. **The waveform was not used to invent the missing visual event.** The two clear near-arm contact windows are frames 13–15 and 72–74. CT is therefore not an exact-two-punch success even in its favorable wardrobe/ending seed.

Character-only has three separated broadband maxima around 0.7225, 2.1475 and 3.5975 seconds. CT has six maxima in three broad groups; additive and the stable winner each have twelve detector maxima spread across their repeated action sequences. Peak count is plainly not punch count. Peak-minus-visible-contact also is not a perceptual synchronization score: multiple maxima, occlusion, sound attack/reverberation and unknown sound identity preclude that interpretation. No audio offsets were corrected. The control already overproduces punches, while additive/winner can add further repetition; a valid optimization objective must distinguish both effects.

### Frozen head-region diagnostic applied to all 36 calibration clips

The new research-only `scripts/h3_character_similarity.py` applies the previously calibrated CPU probe **unchanged**, rather than fitting a score to favorable merge outputs. The [frozen recipe](data/2026-09-08-h3-character-similarity-recipe.json), SHA-256 `5b8b8ffc6e2cf390e33feba12ee7a6b56cb40f11acc31981886ced15abfeea81`, was saved while seed 32 was rendering, before any seed-32 visual inspection. It inherits the exact cached SigLIP2 model/config hashes, official 512-pixel processor, reference-head rectangles and six frame indices (0, 24, 48, 72, 96, 120) from the earlier qualification probe. It pins its own implementation and helpers. No masks, sampling, model or thresholds were changed after reading results.

Both [seed-31 scores](data/2026-09-08-h3-character-similarity-seed31.json) and [seed-32 scores](data/2026-09-08-h3-character-similarity-seed32.json) preserve every sampled frame, whole-image comparator, clip hash and raw paired differences. Their SHA-256 identities are respectively `3d56fcc733c7cf5389bc4fb1913d8ee3a94294848095e7a681b0dfc2b24e5b69` and `f1fa085ea8ff063c41c6e0b596ce1d66a0bb8f7f8f8d60a22795500c3cd3c12b`. All original outputs first passed complete-media audits and exact frozen-graph checks. The vision tower loads fully on CPU, four Torch threads, FP32; unused text/logit keys are expected. Scoring 108 output frames per seed took **117.453 and 117.436 seconds**, excluding loading/reference extraction. No GPU model or external service was used. Seven new tests cover complete nine-arm comparisons, signed differences, invalid/duplicate/missing controls and rejection of held-out seeds before model/media access.

Raw median reference-head patch similarity (cosine units, **not a preservation percentage or human-validated identity score**):

| Arm | Series30 seed 31 | Series30 seed 32 | Sully seed 31 | Sully seed 32 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.836235 | 0.858176 | 0.611753 | 0.609478 |
| Character-only | 0.871300 | 0.873722 | 0.632996 | 0.628196 |
| Effect-only | 0.845375 | 0.827902 | 0.606949 | 0.616479 |
| Additive | 0.874913 | 0.854413 | 0.634408 | 0.637758 |
| SLERP | 0.877685 | 0.852402 | 0.632729 | 0.624240 |
| TIES | 0.874955 | 0.860006 | 0.639897 | 0.627540 |
| Stable winner | 0.869669 | 0.857218 | 0.629030 | 0.626190 |
| NP | 0.870940 | 0.856019 | 0.633307 | 0.630081 |
| CT | 0.867084 | 0.844406 | 0.621842 | 0.633755 |

The character-only diagnostic exceeds base in all four new matched cells, consistent with the earlier qualification check. But the merge comparisons do **not** show a universal experimental benefit:

- CT is below additive in all four cells, by about **0.0040–0.0126 raw cosine units**. Its Sully difference from character-only changes sign between seeds (−0.01115, +0.00556), even though the vest is present in both.
- NP and the stable winner are each below additive in three cells and above in one. SLERP's sign also varies. TIES exceeds additive in three cells, but one gap is only 0.000042 and the Sully seed-32 difference reverses to −0.010218. These tiny samples and unqualified perceptual meaning do not justify selecting a global winner or paying dense TIES's 157 GB two-file storage cost on this basis.
- **Series30 base seed 32 scores above additive** despite the visually distinct depiction. Character-only also has more frontal frames at this seed. The coarse metric may reward pose, face visibility, hair or lighting; without independent identity labels, neither the numerical order nor the assistant's coarse depiction grouping should be treated as ground truth.

These are four matched cells from **two characters**, not four independent identities or a significance test. No fitted thresholds, normalized retention ratios, method-average winner, evaluator feedback or learned ranking update were introduced. The diagnostic remains useful for recording deviations and testing controls, but is **not qualified as an autonomous merge judge**. Clothing, framing, exact action count, ending and audio need separate evidence; retaining a vest cannot erase an identity deficit, and a high similarity score cannot erase extra punches.

Validation at this checkpoint: **765 passed, 3 skipped, 69 subtests passed** (20.80 seconds). Both real-Comfy CPU integration suites were rerun successfully, including exact ordinary/mapped FP32 and BF16 payload/application equality and native NP/CT round trips. Frozen execution/recipe helpers and installed same-inode model links revalidate unchanged; all four waveform source records and all eighteen second-seed observation identities also match. No character held-out histories exist yet. The only production change since F07 remains the fresh-dictionary storage factory (`d1a9f9657580229c30aa4a7e43d700a2308c800f4afe0ebee81a199ebae00360`); this checkpoint adds research evaluation/diagnostics, not new UI behavior or scoring weights. The H3/ComfyUI skills guided local profile and queue/cache checks without changing the frozen prompts. No installed node update, commit, push or version bump occurred.

**Shipping position remains provisional:** retain the tested correctness/efficiency fixes and keep NP/CT opt-in; there is no evidence-backed automatic quality boost to ship from these calibration results. Next, finish the crossed character/effect pairs and freeze a separate held-out evaluation policy before exposing their remaining prompts/seeds. R1 is still one person's earlier AV perception, not a label for these clips. The larger goal remains active; neither the 36/144 render count nor these CPU scores complete it.

### Crossed pairings: all export arms numerically qualified

Series30/Combat and Sully/Cinema now have the remaining global TIES, global SLERP and exact stable-tuner winner exports. Their already-qualified additive, NP and CT files are reused without regeneration. Raw adapters, role ordering, strengths (0.8/0.8 and 1.0/0.8), formulas and production code are unchanged. New merges record RNG seed 2026090800; this does not retroactively seed older SLERP runs.

The [crossed TIES record](data/2026-09-08-h3-character-crossed-ties-numerical.json), SHA-256 `b3abf0f5bb65ad39d088e9e9bff60bbe34bb4e91f6070173c912f73ca9ec09e6`, covers both complete mapped exports and native-reader checks. Each check consumes all source/export keys and all 312 split components across 208 native targets; all values are finite. Series30/Combat's 77,082,394,664-byte file takes 157.803 seconds to export, peaks at 3.696 GiB process RSS / 5.477 GiB allocated CUDA, and preserves all 300 dense stored components exactly; its 12 unique low-rank components differ by at most 7.75e-8 relative error. Sully/Cinema's 80,153,280,520-byte file takes 200.959 seconds, peaks at 3.418 GiB RSS / 5.477 GiB CUDA, and preserves all 312 normal stored components exactly. Its ordinary BF16 storage rounding remains separately visible: at most 0.0016635 relative to the FP32 formula. Exact normal storage is not lossless FP32 mathematics.

The [crossed stable record](data/2026-09-08-h3-character-crossed-stable-numerical.json), SHA-256 `66010f9f2a7704d30ba51ae7d59c8516fb53dfdd5ada2ec9b97a06d37c1babc0`, captures both full-target top-three sweeps, the exact rank-one configurations/decision maps and four native export audits:

| Pair / arm | Export bytes | Merge/export seconds | Maximum relative FP32-reference discrepancy |
| --- | ---: | ---: | ---: |
| Series30/Combat SLERP | 1,892,888,560 | 23.032 | 1.0838e-5 |
| Series30/Combat winner | 694,398,120 | 14.333 | 2.6246e-6 |
| Sully/Cinema SLERP | 1,514,877,264 | 18.938 | 0.0024338 |
| Sully/Cinema winner | 847,392,024 | 13.369 | 0.0024255 |

All four audits cover 312/208 components/targets, finite values and every key. The Sully SLERP branches retain ordinary BF16 rounding/compression, rather than adopting a new precision policy. The stable sweeps take 16.445 and 15.723 seconds respectively. Series30 selects 173 sum / 11 SLERP / 128 average decisions with auto-strength enabled (actual scale 1); Sully selects 128 sum / 79 SLERP / 105 average decisions with auto-strength disabled. Their internal scores are 0.488964 and 0.656776, **not perceptual quality scores**. Full scoring disables subsampling but not all caching: the Series30 log's 202 patches plus six separately scored/reused single-adapter patches account for all 208 native targets. No omitted-target bug was found from that log.

Peak allocated CUDA in the Sully/Cinema stable sweep is **16.460 GiB**, versus 6.553 GiB for Series30/Combat. Thus the earlier approximately 1.73-GiB experimental tuning result must not be generalized to all stable candidate mixes. This is a documented performance limitation, not a controlled before/after comparison or an OOM. No concurrent GPU study process was used. CPU record hashing can overlap; recorded times are not standardized throughput trials. Four retained dense TIES files now occupy **314,471,350,368 bytes** in total; the two newest account for 157,235,675,184 bytes. Disk free after the stable exports is approximately 306 GiB. No files were deleted.

Two separate research extensions, `h3_crossed_character_benchmark.py` and `h3_crossed_character_similarity.py`, reuse private instances of the original hash-pinned engines. They extend pair selection without rewriting the old plans or helpers. The new execution scope is 72 disjoint jobs; with the unchanged parent it specifies all 144 protocol cases, not 144 completed renders. Crossed similarity keeps the same CPU model, reference rectangles, processor, frame samples and raw differences, and must be frozen before any crossed calibration history exists. It rejects held-out seeds; no quality labels or evaluator feedback are introduced. Eighteen added tests cover complete/disjoint matrices, control/graph/provenance guards, no overwrites, isolation and unchanged metric math. They also pass in a relocated temporary checkout with original-repository reads denied.

Full validation at this numerical checkpoint: **783 passed, 3 skipped, 69 subtests passed** (18.75 seconds). The original 36 character clips remain the only completed character-merge generations at this checkpoint; the newly exported crossed adapters have not yet been forward-rendered. Existing local ComfyUI remains running with an empty queue and 27,480 MiB free GPU memory at the pre-render check. The H3 and ComfyUI skills guided the unchanged low-resolution INT8-convrot / FP4-TE profile, queue handling and licensing caveat. No remote-server contact, installed node replacement, ranking/default change, commit, push or version bump occurred.

#### Crossed execution and diagnostic freeze

The [crossed execution plan](data/2026-09-08-h3-character-crossed-plan.json), SHA-256 `110d58a4adeab1dce4c3ef288bc3036c0a1b3ff79a50d862c49818103b9c40c2`, pins twelve fully hashed merged assets, four unchanged raw controls, all numerical records and both parent/extension engines. Twelve hard links were successfully added to the existing temporary model directory and verified as the same inodes; there are no duplicate weight copies or replaced files. The [crossed similarity recipe](data/2026-09-08-h3-character-crossed-similarity-recipe.json), SHA-256 `86ddcf934e3337cf47aafe83991efc87886110afd890d0bc2f10de3fce445643`, was saved before any crossed calibration history existed. It declares the already-exposed diagonal evidence and does not refit the metric. Extension implementation identities are `38851e383a4822e2462baaa7581506d414f1515005a2c7f1f81a49441d9db91c` (execution) and `0f81390da6e082b0349d4e87c22f92f87f82666919df32ddb81eb488a68a6fba` (similarity).

The serial runner was then started for the explicit calibration split, seed 2026090831, limit 18, on existing localhost 8189. Completion and media-audit counts must be updated from terminal receipts, not from this launch. No character held-out outputs have been opened or generated.

#### F08 — reproduced candidate-deduplication inefficiency, fix pending

While crossed renders run, [a separate diagnostic record](data/2026-09-08-h3-candidate-signature-observation.json) captures a repeated search-efficiency issue. In **all four character/effect sweeps**, ranks 2 and 3 (`no_slerp` and `basic`) have identical complete per-prefix decisions and recorded metrics. Inspection explains why they survive deduplication: the orthogonal `no_slerp` branch returns the unused sign-method placeholder `frequency`, whereas `basic` returns `total`. Plain weighted sum and weighted average do not use sign voting, but `_strategy_signature` and `_group_merge_cache_key` include that distinction.

A CPU-only probe calls the actual decision, cache-key and merge functions with tiny FP32 tensors and three synthetic prefix-statistic cases (balanced near-orthogonal, magnitude-imbalanced near-orthogonal, and slightly opposing). All three produce **different dedup/cache keys but bit-identical merged tensors**. The record pins implementation/fixture and real tuner-data hashes, supplies inputs and reproducing procedure, and distinguishes synthetic statistics from real measurements. No full-size rank-2/3 patch-hash comparison or controlled speedup has yet been measured. An initial one-off fixture import failed because the `tests` namespace did not resolve outside pytest; loading the exact fixture by file path succeeded, without modifying it.

Consequently, “top three” in these records means three configurations, **not three distinct effective merge policies**. This can waste a slot and cache reuse; it does not demonstrate a quality improvement from any replacement candidate. The current frozen winner and baseline exports stay unchanged. A safe next experiment is to canonicalize only mathematically inactive fields, initially behind explicit experimental enablement, with tests that keep real TIES voting/density, sparsification, refinement and strength distinctions. F08 is an open, reproduced finding, **not a shipped fix or a reason to change old defaults**.

### First crossed calibration batch: 54 of 144 character clips audited

All 18 crossed seed-2026090831 jobs completed successfully without retries, cancellations or a server restart. The [whole-clip audit](data/2026-09-08-h3-character-crossed-seed31.json), SHA-256 `16ed55e5d127ae2d7295f222f3cc8d80d11ed4c1400232f5b18fce90108f2862`, records another **2,232 decoded frames**, unchanged 640×384 / 124-frame / 24-fps video, 32-kHz stereo audio, and exact frozen/prepared/executed graphs. Both new dense TIES adapters have now run through actual INT8-convrot / FP4-TE generation, not just native-reader checks. All eighteen outputs remain in the comparison. Recorded execution intervals total **813.125 seconds**, minimum 28.988 and maximum 110.968 seconds; loading/cache state and overlapping CPU audits were uncontrolled, so these are not method-speed comparisons.

The **Series30/Combat stable-winner** clip has three decoded scalar channel samples at or beyond absolute amplitude 1 (3 / 330,668), peak **1.003458142**, fraction **0.000009072544**. The other 17 clips have no such samples. This is an objective full-scale exceedance flag, not established audible clipping, its cause, synchronization quality or a merge-quality score. No normalization, repair or exclusion was applied.

The new self-contained method-blinded review is `.h3-study-artifacts/20260908/character-crossed-review-seed31/review.html`, ID **`172d5996e34b82dd0505`**. All 18 candidate copies pass independent full decoded-video/audio equality and metadata removal; verification SHA-256 `ad3aab6f741e48121d3c1f081bbef33b9d2f6e775b77c37e18ae0a0a5d562674`. Public manifest SHA-256 `1f78115c54fe0066c5af8c1dc2e3da179596a4c76b8b298e4a5b80709abdbb2f`; private-key SHA-256 `c85cb34517175e1329cda9c4ac5c46e0a1dcc94f08db062f2081ea8a36078fce`. Creator anchors are separately hash-checked/stream-copied, not counted as two extra candidate equality audits. Eighteen chronological 4-fps public-ID sheets are prepared, **not yet visually inspected** at this checkpoint. The assistant has not read the method-key mapping, assigned observations/ratings, listened to audio, or computed crossed similarity scores. Known earlier diagonal results and per-method technical audit flags remain prior exposure; future inspection must not be described as fully double-blind.

After terminal completion, a fresh empty-queue check allowed a successful cache release (HTTP 200). Local ComfyUI remains running, with **27,482 MiB GPU memory free**, 0% GPU utilization and **38,858 MiB available RAM** at verification. The remote server was untouched. Both execution plans, same-inode installations, diagnostic implementation pins and F08 supporting records revalidate. Both plans have **zero held-out histories**. Research code remains at the 783-test checkpoint; no code changed after that suite, only evidence/documentation. No commit, push, bump, installed node replacement or ranker/default update occurred.

Next: inspect all eighteen crossed sheets under public identifiers and preserve observations before unblinding/scoring; apply the already-frozen diagnostic; finish the remaining crossed calibration seed; then freeze the separate held-out policy. **90 character cases remain unexecuted** (18 calibration and 72 held-out). F08 needs its own opt-in implementation and validation without rewriting these baseline records. The larger goal remains active, and no universal merge winner is established.

### Crossed calibration review: identity, clothing and action remain separate

The [crossed seed-31 blind observations](data/2026-09-08-h3-character-crossed-seed31-blind-observations.json), SHA-256 `bdb7a8b72232a8d98036cadceb9df5e94511cf6e812e61a80e833c8f56c0208d`, cover all eighteen public-ID grids, **378 chronological samples**, plus a fresh inspection of both creator-reference grids. The record was saved and matched against the public manifest/media before reading the private method key or CPU scores. Earlier qualification and diagonal results were known; this is not fully double-blind. No continuous playback, listening, human rating or precise punch count was claimed from sparse images.

Unblinding gives these findings:

- **Series30/Combat:** all nine arms retain a navy jacket, red gloves and a lowered-glove ending. Additive and NP look particularly close at this scale. The stable winner, character-only, TIES and SLERP also form a close coarse-depiction cluster; small dim faces, hair/arm occlusion and different poses do not support a fine identity ranking. Two early action bursts must not be mistaken for exactly two punches.
- **Sully/Cinema:** character-only, additive, SLERP, TIES, stable winner and NP retain the reference-like green/purple, short-horned, broad-muzzled depiction, but **all six omit the requested vest**. CT keeps the palette and vest, but noticeably changes the muzzle/eyebrows/horns and ends with visible teeth. Base and effect-only differ more substantially from the reference, with long horns and beards. Clothing compliance cannot stand in for identity preservation.

The unchanged [crossed seed-31 CPU diagnostic](data/2026-09-08-h3-character-crossed-similarity-seed31.json), SHA-256 `6fa77fd7cd45afa7c6a4ddc28dae811141ea8453649401ccc28baef2b5a62dc6`, scored 108 sampled frames in **121.777 seconds**, four CPU threads, FP32, excluding model/reference loading. No metric, reference rectangle, processor, sample index or reporting threshold changed. Expected unused text/logit keys were excluded; the vision tower loaded completely. Times are not controlled throughput comparisons.

| Arm | Series30/Combat seed 31: median head-patch cosine | Sully/Cinema seed 31: median head-patch cosine |
| --- | ---: | ---: |
| Base | 0.680211 | 0.620388 |
| Character-only | 0.715339 | 0.628729 |
| Effect-only | 0.694686 | 0.613278 |
| Additive | 0.717523 | 0.635437 |
| SLERP | 0.697836 | 0.631158 |
| TIES | 0.691626 | 0.635189 |
| Stable winner | 0.725995 | 0.627825 |
| NP | 0.714494 | 0.633836 |
| CT | 0.683821 | 0.635315 |

The important negative evaluator result is **Sully CT nearly ties additive in this probe despite visibly different face/horn design** (raw difference −0.000122). Broad palette/part similarity is insufficient as an autonomous identity judge. The Series30 stable winner's higher cosine is likewise not a validated perceptual win. Character-only exceeds base in both new cells, but these add only more prompt/seed conditions for the same two identities, not more independent subjects. Absolute scores across different prompts are not comparable quality scales. No score was fed back into autotuning.

#### All-frame action and waveform follow-up

The [Series30 seed-31 frame/waveform observations](data/2026-09-08-h3-character-crossed-frame-waveform-seed31.json), SHA-256 `3f66b6d623331a7c54d0ed5083649e243cc31d2b5430006505f4809eccf8ab0f`, record inspection of **all 496 frames** across character-only, additive, stable winner and CT, alongside unchanged stereo waveforms. This is a declared post-unblinding exploratory four-arm selection, not a new blind trial or full nine-arm event comparison. Two original-resolution detail grids resolve smaller additional glove extensions without inferring them from sound energy.

All four exceed the exact two-punch target: character-only shows four alternating extension/recovery cycles, additive six, stable winner five, and CT four. These are visible movement cycles; precise first contact remains uncertain with bag-edge occlusion and blur. Every selected clip lowers the gloves and holds them down through the ending. CT's earlier Sully-specific ending advantage is therefore **not unique here**. Additional additive/winner cycles relative to character-only are one-seed observations, not a global method ranking.

Waveform detector maxima number four, eight, four and seven respectively. Thus maxima can both split a single visible event and miss smaller movements under the relative detector floor. **Waveform peak count is not punch count**; a missing peak is not proof of missing impact sound. No listening, sound classification, synchronization score or timing correction was invented. This follow-up strengthens the requirement to keep identity, action count, recovery/ending and AV timing separate.

### Calibration complete: 72 of 144 character cases audited

The final eighteen crossed seed-2026090832 jobs completed without retries, cancellations or restart. The [whole-clip audit](data/2026-09-08-h3-character-crossed-seed32.json), SHA-256 `53c6fd5ad6c190f211e20c8bca2eaeba941f344bba7fad9205a10ae0186cae7f`, verifies another **2,232 decoded frames**, exact frozen/prepared/executed graphs, and the same INT8-convrot diffusion / FP4 TE / 640×384 / 124-frame / 24-fps / 32-kHz stereo profile. All four pairings now have all nine arms at both calibration seeds: **72 videos / 8,928 frames audited**. Recorded execution intervals for this last block total **829.083 seconds**, range 29.672–125.331, with uncontrolled loading/cache state and overlapping CPU diagnostics.

Three Series30/Combat clips have decoded full-scale exceedances: character-only **1 / 330,668** scalar channel samples, peak 1.000463843; CT **3 / 330,668**, peak 1.036539435; effect-only **1 / 330,668**, peak 1.002343893. The remaining fifteen have none. These technical flags remain in the comparison; they are not established audible clipping or its cause, and no normalization/exclusion was applied.

The new review is `.h3-study-artifacts/20260908/character-crossed-review-seed32/review.html`, ID **`d8e60d442a967f969876`**. All eighteen candidate copies pass full decoded-video/audio equality and metadata removal, verification SHA-256 `ab306cedb342db867069cb17c6ba9850f7ef89e32d62effb9c98d2f61bc7d3c9`. Public manifest SHA-256 `ebd341658f85f37209f12907840a8bda7b35125215c7a3562bc8e8183bd45034`; private key `aa5c6e74615e0a294ac8f2daa746be3f77cae02aed4e15a2628f49798bb5ce4c`. Creator anchors were independently hash-checked/stream-copied, not counted as extra candidate decode audits. Ratings remain blank.

All eighteen grids and both creator references were inspected before saving [seed-32 blind observations](data/2026-09-08-h3-character-crossed-seed32-blind-observations.json), SHA-256 `5a77eb607f88a1eee57bb7ab5d73c5f9876987c3e4c1fc59974f156a88fcc35c`. Public IDs/media were verified before unblinding or similarity scoring. A one-off bookkeeping command first requested nonexistent `verification.json`; the actual successful verifier output is `media-verification.json`. No review or media was rerun or overwritten. Prior depictions and method-named audio flags were known, so this is not fully double-blind.

Unblinding shows:

- **Sully/Cinema:** the same six reference-like arms again omit the vest. CT again keeps vest/palette with changed face, ear/horn and corner-fang depiction. Unlike seed 31, it ends with a smaller smile rather than a broad toothy grin, although corner fangs remain visible. Base has the angular, long-horned toothy depiction; effect-only has a long-horned, two-tone goatee depiction. The vest/identity trade-off repeats, but a single ending fault should not be generalized.
- **Series30/Combat:** additive, NP, SLERP, TIES, stable winner and CT lower the gloves by approximately the middle samples and retain the ending. Character-only has three separated action bursts and lowers late; base has three bursts and **does not lower**; effect-only lowers late. All retain a navy jacket and red gloves. Sparse frames cannot certify the individual punch count. Additive/SLERP/TIES/winner remain close at this inspection scale; CT's darker fringe/face presentation requires visibility caution, not an automatic identity-loss label.

After terminal completion, a fresh empty queue allowed cache release (HTTP 200). Existing local ComfyUI remains running; no remote server was contacted. No production/research script changed during this calibration-review checkpoint; the latest full-suite result remains **783 passed, 3 skipped, 69 subtests passed**. No installed-node replacement, scoring-weight/default change, commit, push or version bump occurred. H3/ComfyUI skills guided the unchanged profile and idle-only cache handling.

The second crossed CPU diagnostic is being applied under the frozen calibration recipe, after preserving the observations. The remaining **72 held-out character cases** have not been generated or opened at this checkpoint. F08 remains an unimplemented, reproduced candidate-search issue. Next: preserve the final diagnostic, freeze a held-out reporting policy without fitting a quality judge, and execute the remaining predefined comparisons. The overall goal remains active.

#### Final calibration diagnostic and held-out freeze

The [crossed seed-32 diagnostic](data/2026-09-08-h3-character-crossed-similarity-seed32.json) completed with SHA-256 `01eeb36b91e5ffc7a8aba4bef1a89b43fbb4813c80cb7234602b4176904ebaba`: **108 frames / 118.004 seconds**, four CPU threads, unchanged FP32 vision model/processor/reference rectangles. Loading/reference extraction is excluded from timing. Observations were saved before computing or reading these values; no labels or autotuner feedback were added.

| Arm | Series30/Combat seed 32: median head-patch cosine | Sully/Cinema seed 32: median head-patch cosine |
| --- | ---: | ---: |
| Base | 0.683765 | 0.630218 |
| Character-only | 0.746453 | 0.623467 |
| Effect-only | 0.712649 | 0.611175 |
| Additive | 0.769973 | 0.630278 |
| SLERP | 0.758163 | 0.623851 |
| TIES | 0.758543 | 0.623008 |
| Stable winner | 0.768493 | 0.634941 |
| NP | 0.766045 | 0.624208 |
| CT | 0.740894 | 0.627524 |

**A second concrete evaluator failure:** Sully/Cinema's visibly different base depiction nearly ties additive (−0.000060) and exceeds character-only. Across all eight calibration pair/seed cells, character-only exceeds base in seven, not eight. CT is below additive in eight, NP and SLERP in seven, and TIES/stable winner in five, but these are **raw diagnostic directions, not validated perceptual wins/losses or independent subject trials**. They cannot justify an automatic CT penalty, NP boost or learned ranker. Fine identity, visibility, wardrobe, exact action and sound are not captured adequately by this one cosine formula.

The [held-out policy](data/2026-09-08-h3-character-heldout-policy.json), SHA-256 `40da10ebc71044f09ded4f7761388a367ac145116f45a34ed7088291b99bb760`, was saved after these calibration results and before any held-out submission. It pins both execution plans, the original protocol, all fourteen calibration evidence records and four review/audit helpers. Fresh checks verified **36 calibration / zero held-out histories in each plan**, the complete disjoint 144-case union, installed asset identities and every prerequisite hash.

The policy retains all **72 predeclared held-out cases**: four pairings × nine arms × two new seeds, using the already-frozen library/book and ordered jab/straight prompts. Generation settings and baseline exports do not change. Four 18-case blocks run serially in the order diagonal seed 41, crossed seed 41, diagonal seed 42, crossed seed 42. The first started on existing 8189 after a fresh empty-queue check; its first watched job is `idm-sully-combat-2026090841-slerp`, prompt ID `286110c6-e473-4790-b94d-5b467be56c50`. This is a launch record, **not a held-out completion count**. The existing durable submission/receipt and WebSocket safeguards remain in force; unrelated queued work stops new submissions.

Each complete block receives an AV-equality-verified, method-blinded review and saved public-ID observations. The policy additionally fixes **all-124-frame plus unchanged-waveform inspection for every held-out candidate**, not just favorable methods or flagged outputs; ambiguous events receive original-resolution contiguous-frame follow-up. Detailed observations must be saved before observer unblinding. Identity, visibility, instruction compliance, action/hand ordering, ending, temporal defects and waveform timing remain separate. Missing audio listening is not filled in with invented sound labels or scores. Any later human judgments retain their own provenance and do not retroactively make assistant observations human ratings.

The calibration-only cosine recipes will **not** be widened to held-out outputs: their fine-identity ranking failure is already demonstrated. No fitted replacement evaluator, new weights, automatic NP/CT bonus or quality-driven candidate change is proposed. F08 remains a separate opt-in search-efficiency correction requiring active-parameter preservation, actual tensor equivalence, disconnected regressions and controlled measurements; these frozen nine baseline arms must not be rewritten to hide duplicates or replace winners.

Fresh full-suite validation after these evidence records: **783 passed, 3 skipped, 69 subtests passed** (23.71 seconds); `git diff --check` is clean. Production and pinned research-helper hashes revalidate unchanged. The current pass changed evidence/documentation, **not node behavior**. No installed-node replacement, commit, push or version bump. The goal remains active: held-out generation/review, F08 implementation/validation and the final shipping audit are still required.

### F08 opt-in prototype: effective-decision deduplication and cache reuse

The reproduced inactive-parameter issue now has an **uncommitted production prototype** in `lora_optimizer.py`, SHA-256 `44695e780a8bb8e67d206827ce4a561fd66b0a710fe89ed63026b521753c7461`. Only a normalized, enabled experimental-options connection activates it. The disconnected/empty/master-disabled/both-methods-disabled paths retain the original search and cache identities. NP/CT formulas and `experimental_merge.py` remain unchanged (`36bd479c1961dc6d2e5c79e1d3a79202e3149f47b1025483912f72ac1a3b57d4`). Existing frozen exports and the installed node pack are untouched.

- Plain weighted-sum/weighted-average decision signatures ignore unused density/sign placeholders. Shared-group cache planning and execution use the same canonicalization. TIES voting/density and all other modes retain their original distinctions; refinement, sparsification and associated parameters remain separate.
- Actual streamed auto-strength math is computed once for admission. Enabled/disabled trials collapse only when **both model and CLIP scales equal exactly 1.0**. A nontrivial CLIP scale or even a tiny nonexact model scale keeps both trials. External evaluators retain strength variants; a precompute failure disables this shortcut rather than the sweep.
- Experimental-only search version 1 invalidates in-node and persistent experimental tuning results. Stable identities and the global stable algorithm version are unchanged. Experimental tuner data records the version, number of skipped candidates and no-op-strength decision.

The stable `top_n` **budget**, not necessarily its old configurations, is retained. A freed slot can admit a different next candidate and change the winner. Different formulas can still coincide on particular inputs, so this is not a guarantee that every surviving output is unique. No quality bonus or scoring-weight change is introduced.

Fourteen new regression tests (`tests/test_experimental_candidate_dedup.py`, SHA-256 `68ab1af301c7b7fdbdee9a0a512779f878b130617a49e47010973571e73b1842`) cover actual reproduced decisions, 60 signed FP32/BF16 linear tensor-equality cases with refinement/sparsification, genuinely active TIES fields, opt-in admission, failure/callback handling, disconnected identity, and versioned in-node/persistent caches. The two-target integration fixture proves positive runtime group-cache hits with zero unclaimed entries; disabling the cache or setting a zero RAM budget preserves candidate scores and both winner tensors. Active refinement/sparsification candidates remain admitted.

The initial cache-version test failed because it incorrectly expected the single-entry in-node cache to retain a stable sweep after an experimental sweep replaced it. Source inspection confirmed intentional eviction; the corrected test uses separate stable/experimental nodes, first proves both cache hits, then proves a version change invalidates only experiments. Importing a fixture module instead of its public TestCase also avoids accidentally collecting its unrelated tests twice. These are test corrections, not evidence that a production cache defect was fixed.

The optimizer checkpoint passes **797 tests / 3 skipped / 155 subtests** (30.20 seconds), both actual-ComfyUI CPU integration suites, and all three JavaScript workflow-migration tests. **F08 is not fully qualified yet:** full-size rank-2/3 tensor equivalence, real-stack shortlist effects and controlled efficiency measurements remain pending. CPU test timing while other study work runs is not a speed benchmark. Any changed winner needs a separate prospective render comparison, not replacement of these frozen nine arms.

### First held-out seed: 36 outputs technically audited, reviews still blinded

Both seed-2026090841 blocks completed without retries, cancellation or server restart. The [diagonal audit](data/2026-09-08-h3-character-heldout-seed41.json), SHA-256 `b5460850bff79fb6b641fec6d66a2886af91fd26c2a3044b5354256480523b6b`, and [crossed audit](data/2026-09-08-h3-character-crossed-heldout-seed41.json), SHA-256 `23e1a63f00682f538f3a9e5892985aec90d3d2b3885dedfdf34f0484619f3f20`, cover another **4,464 decoded frames** and exact executed/prepared graphs. Both keep the frozen INT8-convrot / FP4-TE / 640×384 / 124-frame / 24-fps / 32-kHz-stereo profile. The study now has **108 technically audited clips / 13,392 frames**, not 108 completed perceptual reviews. The serial parent advanced to diagonal seed 42; no second GPU study was started.

Execution intervals total 833.660 seconds for diagonal (range 28.981–107.989) and 798.467 for crossed (29.329–106.695). Loading/cache state and CPU work overlap are uncontrolled. Diagonal Sully/Combat effect-only has six scalar audio samples at or beyond absolute amplitude 1 out of 330,668, peak 1.016740441. Crossed Series30/Combat stable winner has one, peak 1.015299678. The other 34 clips have none. These are amplitude flags, not established audible clipping, sound quality or causal attribution; every output remains in the study unchanged.

The diagonal method-blinded review is `.h3-study-artifacts/20260908/character-heldout-review-seed41/review.html`, ID `ed3a302f22dd4a93227c`. All eighteen candidate copies pass full decoded AV equality and metadata removal (verification SHA-256 `9a79fdf79c7c59a6fc29a241cbaff9fbdeac32ac7a786e13dc9d59b16df8e282`; public manifest `a760ed94d94c84b60f253fa222be7b1e6f3002d4634fbf4b15488f534d4c91e0`). Its private-key contents have not been exposed to the observer.

All eighteen sparse grids and both creator-reference grids were inspected before saving [initial public-ID observations](data/2026-09-08-h3-character-heldout-seed41-blind-observations.json), SHA-256 `dedc6c9e9f7e53febc72792166007f94a44242d78d28bb2164a5a3f9f4d4a073`. At this scale, C03 holds an open book rather than the requested closed book; several other library candidates begin with a hand already on the book, making action order important. Sully C01/C12/C14 retain the vest with altered face designs, while the other six omit it. C11/C13 lack the sampled lowered-glove ending. These are anonymous observations, **not method results or sound judgments**. Full-frame inspection remains required before observer unblinding.

#### Separate held-out frame/waveform gate

Inspection revealed that the frozen `h3_av_sync.py --observations` entry point explicitly rejects held-out stages. It remains unchanged. The separate research-only `scripts/h3_character_heldout_sync.py` (SHA-256 `d76fcf18d109bb65631eab9e6cf1ea16dfe03c53292812ff8bfe205c9eefe4f5`) implements the already-frozen held-out policy rather than weakening calibration gates. It checks the policy/plan/helper/evidence hashes, complete 18-case observation/key/public/AV-verification alignment, every original/review media identity, full-frame audit and exact graph before preparing the block. It cannot select only favorable methods or overwrite an existing output directory.

A private instance of the original renderer changes only its legacy HTML heading. Frame decoding/PTS, stereo 5-ms envelopes, anonymous maxima and all-frame sheets are unchanged. Outputs use public C-identifiers and hash-linked provenance, without execution IDs or method-named source paths; extraction errors suppress potentially unblinding ffmpeg paths. Twenty-four new tests cover complete scope, tampering, missing/duplicate cases, calibrated-versus-held-out separation, unchanged original module/math, anonymous full-block output and failure/no-overwrite behavior. Combined sync tests: **41 passed**. A real 18-case preflight succeeded, and serial CPU extraction is in progress. Matplotlib used a temporary `/tmp` cache because the user's config directory was not sandbox-writable; this was not a render failure or a reason to restart.

The [detailed observation record](data/2026-09-08-h3-character-heldout-seed41-frame-waveform-observations.json) is explicitly **in progress**: C01 alone has all 124 frames/four waveform sheets inspected, plus three original-resolution eight-frame grids. It shows three distinct punch/recovery cycles: twice with the far-side glove, then once with the near-side glove. Contact is partly occluded, so windows remain ranges rather than exact first-contact timestamps. Gloves lower around frames 98–107 and stay down. Eight waveform maxima fall into three broad energy groups; no sound classification, listening or synchronization score is inferred. The other seventeen candidates must receive the same detailed scope before the method key is opened.

The crossed seed-41 blinded review is also being prepared under separate public identifiers; no crossed held-out images or method mappings have been inspected yet. All remaining seed-42 generation/review, complete held-out detailed observations, F08 full-size/efficiency checks and final shipping audit remain required. Fresh full suite after the new gate: **821 passed, 3 skipped, 155 subtests passed** (33.31 seconds), with `git diff --check` clean. The H3/ComfyUI skills guided the unchanged low-memory render profile and serial job handling. No model unloading during the active runner, remote-server contact, installed-node replacement, commit, push, version bump or quality-ranking update occurred. The goal remains active.

#### Held-out preparation completion checkpoint

The diagonal all-frame preparation finished successfully for **all eighteen candidates / 2,232 frames / 72 waveform sheets**. Every generated artifact hash in `character-heldout-sync-seed41/prepared.json` revalidates; manifest SHA-256 `c15b32b926eca7464cc11f7e842df07977fa48d931ee705ac1fecb9748682fd6`. This is preparation, not eighteen detailed reviews: only C01 is inspected so far. Its partial observation record currently hashes to `62334ef86097c47a82061743fc9ee5d39293d3e83e25bdf8377fd9e3844776ea`; the remaining seventeen entries are explicitly pending. Source/frame/detail/provenance checks pass and the observer has not read the method mapping.

The crossed review finished and all eighteen copies pass decoded AV equality/metadata verification: review ID `8a2cfbe0a20999f6cfed`, public-manifest SHA-256 `72bc7cf8fc9795228a23314eba94a83bc2c297296d11e2a721b302856a8a086b`, verification SHA-256 `6c400af7e9272f9300347cc9d0e81c8ee9ba9c458988122d69fa4d34ae41d697`. Its eighteen sparse grids are ready but uninspected. The existing GPU runner is still live: both seed-41 blocks and nine cases of diagonal seed 42 have terminal-success receipts (**45 held-out generations completed**, only 36 held-out originals audited). The currently watched job is `idm-series30-cinema-2026090842-stable-tuner-winner`, prompt ID `4df0a9dd-2b8f-41a5-b8a5-3fc195a1e871`. No new submission process or cache release was started.

### Third held-out block audited; full-size duplicate replay prepared

The existing serial runner completed diagonal seed 2026090842 and advanced to the last crossed block, without resubmission or restart. The [third held-out audit](data/2026-09-08-h3-character-heldout-seed42.json), SHA-256 `d749f80809121f71fe429e62a88a2fdac4d7c3e1641079ed44c3b41d55bae009`, covers eighteen exact frozen graphs and another 2,232 fully decoded frames: **126 clips / 15,624 frames technically audited**. Recorded execution intervals total 796.496 seconds, range 28.853–109.246, with uncontrolled caches/loading and overlapping CPU work. Sully/Combat TIES has four full-scale scalar samples out of 330,668, peak 1.032587886; the other seventeen have none. This remains an amplitude flag, not established audible clipping or an exclusion criterion.

The new blinded review is `character-heldout-review-seed42/review.html`, ID `4fac4d820326bc91440d`. All eighteen copies pass complete decoded-AV equality and metadata removal: public SHA-256 `00da43bc1d5d59f33d5bc000095d56f31155953a19b6c3583ef446f2a993e8b9`, verification `13e416cabd804c58fe9d384d27ff83c89a59700843d4627c1b1af02b55c1af0e`. No visual observations, similarity scores or method mappings from this block have been read. Sparse grids are being prepared, not rated.

Diagonal seed-41 detailed observations now cover **C01–C04 / 496 frames**, with fourteen candidates still pending. The [partial record](data/2026-09-08-h3-character-heldout-seed41-frame-waveform-observations.json), SHA-256 `e1217a52499f35b6116ad634ff3274f38fb35bdb29ae06e1058b51702b3848f0`, incorporates the previously inspected C02–C04 full-frame sheets plus new original-resolution hand details. C02 moves its hand before finishing the gaze lift and ends with curved fingers at the book edge, not a clearly flat palm. C03 holds an open book throughout. C04 has initial book contact, then later repositions flatter after looking up/smiling; complete palm contact remains uncertain. All frame/source/detail hashes and 124-frame counts revalidate. Forty-five to forty-seven very low-amplitude waveform maxima in these library clips are not identified sound events. Method identities remain unopened; no listening or AV preference is fabricated.

`scripts/h3_candidate_equivalence.py` (SHA-256 `21575d6ee0c1e7ac9b57f1172647a5c6cc3a9a300f962a6a3c5fd8d2d109a276`) prepares eight separate selector replays for the original rank-2/rank-3 records in all four observed pairs. It requires identical complete linear decisions and matching active settings, pins all inputs/implementation, and changes only the ordering in derived tuner-data copies; original rank labels and settings remain intact. Execution uses fresh serial real-Comfy processes, checks the local queue before each GPU subprocess, and refuses existing outputs. Every replay must pass the independent native key/target checker (208 native targets / 312 components). The final comparison reads and hashes every stored tensor byte, including signed alpha, BF16 factors and tail chunks. Metadata is deliberately excluded because selected strategy labels differ. This proves only representation equality if successful, not video quality or a controlled speed gain.

The prepared replay plan is `.h3-study-artifacts/20260908/f08-candidate-equivalence-01/plan.json`, SHA-256 `2e955b378b747e248b5cfe69c7b169c573af02c90e309d3d65080fae9b2587f0`; **GPU execution has not started at this checkpoint** because the last render block is still active. Thirty-four new offline tests cover selection immutability, complete scope, changed active parameters, missing/nonlinear decisions, provenance/path/strength tampering, occupied/unknown queues, and chunked tensor mismatches including signed zero. An additional real-safetensors CPU check confirms FP32/BF16/scalar-alpha equality despite differing metadata and catches a last-element mutation. Full suite: **855 passed, 3 skipped, 155 subtests passed** (58.65 seconds with other work overlapping, not a performance benchmark). Production F08 code and frozen render helpers remain unchanged. No installed-node replacement, commit, push or version bump. The H3/ComfyUI skills preserve the original low-memory profile and serial GPU execution; the goal remains active.

### All character renders audited; F08 duplicates proven at full size

The original four-block render parent reached terminal success for all 72 held-out jobs. The [last crossed audit](data/2026-09-08-h3-character-crossed-heldout-seed42.json), SHA-256 `18cdc49cbd7c99266eaf4cdb3cbd1d73786bf891accc27cae88b4c586d4b9d27`, adds eighteen clips / 2,232 frames. Recorded execution intervals total 808.073 seconds, range 28.998–113.006, not controlled throughput. Series30/Combat stable winner has one full-scale scalar sample out of 330,668, peak 1.031994343; the other seventeen have none. No audio normalization, exclusion or listening claim follows.

A fresh union check traversed every job in both frozen plans, verified terminal success and exact executed/prepared graphs, rehashed every media file through `collect_render_run`, and checked the fixed AV stream profile. It found **144 distinct jobs / 17,856 decoded frames**, exactly nine arms in each of sixteen pair/stage/seed cells. This proves technical completion, not complete audiovisual evaluation.

The last blinded review, `character-crossed-heldout-review-seed42/review.html`, ID `028333549187dc0e45a0`, passes full decoded-AV equality/metadata removal for all eighteen copies. Public SHA-256 `65d657655bb5d4309327d08f60e258f79c9c572269615616551ef0678e50551a`; verifier `c33cd20117838e08b360a60e19e904b6944cfca9e22962591118847089e28878`. Both seed-42 blocks now have all sparse grids prepared, but neither has been visually inspected or unblinded. Detailed diagonal seed-41 observations cover **C01–C06 / 744 frames**, twelve still pending; record SHA-256 `e89438bbd62755292dc5219d2f1c945a690b56ec2b1ed69cec5cc53bbe6e7b9e`. C05 has initial hand/book contact and a later lift/replacement after the gaze/smile; original-resolution ending confirms a closed-mouth smile. C06 omits the vest and makes three same-far-glove attempts, the first contact particularly occluded, before lowering the gloves late. Its eight waveform maxima are not eight punches. These remain anonymous observations without sound judgments.

After the render parent became terminal, a fresh localhost-8189 queue check was explicitly empty. Idle model-cache release returned HTTP 200; GPU free memory rose from 1,789 to **27,481 MiB** before any merge work. No server was stopped/restarted and no remote server was contacted. The eight pinned duplicate replays and independent native-loader checks then ran **serially to terminal success**, without generation or installation.

The [full-size numerical record](data/2026-09-08-h3-candidate-equivalence-numerical.json), SHA-256 `8d33de37baa6ef7f08cd04ef8157e289fdd4adca2d47528ebd7808a68c55dcb5`, preserves all eight replay configurations, source identities, complete decision maps, numerical coverage and measurements. Every export passes **208 native targets / 312 components**, all source/export keys consumed and all results finite. Maximum relative FP32-reference discrepancy is `1.40446e-7`; no matrix subset substitutes for the full coverage. The byte-comparison result is `.h3-study-artifacts/20260908/f08-candidate-equivalence-01/result.json`, SHA-256 `72251edc9917946a16e9ff0f6d909cb9f994d170a653270d06836966a9d7a377`.

**All four original rank-2/rank-3 pairs have exactly equal stored tensor bytes**, across 624 tensor pairs each (2,496 comparisons total), including factors and alpha. Whole-file hashes differ because metadata legitimately records different strategy labels. This closes the full-size duplicate-equivalence gap; it is stronger than equal decision names or scores. Implementation/plan/input hashes and all eight manifest/check/export identities were independently revalidated after completion. The original baseline files and tuner evidence are untouched.

Each replay peaks at 2,476,961,792 bytes allocated CUDA (about 2.31 GiB); elapsed merge/export times range 11.155–21.490 seconds. These fresh-process replays had overlapping CPU audit/review work and uncontrolled file/OS caches, so they are **not** a controlled old/new search speed measurement. The opt-in deduplication prototype still needs real experimental-shortlist/candidate-budget checks and controlled efficiency trials; any changed winner needs a separate prospective comparison. No visual-quality ranking/default change, installed-node replacement, commit, push or version bump occurred. The goal remains active.

### Paired search timing protocol frozen before execution

The new research-only `scripts/h3_search_benchmark.py`, SHA-256 `c118379ff0ef8a3b6b89b9df4202c23fd0935ef4b29fc07005add6ca8d5a5357`, prepares isolated source snapshots, not installed nodes or a worktree checkout. Baseline is committed `27bedd7621e8e53b5d9c4d87ba594c2b7aa76008`, optimizer SHA-256 `3d245ea08cda5aa4faf49cec9a878a93a521cb2bc71088d989f30c1d63346145`; current optimizer remains `44695e780a8bb8e67d206827ce4a561fd66b0a710fe89ed63026b521753c7461`. Both use byte-identical `experimental_merge.py` and `kernel.py`. Inspection confirms the optimizer delta is F08 plus the inert fresh-dictionary storage hook; no formula/scoring-weight update is included.

The predeclared plan is `.h3-study-artifacts/20260908/f08-search-benchmark-01/plan.json`, SHA-256 `85e032b0214907902b460e5001dd40a0a0817043647de102a578b7ed222cf7b5`. It contains **40 trials**, all four existing pairs: eight disconnected controls, eight experimental warm-ups, and twenty-four measured experimental trials (three repetitions per pair/version). Version order alternates and is balanced across measured pairs. Both versions receive the same enabled NP/CT options in experimental trials. **Correction:** the first harness guard incorrectly expected only three stable plus two experimental rows, omitting the existing explicit additive control; see the preserved failure below. Comparing an old stable-only run with a new experimental run would confound F08 with adding NP/CT, so that is not the timing contrast.

Each worker is a fresh process with isolated model/user/temp/cache directories, fixed four-thread settings, merge seed 2026090800 and highest FP32 matmul precision. It uses real ComfyUI native source mapping against the same header-pinned, shape-only FL2VA/pruned model; no checkpoint payload or render is loaded. Source roles/strengths/hashes are pinned. A lightweight CUDA initialization precedes the timer; explicit CUDA synchronization brackets the actual full-target `auto_tune(..., output_mode="merge")` call. Imports, source loading/hashing, initialization, output hashing and serialization are excluded. Full scoring and disabled persistent/diff/community caches match across versions. Actual kernel availability, TF32 setting, GPU memory/temperature/clocks/power and available RAM are recorded. Clocks are observed, not locked; this is a controlled local procedure, not complete system-wide isolation.

The worker then hashes every stored model/CLIP winner patch outside the timing interval, including dtype, shape, signed alpha and slice offsets. Default regression checks compare ordered configurations, complete decision maps, scores and winner-payload hashes. Experimental reports keep newly admitted/removed configurations, common-candidate score changes, selected winner and output changes separate. A different candidate changes the work being performed: a time ratio is not an equal-candidate kernel speedup or a quality score. No favorable subset, automatic retry or overwrite is allowed. A parent watchdog may terminate only its own worker if its RSS exceeds the conservative bound, preserving a failure receipt; it never terminates ComfyUI or unrelated processes. Queue checks before/after each trial stop new work if user jobs appear.

Twenty-four new tests cover order balance, source snapshot/plan integrity, role/strength tampering, typed nested winner hashes, non-finite/unknown payloads, complete timing scope, warm-up exclusion, admission/default-output separation and refusal to restart partial execution. Full suite **879 passed, 3 skipped, 155 subtests passed** (28.80 seconds), completed before timing starts; `git diff --check` is clean. Fresh outside-sandbox preflight: local queue explicitly empty, RTX 5090 free memory **27,479 MiB**, utilization 0%, temperature 39°C. No cache release was needed. Other heavy study computation (tests, ffmpeg, learned diagnostics or another GPU study) is paused during timed execution; only monitoring and light note-taking continue. The H3/ComfyUI skills guide this serial scope. These are preparation/preflight results, not forty completed trials. Held-out method keys and observations remain unchanged; no commit, push, version bump or installed-node replacement.

#### Preserved first-attempt failure and corrected six-candidate guard

Revision 1 completed the two Series30/Combat disconnected controls, with identical full winner-patch hash `2e7e609966c9534acaab5a43d3d2570f0de5f0244fbd5a993763cce2169ebb91`, the same 7,036,706,816-byte CUDA allocation peak, and expected full/no-SLERP/basic choices. Their 25.353/16.776-second timings are controls, not a speed claim. The next baseline experimental warm-up completed its actual merge/search, including all six candidates, but the harness rejected its result because it expected five. The parent stopped with return code 1; the memory watchdog did not trigger. No measured repetitions ran.

Source inspection confirms `experimental_merge.candidates()` intentionally inserts a separate `experimental_baseline=True` weighted-sum comparison before NP/CT. Thus the correct connected budget is **three stable + one additive control + NP + CT = six**, and disconnected remains three. This is a benchmark-validation mistake, not an autotuner defect or permission to remove the additive control. All original attempt files remain; `stopped.json` hashes to `180aa6a70e3a6f17ea1e2a8cb0052c0422888ee65a4f34cd5222a3bc7edae7c0`, and `helper-before-cardinality-fix.py` preserves the exact original helper bytes/hash `c118379ff0ef8a3b6b89b9df4202c23fd0935ef4b29fc07005add6ca8d5a5357`.

The corrected helper (`ec56c1218cc9ec4cc263a445bb6c36adb45a3d9a177b161ef9058e6b0a085841`) validates stable candidates separately from the additive comparison and NP/CT, and records all four counts. Nine added regression cases use the **actual** experimental candidate constructor and reject missing additive/NP/CT rows, extra stable rows, non-finite scores, partial target counts, an incorrect additive mode and experiments on the disconnected path. The thirty-three benchmark tests pass; fresh full suite **888 passed, 3 skipped, 155 subtests passed** (28.70 seconds). No production formula, candidate list, score or default changed.

Revision 2 is a new `.h3-study-artifacts/20260908/f08-search-benchmark-02/plan.json`, SHA-256 `dc62e2c58c2856948e321a4c985aee3a9c2719bd418b66ee118d82f87eaca599`. It pins the corrected helper and explicitly lists the six-candidate budget. All forty controls/warm-ups/measured runs will be newly executed; no revision-1 timing is folded into the comparison. Both optimizer snapshots, adapter strengths/order, merge seed, cache isolation, synchronization and timing boundaries are unchanged. The earlier partial directory is not resumed or overwritten. Benchmark results, held-out reviews and shipping qualification remain incomplete.

#### Interim controlled result: first scheduled pair is slower, not improved

The revision-2 parent is confirmed live through execution session 15660, not inferred from a lock or plan file. It has completed the ten scheduled Series30/Combat trials and has advanced into Sully/Cinema measured repetitions. Both baseline/current experimental warm-ups pass the corrected six-candidate guard. The previous status-only turn was a verified wait on this same live handle; no second parent, retry or replacement source snapshot was started.

All three measured repetitions for the **first predeclared pair**, Series30/Combat, are now available. This is an interim result, not a selected favorable subset or the completed four-pair report. Timing excludes the controls and warm-ups:

| Version | Three measured search times (seconds) | Median (seconds) | Peak allocated CUDA in every repetition |
| --- | --- | ---: | ---: |
| Committed baseline | 21.871067, 21.557277, 21.672249 | 21.672249 | 7,036,706,816 bytes |
| F08 prototype | 24.012013, 24.097597, 24.276930 | 24.097597 | 7,036,706,816 bytes |

The median ratio is **1.111910**, or **11.19% longer**, with no allocated-VRAM improvement on this pair. The raw receipts remain in `f08-search-benchmark-02/runs/series30_combat-measured-{0,1,2}-{baseline,current}/result.json`; each pins its tuner data, merge log, plan and implementation. All six full winner-payload hashes equal `51ad0549cd4858bef0fcdb2940912d5492227a1b230b66854d07181ba8ccbf2c`. All select NP under the unchanged internal mathematical score; this is not an audiovisual preference or a new NP-quality finding.

Independent comparison of all three old/new tuner-record pairs finds five shared configurations each, **all fifteen score differences exactly zero** and all complete 312-prefix decision maps equal. Every current run drops `basic` and admits `full + della_conflict`; its metrics report **zero applied groups and 300 conflict-skipped groups**. The other twelve targets have only one contributor. The new candidate ties ordinary `full` in all reported numerical score components; its whole stored patch payload has not yet been independently compared against `full`, so this is not another byte-equivalence proof. Removing the original placeholder duplicate does not establish that the replacement offers a useful new result.

All six measured merge logs show **289 planned shared-group replays but zero actual replays and zero cached bytes**. The source's `_gc_store` explicitly refuses CUDA-resident patches, while this protocol permits retaining patches within its VRAM budget. Thus the prior CPU cache-hit regression tests do not demonstrate a real GPU-run speed gain; planning counters must not be reported as realized savings. The admitted conflict-aware candidate takes about 2.6–2.7 seconds to merge versus about 0.3 seconds for the removed baseline candidate in these logs. Those rounded phase timings help explain the search-cost change, but are not a separate synchronized microbenchmark.

The completed disconnected controls for **both Series30/Combat and Sully/Cinema** independently retain ordered configurations, exact scores, full decision maps and full winner-payload hashes. Sully/Cinema's control timings and allocated peaks are not identical, so unchanged outputs must not be misreported as identical resource measurements. The remaining pairs still require their own controls and measured comparisons.

This checkpoint changes documentation, not production code or benchmark settings. The optimizer, helper and plan hashes revalidate unchanged. Heavy study work remains paused during timed execution; only monitoring, small-record/source inspection and note-taking overlap. The H3/ComfyUI skills preserve serial GPU execution without unloading or restarting the live server. Latest full suite remains **888 passed, 3 skipped, 155 subtests passed**, run before timing, not rerun concurrently. Held-out observations/method keys remain unchanged; no installation, commit, push or version bump occurred.

Next: finish all forty frozen trials and audit their receipts/comparisons before revising or shipping F08. A harmless duplicate-removal rule is not sufficient shipping evidence if the actual replacement is redundant and slower. Any follow-up guard/cache optimization must preserve active sparsification, external-evaluator semantics and complete-target numerical evidence; sampled conflict statistics cannot substitute for the exact skip contract. Held-out visual/AV review and the final goal-wide shipping audit also remain incomplete.

#### Halfway checkpoint: second scheduled pair also costs more without changing the winner

The same parent completed all ten Sully/Cinema trials and started the first Series30/Cinema disconnected control: **20/40 trials complete**, no restart or new benchmark. Sully/Cinema's three measured baseline times are **18.841658, 19.178509, 18.339692 seconds**, median **18.841658**. Current times are **23.401686, 22.799548, 22.379331 seconds**, median **22.799548**. The ratio is **1.210061**, or **21.01% longer**. Baseline CUDA allocation peaks are 17,673,527,296 / 17,673,527,296 / 17,670,086,656 bytes; current is 17,673,658,368 in all three runs. There is no allocated-VRAM saving here either. Observed before/after GPU temperature is 38°C and clocks 2557/13801 MHz across these measured trials, but those endpoint observations do not constitute continuously locked/isolated hardware.

All six winner-payload hashes equal `6e06b653c052c43ba8a0eea4b4ec158b9f717accef5d187b8eaead26d46208c2`; all select the same NP configuration. Each old/new repeat has five common configurations with **exactly zero score deltas and equal complete decision maps**, another fifteen comparisons. The current run reports four skipped admission duplicates, removes `basic`, and admits `full + dare_conflict` with auto-strength disabled. All **312 conflict groups skip, zero apply**, and its score/ordinary numerical components match the retained `full` candidate. As with the first pair, the admitted no-op candidate's full stored patch equivalence remains unproven; the unchanged selected-winner hash is a separate verified fact.

Sully/Cinema differs from Series30/Combat in actually realizing some CPU cache reuse: current logs report **48 replays / 58 MB peak** in all three runs; baseline reports 48/48/50 replays and 58/58/62 MB. Both plan up to 361 replays. Cache activity therefore depends on actual patch placement and must not be generalized as uniformly zero or fully effective. The newly admitted candidate's logged merge phase takes about 4.3–4.8 seconds versus the removed candidate's 0.3–0.4 seconds. The baseline's repeat-to-repeat cache/peak variation is retained, not normalized away.

A fresh lightweight integrity check rehashes **80 completed files**: result, tuner data, merge log and telemetry for each of the first twenty scheduled jobs. All twenty tuner/log digests match the corresponding result receipts, all plan hashes match revision 2, every winner covers 208 native targets, and all disconnected/experimental counts match 3/6 respectively. There are twenty distinct jobs, with all controls and warm-ups retained and excluded from the twelve measured timings. This is a half-plan audit; no final `summary.json`, remaining-pair pass, perceptual gain or speed-up is claimed. The final report must include every scheduled pair, including these negative results.

### Complete paired search benchmark: F08 does not earn shipping as-is

Execution session 15660 reached **terminal exit 0 after all forty trials**, preserving the original order, both disconnected controls and warm-ups per pair, and all three measured repetitions per version. No retry, cancellation, second parent or source change occurred. The final summary is `.h3-study-artifacts/20260908/f08-search-benchmark-02/summary.json`, SHA-256 `d0991a1b46e1deccd4c399f9b56021134096621a1a08ac1cccc5eb053ec24a7f`. The failed revision-1 harness attempt remains separate and contributes no timings.

| Pair | Baseline median (seconds) | F08 median (seconds) | F08 increase |
| --- | ---: | ---: | ---: |
| Series30 / Combat | 21.672249 | 24.097597 | 11.19% |
| Sully / Cinema | 18.841658 | 22.799548 | 21.01% |
| Series30 / Cinema | 41.302192 | 47.974887 | 16.16% |
| Sully / Combat | 15.810048 | 17.711764 | 12.03% |

All twelve paired measured comparisons take longer with F08. The Series30/Cinema raw baseline times are 26.307047 / 41.302192 / 42.868579 seconds; current is 33.522145 / 47.974887 / 51.386871. **Both versions vary substantially**; none of these repetitions was dropped or replaced. Endpoint GPU temperature/clocks do not establish continuous isolation. A read-only process snapshot near the final pair showed resident desktop/Resolve/ComfyUI GPU processes; it neither proves nor rules out contention during earlier timed intervals, and no unrelated process was modified. These are measurements under the predeclared local procedure, not a universally transferable percentage or an equal-candidate kernel benchmark.

All four disconnected comparisons preserve ordered configurations, full decision maps, scores and full winner-payload hashes. Across experimental repeats, **all sixty common-candidate score differences are exactly zero**, with complete decision-map equality. **All twelve old/new winner-payload comparisons are equal**, with one repeat-stable winner hash per pair/version. Both versions select the same NP settings in every measured run. This is unchanged mathematical ranking/output, not evidence that NP is perceptually best. No new F08 winner requires a prospective render at this checkpoint.

Every current run removes `basic` and admits a `full` conflict-aware candidate. Combat pairs admit DELLA-conflict; Cinema pairs admit DARE-conflict. All 300 shared Series30 groups or 312 shared Sully groups skip sparsification, zero groups apply it, and the added candidate matches retained `full` numerical score components. Whole stored-payload equivalence of those added candidates is still a separate, uncompleted check. Unlike the original rank-2/rank-3 duplicate proof, the benchmark retains only selected-winner payload hashes, not every candidate's patches.

There is **no repeatable allocated-VRAM reduction**. Series30/Combat and Sully/Combat have exactly unchanged peaks; Sully/Cinema's current peaks are slightly larger; Series30/Cinema has a roughly 3 MB reduction in only one repetition and identical peaks in the other two. GPU/CPU placement affects cache realization: Series30/Cinema has 98/100/98 current replays versus 98/98/98 baseline, while Sully/Combat has zero in both versions. The earlier two-pair cache observations remain valid. Planned cache reuse is not measured reuse, and candidate replacement dominates any savings observed here.

#### Independent complete-record audit and regression coverage

New research-only `scripts/h3_search_audit.py`, SHA-256 `7613aa5d37ca9e583ecc97110a2f8dff8e841d530c6cea7a8436b5bc4ed59623`, runs **only after the timed parent exits**. It requires a complete/non-stopped summary and exact scheduled scope; revalidates pinned snapshots/helpers/base header; rehashes all four actual adapter payloads and **200 trial artifacts**; matches result/tuner/log/telemetry receipts; checks native output coverage, six/three-candidate budgets and complete consistent per-prefix decisions; and recomputes medians/comparisons from raw records. It has no GPU, queue, generation, installation or rerun path. Changed-winner/default/repeatability findings remain reportable failures, not silently discarded runs. It explicitly states that worker-generated full-patch hashes cannot be independently rehashed after the patches were released.

The [durable audit](data/2026-09-08-h3-search-benchmark-audit.json), SHA-256 `78160bc4aaea475d1727c994a1f5eb415b4c453010d5f6801e7bdc2b2ad81b4b`, preserves all forty records, raw timings, comparisons, runtime profile and artifact/source identities. All five scoped fidelity gates pass: disconnected output preservation, common score/decision preservation, unchanged-winner payload preservation, measured repeatability and runtime-profile consistency. `quality_claim` and `shipping_qualification_complete` are deliberately **false**.

Nineteen new offline tests cover complete-scope reporting of negative timings, missing/stopped/partial/duplicated/extra runs, changed source/log/tuner/telemetry/receipt files, false saved medians/comparisons, mismatched full-target coverage, non-finite JSON, and default/repeat/runtime/payload regressions that must not become passing qualification. With the existing benchmark tests, **52 pass**. After terminal timing completion, the full suite passes **907 tests, 3 skipped, 155 subtests** (31.44 seconds); the existing pynvml deprecation warning remains. `git diff --check` is clean. No tests or full-adapter rehashing overlapped the timed trials.

**Shipping decision for this prototype:** withhold F08 as-is. The original inactive-field duplicate is real and the mathematical fix is numerically safe on this evidence, but this admission/backfill policy buys extra cost without a changed selected output or demonstrated quality benefit. Do not relabel it as a speed optimization or install it automatically. A further revision would need exact no-op-aware admission/cache evidence and a separately frozen comparison; do not infer the exact >40% skip guard from sampled pair statistics. Production source hashes and the installed node pack remain unchanged in this checkpoint; only research audit code/tests/evidence/documentation were added.

The previous goal turn made concrete progress by preserving the first two complete pair comparisons; this turn closes the full controlled-search measurement/audit gate. The broader goal remains active: held-out all-frame/waveform review, any justified subsequent candidate work and the final goal-wide shipping audit remain incomplete. Method-blind held-out observations and keys have not been changed or opened during the timing study. No commit, push, version bump or installed-node replacement occurred. H3/ComfyUI skills kept GPU work serial and deferred heavy checks until the live parent was terminal.

### First held-out block fully reviewed and unblinded: preserve the control faults

The intervening user-status turn was **no progress/status only**. Detailed review then resumed on the existing artifacts without a GPU job, ComfyUI contact, cache release or node change. The [detailed diagonal seed-41 observations](data/2026-09-08-h3-character-heldout-seed41-frame-waveform-observations.json) now cover **all eighteen candidates / 2,232 frames / 72 unchanged stereo-waveform sheets**, with **42 original-resolution contiguous detail grids**. This includes the previously saved C07–C15 reviews and newly completed C16–C18. All eighteen initial sparse observations remained unchanged.

C16's larger frames resolve the earlier torso/guard movement as preparation, not a third punch: two same-glove attempts and a lowered ending. C17's larger hand frames reveal a small finger reposition after gaze lift, not complete stillness; the ending remains curved fingers at the edge rather than a demonstrated flat palm. C18 shows later flatter placement after gaze/smile, but begins with hand/book contact. Those details were saved before method labels were opened. No sound identity, listening, exact sound onset or synchronization score was invented.

The complete blind record was frozen at SHA-256 `b59d68eb13f243249744d8e6c581fc0d791395b334c654fe9fe2c79643c49f0f`. A read-only check through the unchanged held-out preparation gate revalidated all prerequisite pins, every original/prepared graph, all original and anonymous media identities, complete eighteen-case scope, **108 prepared artifact hashes** and all **42 detail hashes**. It also checked observation counts, frame/contact ranges and null sound fields. Existing decoded-AV equality remains hash-pinned; this checkpoint rehashed media rather than claiming another full AV decode. The [pre-unblind checkpoint](data/2026-09-08-h3-character-heldout-seed41-preunblind-checkpoint.json), SHA-256 `c53501342dc865d26cf5dea074b99223597eb5b3e39a8feb6aca4f84e643b977`, was saved before displaying this block's private mapping. Clock observation immediately after display: **2026-09-08 19:46:12 UTC**. The blind records remain byte-for-byte unchanged afterward; their `method_mapping_read: false` describes their frozen observation stage, not current observer exposure.

The separate [matched comparison](data/2026-09-08-h3-character-heldout-seed41-matched-comparison.json), SHA-256 `8a96cf7756bddb6e157a7848f52a243ba57315c6b340fe6e0979111925674a31`, covers all nine arms for each pair. Both creator references were re-inspected. Every candidate also received the same additional original-resolution start/middle/end view (frames 0/60/123), **eighteen hash-pinned grids**, for side-by-side appearance comparison with its matched character-only control. These post-unblind views supplement, not replace, the full-frame blind action observations. No crop, color/gain/timing adjustment or source-media change was made.

#### Sully / Combat, held-out seed 41

| Arm / blind ID | Visible attempts / glove sequence | Lowered ending | Navy vest |
| --- | --- | --- | --- |
| Base / C14 | 2, far then near | Yes | Yes |
| Character-only / C11 | 8, four far-then-near pairs; last unfinished | No | No |
| Effect-only / C12 | 2, far then near | Yes | Yes |
| Additive / C16 | 2, same glove | Yes | No |
| SLERP / C09 | 3, same glove | Yes | No |
| TIES / C13 | 4, same glove | No | No |
| Stable tuner winner / C07 | 3, same glove; first heavily occluded | Yes | No |
| NP / C06 | 3, same glove; first small/occluded | Yes | No |
| CT / C01 | 3, far-far-near | Yes | Yes |

Counts come from visible extensions and contiguous details, **not waveform maxima**. Partly hidden attempts are not independently certified impacts; far/near ordering is not tracked anatomical handedness. Detailed contact and recovery windows remain in the blind record.

The two alternating-punch successes are **base and effect-only controls**, both depicting long-horned/bearded monsters substantially different from the creator and matched character-only design. The character-only anchor already repeats eight attempts, lacks the vest and misses the lowered ending. Therefore it is wrong to attribute all repetition or missing clothing to merging. Additive reduces the attempt count to two and restores lowering while keeping coarse character features, but loses alternation. NP and the stable winner have three same-glove attempts and no demonstrated identity advantage over additive in this cell. SLERP/TIES also retain the broad character design but have their own count/ending faults.

CT keeps the vest and short-horned green/purple character cues, unlike the base/effect-only long-horned design. Its squarer/flatter muzzle and brow/eye/teeth presentation differ from character-only; pose and expression limit a fine-severity ranking. This is a **partial identity/action trade-off**, not complete identity loss or an unqualified win. It still makes an extra punch. None of the six merged arms meets every requested character/clothing/action condition here. A single subject/prompt/seed cell cannot establish a general method ranking or a specific layer-level cause.

#### Series30 / Cinema, held-out seed 41

Base C03 holds an **open** book throughout; every other arm keeps it closed. Character-only C17 already has both hands at the book and ends with curved fingers at the edge after a small reposition, so these faults are not unique to merging. Additive C10, SLERP C04, stable winner C08, NP C05 and CT C15 show later hand repositioning after gaze/smile and a flatter top-of-book ending; effect-only C18 also does. Initial contact and untracked underside palm contact remain caveats rather than perfect-compliance labels. TIES C02 starts raising the hand before completing the gaze lift and ends with curved edge-resting fingers.

All six merged arms retain a broadly similar long-haired facial presentation. NP and additive are especially close in the matched views. Pose, scale, shadow, hair drape and smile differ from character-only/creator footage; no reliable fine-identity ordering or independent style-quality gain is established. CT's altered lamp/background and two-hand starting position are not themselves identity failures. This cell gives **no clear NP/CT advantage over additive or the stable winner**.

#### Scope, validation and next work

The matched record's complete eighteen-case mapping, control identities, six provenance hashes, original-source links and eighteen additional grid hashes pass a separate read-only integrity check. This proves record alignment and completeness, not perceptual truth. No held-out learned similarity metric, score weights, NP/CT bonus or composite AV score was added. All audio preferences remain unassessed. The earlier waveform observations remain descriptive timing evidence only.

Review progress is now **18/72 detailed held-out cases, one of four blocks**, not complete audiovisual qualification. Other blocks' keys remain unopened. Next: crossed seed 41 under the same initial-observation-before-detail-before-unblind rules, followed by both seed-42 blocks, then the goal-wide shipping audit. F08 remains withheld as-is on its negative timing evidence. Optimizer, experimental formulas, benchmark helper and audit helper hashes revalidate unchanged; latest full suite remains **907 passed / 3 skipped / 155 subtests**, not rerun for data-only review. No installed-node replacement, commit, push or version bump. The goal remains active.

### Crossed held-out seed 41: six detailed blind reviews saved

The preceding user-status turn was **no progress/status only**. This continuation revalidated the next available safe work and resumed the existing CPU preparation handle, execution session 43998. It was terminal **exit 0**, with all eighteen cases prepared; no replacement process, GPU job or ComfyUI contact was started.

All eighteen initial sparse observations had already been saved before this detailed pass in [the crossed blind record](data/2026-09-08-h3-character-crossed-heldout-seed41-blind-observations.json), SHA-256 `3c75118448b780df7eb936c4ad56c511ed3e1791169795a60a2cc55a1cc81ccb`. The unchanged held-out gate revalidated the complete block's policy, plan, initial observations, public/key/AV-verification alignment, original/anonymous media identities, 124-frame source audits and exact prepared/executed graphs. All **108 prepared artifact hashes / 2,232 frames / 72 waveform sheets** pass. Preparation manifest SHA-256: `9231678dacec5af0251a899e96b2ba0e78292de6a36be8ec1820842734a3ef31`. This is complete preparation, not complete inspection or a repeated decoded-AV equality test.

The new [detailed crossed observation record](data/2026-09-08-h3-character-crossed-heldout-seed41-frame-waveform-observations.json), SHA-256 `540da61225c192082c300354b5367b9178ab9141547dc9ccd34304e249460b02`, is explicitly **in progress: C01–C06, 744 frames, 24 waveform sheets, 14 original-resolution contiguous detail grids; twelve candidates pending**. The record pins its sources and preserves initial uncertainties rather than rewriting the sparse observations.

- C01 and C04 each show three attempted punches, far-far-near. Both begin lowering gloves only late in the ending; the final frame still has bent elbows and gloves in front of the torso. Lowering is visible, but a settled lowered pose is not shown. The prompt does **not** require arms fully at the sides, so that is not imposed as an extra condition. C04's small early rise/dip resolves as guard preparation, not a fourth demonstrated punch.
- C02 shows four attempts, far-far-far-near, including an early double extension within one broader action episode. Its gloves lower around frames 85–92 and remain down. The count comes from contiguous glove motion, not eight waveform maxima.
- C03 has an early gaze lift and closed-mouth smile but initial hand/book contact, no demonstrated later flat-hand placement and no vest.
- C05 also omits the vest. Larger frames resolve its later motion as a small hand/finger slide at the book edge, not a clear lifted-hand arrival or demonstrated flat palm.
- C06's larger opening frames correct the sparse uncertainty: the head is frontal but the eyes begin lowered and rise around frames 8–14. The closed book is tilted as a whole and set down early, not opened/closed. The other hand approaches its edge before the smile is fully developed; later flat-palm placement is not established. The vest is absent.

A read-only partial-record check verifies all six case/source/public identities, prerequisite hashes, 124-frame counts, inspected sheet hashes, all fourteen detail hashes and frame ranges, exact detector-maxima lists, null sound fields and the disjoint six-complete/twelve-pending scope. The method key was hashed but its contents were **not displayed**. Prior exposure to calibration and diagonal seed-41 findings is explicitly recorded; this is not an unseen-identity or double-blind human trial. Waveform groups/maxima remain descriptive diagnostics, not sound labels, audible-silence claims or synchronization ratings.

Total saved detailed held-out scope is now **24/72**, with only the earlier eighteen diagonal cases unblinded and matched. Finish C07–C18 before freezing and revealing this block, then review both seed-42 blocks and complete the goal-wide shipping audit. No held-out similarity scoring, fitted weights, ranking/default changes, installation, commit, push or version bump occurred. Optimizer, experimental formula, controlled-benchmark and audit-helper hashes remain unchanged. Latest full suite remains **907 passed / 3 skipped / 155 subtests**, not rerun for data-only observations; source/artifact/JSON integrity and whitespace checks are the relevant validation here. F08 remains withheld as-is and the goal remains active.

### Crossed held-out seed 41: fourteen detailed blind reviews saved

Checkpoint context: **2026-09-08 20:26:32 UTC**. The preceding findings/status turn was **no progress/status only**. Read-only resumption established that C07 had already been saved, giving seven completed cases rather than the six in the preceding documentation checkpoint. This continuation adds **C08–C14: seven cases, 868 frames, 28 unchanged waveform sheets and fifteen original-resolution contiguous detail grids**. No live process was awaited, restarted or replaced, and no GPU job or ComfyUI contact was made.

The [detailed crossed record](data/2026-09-08-h3-character-crossed-heldout-seed41-frame-waveform-observations.json) is now SHA-256 `3a616331c0e38e9b8a2b3006c08496dd02b01d659401720c975e8ec6c1f9fb13`: **fourteen cases / 1,736 frames / 56 waveform sheets / 31 contiguous detail grids saved; C15–C18 pending**. The method mapping remains undisplayed. Initial sparse observations remain unchanged.

- C07, saved before this continuation, and newly reviewed C12 each show five attempts, far-far-far-far-near. Both lower their gloves by roughly frames 85–92 and retain a lowered ending. The first rapid triple is not a single punch just because it belongs to one broader action episode.
- C08 retains a navy vest but has an elongated/angular facial presentation and differently presented horns compared with the creator. Teeth are already exposed at the opening and remain visible; a horn is cropped late. Its late screen-right hand moves away from the book onto the table, rather than newly arriving flat on the cover.
- C09 retains coarse short-horned/soft-muzzle features but no vest. The intact tilted book lowers early; no open spread/page turn is visible. Gaze and small smile follow, but there is no distinct later lifted-hand arrival. Opening hand/book motion must not be counted as the requested post-smile placement.
- C10's larger opening frames resolve a different case: its upper cover is genuinely raised above a page block resting on the table, then closes around frames 7–9. The eyes remain lowered despite the frontal head pose. A smile begins while the eyes are still low, before gaze lift finishes; no later hand arrival is visible. The vest is absent. The small waveform group near cover closure is temporal proximity, not an identified book sound or a synchronization rating.
- C11 shows two clear long attempts plus a short, partly occluded extra far-side reach before the crossing attempt. That middle motion may be a feint/touch rather than a landed punch. Only two clear waveform groups are plotted: they cannot prove exactly two attempted actions, and the absent middle group cannot prove a miss. Gloves lower late and continue moving at the ending.
- C13 retains the vest and shows a later curved-finger reposition onto the book, settling approximately frames 92–98, but a flat palm is unconfirmed. Long pointed horns, ears, projecting muzzle and beard differ markedly from the creator; upper horn/hair cropping and exposed teeth persist. Initial contact by the other hand does not invalidate the later observed finger movement.
- C14 shows two alternating far-near pairs, four attempts rather than two. Guard preparation is not counted as an extra attempt; a separately held guard within each pair is not clear. Gloves lower around frames 86–92 and remain down.

The unchanged held-out scope gate was run again without displaying methods. It revalidated all eighteen original and anonymous video identities, their source audits/executed graphs and frozen prerequisites; all **108 prepared artifact hashes** pass. The partial-record audit verifies fourteen unique sequential IDs, their public pair/source alignment, all 124-frame counts, all inspected sheet and thirty-one detail hashes/ranges, exact anonymous-maxima lists, null audio judgments and the four-item remaining scope. This is integrity evidence, not independent confirmation of the visual judgments. Optimizer, experimental formula, controlled-benchmark and audit-helper hashes revalidate unchanged, as does the earlier frozen diagonal detailed record.

Total saved detailed held-out scope is **32/72**, with **only eighteen cases method-unblinded and matched**. Complete C15–C18, freeze and verify this full block before revealing its key, then perform matched comparisons and both seed-42 blocks. No learned held-out scoring, fitted weights, defaults/ranking changes, installed-node replacement, commit, push or version bump. Latest full suite remains **907 passed / 3 skipped / 155 subtests**, not rerun for this data/document-only pass. Audio identity and perceived synchronization remain unassessed; F08 remains withheld as-is. The full goal remains active.

### Crossed held-out seed 41: detailed block frozen and matched comparisons complete

Checkpoint context: **2026-09-08 20:47:09 UTC**. The preceding achievement/status turn was **no progress/status only**. This continuation completed and saved C15–C18, bringing this block to **18 cases / 2,232 frames / 72 unchanged stereo waveform sheets / 42 original-resolution contiguous detail grids**. C15's previously inspected full-frame sheets were retained; eleven new contiguous detail grids resolved the remaining opening, smile, hand, punch and ending ambiguities. No GPU work, server contact, source-media modification or node-behavior change occurred.

The complete [detailed blind record](data/2026-09-08-h3-character-crossed-heldout-seed41-frame-waveform-observations.json), SHA-256 `3a41e75901e8ef20d151b3fc32f4c148ff1fe5b647c10bd640a7d20ba35eed9e`, was saved at **20:41:51 UTC**. The frozen scope helper and additional read-only checks revalidated all eighteen original/anonymous clip hashes, executed/prepared graphs and original audits, frozen prerequisites, all 108 prepared artifacts, 42 detail hashes/ranges, exact maxima lists and null audio fields. Session 88518 terminated with exit 0. The [pre-unblind checkpoint](data/2026-09-08-h3-character-crossed-heldout-seed41-preunblind-checkpoint.json), SHA-256 `e855fdb4dab3605b820520eba8dc3be61f15aee8c54372730d56165823b8dee4`, was then saved and rehashed before displaying the key. The clock immediately after key display was **20:43:24 UTC**. Initial and detailed blind records remain unchanged after unblinding.

Both creator grids were reinspected. All eighteen original-resolution three-frame rows (frames 0, 60 and 123) were additionally inspected against the matched character-only controls, C10 for Sully/Cinema and C11 for Series30/Combat; these supplement, not replace, the all-frame action review. The independent [matched comparison](data/2026-09-08-h3-character-crossed-heldout-seed41-matched-comparison.json), SHA-256 `f02b6a2f76902cac642afb2f5df37f7b356dff052db03487e47dd3a80a7fa790`, covers all nine arms per pair and pins every supplementary grid.

For **Sully/Cinema**, character-only already omits the vest, starts with a genuinely raised top cover, and lacks a distinct post-smile flat-palm arrival. Additive, SLERP, TIES, stable winner and NP retain coarse short-curved-horn / broad-soft-muzzle features close to that control, but also omit the vest and do not establish the full requested hand sequence. TIES/stable winner keep the book closed throughout; additive closes a raised cover, whereas NP/SLERP lower an intact tilted book. CT is the only merged arm with the vest, but changes facial/horn presentation, keeps exposed teeth, crops an upper horn and later moves a hand away from the cover. Base/effect-only retain the vest while depicting a different long-horned/angular/bearded monster. No arm satisfies the combined identity, wardrobe and action requirements; wardrobe adherence is not an identity score.

For **Series30/Combat**, the scoped attempted-movement results are:

| Arm | Attempts / visible glove pattern | Glove-lowering ending |
| --- | --- | --- |
| Base | Two long attempts plus one short far-side reach; far–far–near, middle contact uncertain | Modest late lowering, still changing at 123 |
| Character-only | Two long attempts plus one short far-side reach; far–far–near, middle may be feint/touch | Late lowering toward abdomen, still changing |
| Effect-only | Four; far–far–far–near | Late lowering toward abdomen, still changing |
| Additive | Five; far–far–far–far–near | Lowered by about 85–92, held down |
| SLERP | Five; far–far–far–far–near | Lowered by about 85–92, held down |
| TIES | Three; far–far–near | Late partial lowering, no settled ending |
| Stable winner | Four; far–far–far–near | Lowered by about 85–92, held down |
| NP | Four; far–near–far–near | Lowered by about 86–92, held down |
| CT | Three; far–far–near | Starts about 119–120, still changing at 123 |

NP has a **local ordering/count trade-off versus additive**: two alternating pairs rather than four far-glove attempts followed by one near-glove attempt, with a settled lowered ending. It still repeats the requested pair and does not restore a separately held guard between each punch. CT/TIES reduce attempt count relative to additive but lower only late. Repetition/extra reach is already present in controls, so merging cannot be blamed for all of it. Anatomical handedness remains inferential, hidden impacts are not independently certified, and lowering does not require straight arms at the sides. All merged arms retain a broadly similar long-haired facial presentation, but dark lighting, hair, blur, pose and expression preclude a reliable fine-identity ranking; CT differs visibly in eye/hair presentation without establishing a quantified identity loss. Waveform groups remain descriptive timing evidence, not punch counts or identified sounds.

The matched-record integrity check passes all eighteen unique pair/method mappings, both character anchors, eighteen supplementary grid hashes, both creator grids and all blind/checkpoint pins. The earlier diagonal detailed record remains at its frozen hash. Optimizer, experimental formula, controlled-benchmark and audit-helper hashes are unchanged. Latest full suite remains **907 passed / 3 skipped / 155 subtests**, not rerun for data/document-only changes; JSON/hash checks and `git diff --check` pass.

Total held-out review is now **36/72 detailed and matched cases, two of four blocks**. Both seed-42 blocks remain visually unreviewed and method-unopened. Next: diagonal seed 42's full eighteen initial blind observations, then all-frame/waveform review, freeze and matched comparison; repeat for crossed seed 42 before the goal-wide shipping audit. No learned held-out similarity scoring, fitted ranker, NP/CT preference, default change, installation, commit, push or version bump. F08 remains withheld as-is on its negative controlled timing results; audio identity and perceived synchronization remain unassessed. The overall goal remains active.

### Diagonal held-out seed 42: detailed block frozen and matched comparisons complete

Checkpoint context: **2026-09-08 21:46:30 UTC**. The immediately preceding user-status turn was **no progress/status only**. This continuation saved the remaining C16–C18 detailed observations, completed integrity checks, froze the block, revealed its mapping and inspected every matched supplementary view. Earlier diagonal seed-42 observations had already reached fifteen saved cases; this section and the summary header now catch up with that authoritative record.

Preparation recovery is preserved explicitly: prior session **11169 terminated with exit 143**, with C01–C03 complete and a partial C04 sheet. The partial was moved recoverably to `character-heldout-sync-seed42/_interrupted-C04-session11169`; no source clip or completed diagnostic was overwritten. Recovery session **75193 terminated with exit 0**, completing only C04–C18 with the unchanged renderer. Prepared manifest SHA-256 `e19b54b7c1154c5ecfed62030fc9e739dd8ebcb994778eeb6fe6e43d40d20fde`; recovery-start SHA-256 `d83e099306ddb72ecdb16a4c8ee004b374502a95bd0da61b1d33aeab6c3d8fe5`. The recovery driver, all three retained cases and the preserved partial sheet revalidate unchanged. This was CPU diagnostic preparation, not new generation or a server restart.

The [initial blind record](data/2026-09-08-h3-character-heldout-seed42-blind-observations.json), SHA-256 `616ad24493baaaa1aa8b50085ba47ac591de41f79dcdbd2db8316950594e6af6`, remains unchanged. The complete [detailed blind record](data/2026-09-08-h3-character-heldout-seed42-frame-waveform-observations.json), SHA-256 `e2b2a7b3eaf7685ad4d9818d275bd72d9f0186afc09954515b6da05c2a9c40be`, was frozen at **21:40:43 UTC**: **18 cases / 2,232 frames / 72 unchanged stereo waveform sheets / 61 original-resolution contiguous detail grids**. Still-frame inspection is not continuous playback.

Read-only validation session **63898 exited 0**: all frozen prerequisites, eighteen original/anonymous media identities, executed/prepared graphs, existing 124-frame audits, 108 prepared artifacts, 61 detail hashes/ranges, exact maxima lists, null audio fields and complete/disjoint record scope pass. The [pre-unblind checkpoint](data/2026-09-08-h3-character-heldout-seed42-preunblind-checkpoint.json), SHA-256 `559ba8952fbe4ee7318e820e2254a51fa6b203bb2a5cc6ae89db47cfb58ab2c3`, was saved and hashed before key display; the clock immediately after display was **21:42:42 UTC**. Existing decoded-AV equality was re-pinned, not recomputed by decoding again.

Both creator grids and all eighteen additional original-resolution start/middle/end rows (frames 0/60/123) were inspected. The [matched comparison](data/2026-09-08-h3-character-heldout-seed42-matched-comparison.json), SHA-256 `0178d7e408a516b10a0e3e7575582d47b5b23ab770e13a0680d9527bc8fc3c23`, covers every arm and pins these supplementary artifacts. Character-only anchors are C10 for Sully/Combat and C15 for Series30/Cinema.

#### Sully / Combat, held-out seed 42

| Arm | Visible attempts / glove pattern | Navy vest | Final recovery / lowering |
| --- | --- | --- | --- |
| Base | 2; far–near | Yes | Lower chest guard, no distinct later lowering |
| Character-only | 4; near–near–near–near; **two figures** | No | Puncher's modest lower-chest recovery; other glove partly hidden |
| Effect-only | 2; far–near | Yes | Lower chest guard, no distinct later lowering |
| Additive | 4; near–near–far–far | No | Direct retreat into lowering; settled ending |
| SLERP | 3; far–far–far | Yes | Held guard then lowering; settled ending |
| TIES | 4; far–far–far–far | No | Held guard then lowering; settled ending |
| Stable winner | 5; all far | No | Partial lowering at 119–123, still moving |
| NP | 5; near–far–far–far–far | No | Direct retreat into lowering; settled ending |
| CT | 3; far–far–near | Yes | Lower chest recovery then lowering; settled ending |

Character-only already depicts an ungloved observer and a red-gloved punching figure; no mirror boundary justifies treating them as reflections. All six merges have one figure. Repetition, duplication and missing clothing are therefore not exclusively merge-induced. Base/effect-only depict a substantially different long-horned, angular/tusked, pale-bearded monster despite their closer action counts.

SLERP **and** CT retain the vest in this seed. Both preserve short-curved-horn green/purple cues while differing in brow, muzzle, eye and tooth presentation; pose and the imperfect two-figure anchor prevent a reliable fine-identity severity ordering. CT's clothing benefit is not exclusive. NP is broadly close to additive in appearance but has a fifth clear extension, not a demonstrated improvement. None meets the complete identity/clothing/two-punch/guard/ending request. Near/far is not certified anatomical handedness; hidden contact remains qualified.

Effect-only has **four main energy groups but two punches**, including groups during pauses. NP has **five visible extensions but four broad energy groups**. These directly show why waveform groups/maxima cannot substitute for attempted-action counts or perceived synchronization.

#### Series30 / Cinema, held-out seed 42

Every arm begins with a genuinely raised or partly open blue cover. Character-only and all six merges fail to establish the requested library, whereas base/effect-only retain visible shelves. Character-only already carries the open-book and setting faults.

Additive, NP and SLERP open the cover farther and end with a wedge/offset cover-to-page alignment; the already-contacting hand follows the cover downward. NP and additive are especially close in the matched appearance views, without an established NP benefit. TIES, stable winner and CT close the book and show a later hovering/flattening hand placement after the smile, clearer than additive/NP here. They still start with an open book and omit the library. Character-only itself shows later finger flattening whose separation from closure is uncertain. Initial contact is not used to erase genuinely later placements.

All six merges retain coarse character-only facial/hair cues, white blouse and full-head/shoulder-outline framing. Hair drape, angle, light and expression vary; no calibrated fine-identity or independent style-quality ranking is claimed. CT's local hand/object-ending advantage is shared with TIES and stable winner, not an overall quality victory.

The separate matched integrity check passes all eighteen unique pair/method mappings, two anchors, eighteen supplementary grid hashes, two creator references and six provenance pins. Both prior seed-41 detailed records and optimizer/experimental-formula hashes remain unchanged. Latest full suite remains **907 passed / 3 skipped / 155 subtests**, not rerun for this data-only pass; JSON/hash checks and `git diff --check` pass.

Total held-out scope is now **54/72 detailed and matched, three of four blocks**. Only crossed seed 42 remains visually unreviewed/method-unopened. Complete that same frozen review protocol before the goal-wide shipping audit. No new GPU job, ComfyUI contact, learned held-out evaluator, fitted ranking, default change, installed-node replacement, commit, push or version bump occurred. F08 remains withheld as-is on its negative timing evidence. Audio identity and perceived synchronization remain unassessed; the overall goal remains active.

### Final held-out block complete: crossed seed 42

All eighteen initial method-blind observations were saved at 21:50:45 UTC after the same 21 chronological samples per clip and both creator references. The original diagnostic preparation session 9774 completed all eighteen cases with exit 0; no restart/recovery or new render was needed. All 2,232 frames, 72 unchanged stereo-waveform sheets and 36 original-resolution contiguous detail grids were then inspected. The [detailed record](data/2026-09-08-h3-character-crossed-heldout-seed42-frame-waveform-observations.json), SHA-256 `92924061c0b0ab4cb35db5cd4b72f582514f62673c7f9b79a9ccbf44ad0aaaa7`, was completed before unblinding at 22:26:01 UTC.

Read-only validator session 50397 exited 0: frozen prerequisites, eighteen original/anonymous source identities, exact executed/prepared graphs, 124-frame audits, 108 prepared artifacts, 36 detail hashes/ranges, complete unique observations and null sound/sync/offset fields all agree. The three earlier frozen detailed records and optimizer/experimental-formula hashes are unchanged. The [pre-unblind checkpoint](data/2026-09-08-h3-character-crossed-heldout-seed42-preunblind-checkpoint.json), SHA-256 `ca45abb2538ae5a9e227d7ddfcedb8fec18f7b871016a462d23feeb34e315345`, was saved and hashed before the key was displayed; immediate post-display clock 22:27:35 UTC. Both blind records remain unchanged.

After unblinding, both creator grids and all eighteen original-resolution f0/f60/f123 rows were inspected for matched comparisons. The [separate matched record](data/2026-09-08-h3-character-crossed-heldout-seed42-matched-comparison.json), SHA-256 `5a28490e2fa945170bfca03970b6c4b3aedb26e68ecbd3eaf4461b5ffdb2c9ce`, passes eighteen unique pair/method mappings, two anchors, eighteen additional grid hashes, two creator references and six provenance pins. These rows supplement, not replace, all-frame action review.

**Sully/Cinema**, matched character-only C10:

| Arm | ID | Wardrobe / character | Book-hand sequence |
| --- | --- | --- | --- |
| Base / effect-only | C08 / C18 | Vest, but long swept horns and angular/bearded face unlike creator/C10; upper-horn cropping | Already grip an intact closed book; set it down later, no independent flat palm |
| Character-only | C10 | Short curved horns/broad soft muzzle, no vest | Hand movement starts during gaze lift before established smile; later curved-finger landing |
| Additive | C13 | Coarse character-only features, no vest | Similar early hand movement before established smile; landing 28–37 |
| SLERP | C04 | Coarse character-only features, no vest | Later lift 54–63 and landing 64–69 after smile; flat palm not certified |
| TIES | C05 | Coarse character-only features, no vest | Fingers unfurl/slide 69–83 with retained edge contact |
| Stable winner | C16 | Coarse character-only features, no vest | Fingers reposition 60–74 with wrist/heel near table |
| NP | C17 | Close to additive/character-only, no vest | Later lift 47–55, hover 56–63, landing 64–68 after smile; still arched fingers |
| CT | C09 | Vest and short curved horns retained; heavier brow, altered horn curl/longer chin; slight horn crop | Sets down an already held closed book 65–88, no independent flat-palm arrival |

All arms retain library, closed book, gaze lift and a closed-mouth smile. Whole-book lowering is not cover closure. NP and SLERP show clearer separated post-smile hand action than additive/character-only in this cell, without demonstrating flat palm, restored wardrobe or better fine identity. CT here preserves more short-horned character cues than crossed seed 41's swept-horn/toothy result; do not generalize the earlier outcome into universal identity collapse. Pose/light confound fine severity. No combined identity/wardrobe/action/framing winner.

**Series30/Combat**, matched character-only C03:

| Arm | ID | Attempted action sequence | Ending |
| --- | --- | --- | --- |
| Base | C06 | Three longer near–far–far extensions; later short hesitation not a certified independent fourth | Guard held, no lowering |
| Character-only | C03 | Two clear near–far extensions with separate guards | Guard held, no lowering |
| Effect-only | C11 | Three near–far–far | Direct retreat/lowering, down 107–123 |
| Additive | C12 | Three near–near–far | Down 93–96 onward, no held full final guard |
| SLERP | C07 | Three near–far–far | Guard held, no lowering |
| TIES | C15 | Three near–near–far | Guard held, no lowering |
| Stable winner | C14 | Four attempted near–far–near–far; only two far contacts confirmed | Guard held, no lowering |
| NP | C01 | Three near–near–far | Asymmetric lowering; both down only 120–123, no held full final guard |
| CT | C02 | Three longer near–far–far, possible additional short reversal in later far episode | Down 95–96 onward, no held full final guard |

Character-only is the only arm with exactly two clear attempts, but still misses lowering. Base/effect controls already repeat. Every merge exceeds the requested two attempts; anatomical left/right remains unconfirmed. Stable winner's two large waveform groups/two confirmed contacts cannot erase its four attempted cycles. NP's earlier seed-41 local count/order advantage over additive does not reproduce: same three-attempt near–near–far pattern here, with later final lowering. CT retains uncertainty around the short reach rather than an invented exact fourth punch.

Merged arms retain broadly similar long-haired facial cues to creator/C03; CT is more frontal/narrower-eyed and TIES more hair/glove-occluded. Smaller fighting faces, pose, light and expression limit fine identity ranking. Improved visibility or different bag color is not proof of identity gain/loss. No merged arm satisfies exact count/order/guard/ending together.

This closes **72/72 detailed and matched held-out cases**, 8,928 chronological frames, across all four blocks. Audio identity, unwanted voices/music, realism and perceived synchronization remain unassessed; waveform diagnostics are not listening. No learned held-out score, fitted ranking, default change, installed-node replacement, commit, push or version bump occurred. Latest tests remain 907 passed / 3 skipped / 155 subtests, not rerun for these data-only observations; JSON/hash integrity and whitespace checks pass. F08 remains withheld. Next is the goal-wide requirement-by-requirement shipping audit, not another generation sweep.

### Goal-wide audit: technical gates pass, original AV gate remains open

The [requirement-by-requirement audit](data/2026-09-08-h3-goal-audit.json), SHA-256 `ca35e983c672b08b9357c50a02ab2d07f515003ddc8abf19fe5d6ed9c0ed0799`, separates completion evidence from unassessed dimensions. This is **not a goal-completion claim**.

Fresh verification on the current worktree:

- Python: **907 passed, 3 skipped, 155 subtests passed**, 32.74 s. The three skips are CUDA-only paths unavailable in the sandbox; the existing pynvml deprecation warning remains.
- JavaScript dynamic migrations: **3 passed**.
- Real ComfyUI CPU round trips: **PASS**, including partial H3 QKV, signed alpha/model/CLIP contributions, AdaLN/bias/norm, LoCon, finite FP32 export, atomic rejection, and dense/native NP/CT save/reload plus metadata.
- The forty-trial search audit reproduces the durable result exactly, rehashing all four adapters and 200 trial artifacts, with all five scoped fidelity gates passing. Negative timing outcomes remain unchanged.
- Final read-only integrity session 68869 exited 0: all **144 original clips**, their exact prepared/executed graphs, successful histories, empty pre-submission queues and stored stream/decode audits revalidate. All four held-out anonymous-media/source alignments, 432 prepared diagnostic artifacts, 181 detail grids and 72 matched identity rows are verified. Nine character-study clips retain above-unit decoded-sample flags; these are not audible-clipping judgments.
- All **24 merged exports'** pinned numerical reports and 312-group summaries revalidate. Current assets retain their frozen inode/size/mtime and same-file installed links. This check does not pretend to rehash all 314 GB of dense exports or repeat GPU numerical calculations. Full hashes were checked at the original freezes.
- Eight earlier F04–F06 follow-up records were independently reconstructed, including actual export hashes and compact numerical summaries. All 60 earlier pilot/AV2 media hashes also revalidate; the 48 AV2 complete render records reproduce exactly.
- `git diff --check` passes. Source HEAD remains `27bedd7`, package 1.8.4. The separately installed pack remains clean at `4811977`; no install, commit, push or version bump occurred.

The ad hoc integrity checker initially assumed the expanded numerical file equaled its compact collected summary, then encountered the first matched block's older schema without embedded creator-grid hashes. The collector and actual schema were inspected; the corrected check rehashes the expanded sources, recomputes aggregates, and uses the independently pinned creator references for that older block. These were checker assumptions, not modified benchmark evidence or repaired production outcomes. The local checker and exact commands/results are pinned in the audit.

| Requirement | Current evidence / disposition |
| --- | --- |
| Reproduced correctness fixes | F01–F07 committed and current regression/loader checks pass |
| Native export efficiency/fidelity | Measured NP/CT size reductions and numerical guards supported; no universal speed claim |
| Staged character benchmark | 144 technical cases; all 72 detailed/matched held-out cases complete |
| Opt-in candidate improvement | F08 numerically safe within tested scope but slower; withhold as-is |
| Ranking quality | Seed-dependent trade-offs; no automatic NP/CT bonus or held-out-fitted evaluator |
| Disconnected/default safety | Four real disconnected controls equal; CPU cache/replay/evaluator regressions pass |
| Earlier AV2 judgments | **Incomplete:** seed04 calibration and seeds11/12 held-out have 36 unrated unique clips / 42 entries |
| Shipping | Technical fixes supported; whole-goal audiovisual qualification still open |

**Technical shipping recommendation:** retain the committed F01–F07 correctness/native-export changes, keep NP/CT opt-in and leave ranking weights/defaults unchanged. Exclude the uncommitted F08 search-admission/backfill prototype from a release as-is: all twelve measured comparisons cost more without changing the selected payload. The offline mapped-storage hook is research plumbing, not an automatically installed dense-export feature. Do not release the entire dirty worktree indiscriminately.

**Remaining gate, not silently narrowed:** the earlier acceptance statement required real remaining AV judgments and matched comparisons, or an explicit user decision to narrow that scope. Only R1's seed03 labels currently exist. Their NP top-pick gaps are one ordinal grade for Cinema and zero for Repair on that one seed, not transferable regret estimates. The later character review cannot fill missing audio labels or establish unwanted voices/music, sound realism or perceived synchronization. Its declared still-frame/waveform scope is complete, but the broader goal remains active.

Next available work is resolving a genuine audio-capable review/input path and the existing AV2 rating gate, with policy/provenance preserved. Do not add another generation sweep or repeat completed timing/character reviews as a substitute. No new audiovisual quality winner, learned ranker or release action is justified by this checkpoint.

### Audio-review capability checkpoint: blocked pending external input

The [capability and resumption record](data/2026-09-08-h3-audio-review-capability.json) preserves the remaining gate without changing the frozen goal audit or AV2 policy. The preceding status turn was **no progress**, not a live-job wait. The same missing AV judgments have persisted through the final technical audit, capability investigation, status report and this recheck. Further autonomous render sweeps or waveform summaries cannot close that gap.

At 2026-09-08 22:54 UTC, only the original R1 calibration ratings were found in the scoped repository/artifact/Downloads checks; their hashes and the imported record are unchanged. The three existing review pages for seed04, seed11 and seed12 are present and rehashed. **36 unique clips / 42 comparison entries remain unrated.** One reviewer's preferences are not objective truth or consensus.

Available integration metadata was inspected under the plugin-management skill's built-in/connected-capability-first procedure. ComfyUI provides inline images, not audio/video understanding. Resolve documents transcription, signal measurements and host-side image review, not a ready independent AV quality judge; no Resolve operation was invoked. Plugin-directory search and suggestion tools are not exposed in this session, so **no directory search was performed** and no claim is made that a suitable plugin does not exist.

A scoped read-only request to the existing Ollama endpoint, outside the sandbox, again exited 7 with connection failure; no service was started. The inspected cached AST and CLAP configurations identify a classifier and embedding model, not a validated AV preference evaluator. This is scoped discovery, not an exhaustive inventory of every local service or model. The earlier native audio probe's explicit audio-input omission remains recorded above; it was not replayed here. **No audio listening, new rating, quality score or synchronization judgment is claimed.**

The goal is blocked pending remaining real ratings or user direction to set up a separate local audio/video evaluator and prospective calibration/acceptance protocol. A new evaluator would need an isolated environment, an explicit model-download scope and calibration-only validation before held-out use. Machine observations must remain separate from human labels; installation or successful inference alone does not satisfy the AV gate. Existing review pages are linked in the capability record. No external clip upload, model download, installation, production edit, test rerun, new render, commit, push or version bump occurred. F08 remains withheld and the earlier technical recommendation is unchanged.

### September 9: authorized local evaluator follow-up

The user approved the isolated model download/setup, resuming the goal. The [new calibration report](2026-09-09-h3-local-av-evaluator.md) and [machine-readable checkpoint](data/2026-09-09-h3-local-av-evaluator-calibration-checkpoint.json) record the installed official Qwen2.5-Omni-7B thinker, verified weights, two failed structured smoke responses, a corrected missing-EOS integration issue and six short modality probes. Audio-only distinguishes original sound from silence on one previously rated clip; joint audio/video incorrectly reports silence for the original on both repetitions. The evaluator is therefore not admitted for AV quality ranking or held-out use. The original AV judgment gate remains open, and no human ratings or production defaults changed. This is concrete setup/diagnostic progress, not whole-goal completion.
