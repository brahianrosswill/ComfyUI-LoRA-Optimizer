# Local H3 audiovisual evaluator: calibration-only setup

The user authorized an isolated local audio/video evaluator and model download on September 9, after the original goal was blocked on unavailable listening judgments. The [prospective policy](data/2026-09-09-h3-local-av-evaluator-policy.json) supplements, but does not overwrite, the frozen AV2 policy. Machine observations will not be imported as human ratings. The earlier single-reviewer ratings remain personal calibration evidence, not consensus.

## Selection and isolation

Initial baseline: official [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), revision `ae9e1690543ffd5c0221dc27f79834d0294cba00`. It supports audio/video input and text output. [Qwen3-Omni's documented BF16 memory requirements](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) exceed this GPU's capacity. This is a feasibility choice, not a claim that the older model is the best evaluator. Native [Transformers integration](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_omni) permits loading the thinker separately; no speech generator, remote model code or pickle weights are needed.

The isolated Python 3.12 environment lives under `.h3-study-artifacts/20260909/audio-evaluator/venv`, with model and caches in the same ignored research directory. Dependencies are pinned in [the requirements file](h3-audio-evaluator-requirements.txt); the completed download receipt records all resolved package versions. CUDA wheels follow the [official PyTorch release matrix](https://pytorch.org/get-started/previous-versions/). Existing conda environments and the installed ComfyUI nodes are untouched. Inference is local/offline; public downloads require no credentials and no clips, prompts or ratings are uploaded.

Outside-sandbox hardware checks found a 32 GB RTX 5090 with approximately 28 GB free and 31 GiB available system RAM. The existing ComfyUI process is on port 8189 and its queue was empty. The harness checks that queue before loading and before every case, and refuses to begin with less than 24 GiB free GPU memory. This does not reserve the GPU against races, but no unrelated process is stopped, unloaded or cancelled.

The first environment-creation attempt failed because uv tried to use the read-only default user cache; the following package command consequently found no venv. Both were terminal setup failures. Setting a research-local `UV_CACHE_DIR` fixed creation; 70 pinned/resolved packages then installed successfully. The official 18-file, 22,379,036,209-byte model download is separately tracked and hash-verified before inference.

## Calibration scope and input fidelity

All 12 unique previously rated seed03 clips were rehashed against the frozen source and R1 records; all 14 human comparison entries are retained, including correlated repeated controls. No seed04/11/12 held-out or unrated outputs are fed to the model. The model receives media and a fixed rubric, not paths, merge names, internal scores or R1 labels/notes. Intended actions/sounds are explicitly distinguished from observed evidence.

Audio is decoded to float32 at the original sample rate, channel-averaged without gain normalization, then resampled to 16 kHz with `librosa/soxr_hq`. Native decoded PCM and actual model waveform/feature hashes are recorded. This discards stereo spatial information and does not establish stereo fidelity. The original files remain untouched. Silence and +750 ms delay controls operate on copied model-input waveforms; delay is zero-filled and tail-trimmed, not wrapped. The audio input remains present even in the silent condition.

Video uses the pinned torchvision reader at nominal 4 fps and at most 200,704 pixels per frame. A real 124-frame/24-fps source produced 20 frames at 560 by 336. The reader selects rounded linspace indices including both endpoints, but reports `N / duration` as its sampling rate. Using that rate directly would put the last model-grid frame more than 0.2 seconds before its actual timestamp. The harness instead uses `(N - 1) / sampled_time_span`, records all actual indices and measures residual source-frame rounding error. This corrects input metadata; it does not confer frame-accurate perception on a sparsely sampled model.

The calibration controls require recognition of silence and sensitivity to an imposed audio delay. Fresh-conversation repeats check deterministic consistency. All six ordinal dimensions preserve nulls, ties and missing evidence. Invalid JSON, duplicate keys, nonfinite/boolean/coerced scores and runtime failures remain explicit failures, not repaired or substituted scores. Raw responses are saved before any comparison with R1. A successful model load or plausible caption alone cannot qualify the evaluator.

Current offline harness tests: **37 passed**. These are synthetic input/schema/immutability/queue/timing tests, not model-quality evidence. The original full merger suite has not been rerun for this isolated research setup. Inference outcomes and qualification decisions will be appended after the live download and actual smoke/control jobs finish. No production optimizer changes, install, commit, push or version bump.

## Completed setup and first calibration checkpoint

At 07:12 UTC, the [checkpoint record](data/2026-09-09-h3-local-av-evaluator-calibration-checkpoint.json) establishes that the download and all three inference sessions are terminal. All 18 assets were verified, including the five published safetensors hashes. The setup receipt is `d4a559cd0c4a20f01e26e624f4ce7cdb93a079f8a12af63a04e2dc53f0c71440`. No second model or inference server was started.

The first structured smoke response repeated zero-time event objects to its 768-token limit and failed validation. [Amendment 02](data/2026-09-09-h3-local-av-evaluator-amendment-02.json) bounded the requested event summaries, shortened evidence and added repetition penalty 1.1 under greedy decoding. The second response still failed: malformed nested evidence, invented event times beyond the clip, then follow-up dialogue. Neither response supplies usable scores. Their original runner files, raw responses and hashes are preserved rather than replaced.

A concrete integration issue was then identified in the installed source: the top-level `generation_config.json` contains no EOS ID, and Transformers 4.57.6 loads it over the thinker-derived generation settings. The thinker config and tokenizer both specify EOS 151645; padding is 151643. Passing those verified IDs explicitly corrected termination in all six subsequent short probes. This explains missing stopping, not necessarily all preceding repetition or perception errors. The helper now rejects absent/invalid tokenizer stopping IDs; **42 synthetic tests pass** after adding those regressions.

[Amendment 03](data/2026-09-09-h3-local-av-evaluator-amendment-03.json) froze a simple modality diagnostic without the target scene description or expected sounds. On the already-rated Combat-only source:

| Input | Original audio | Exact silent control |
| --- | --- | --- |
| Audio only | Described loud thumps and a soft click; no speech/music | Reported no audible sound |
| Audio and video | Incorrectly reported no audible sound | Reported no audible sound |
| Audio and video, fresh-conversation repeat | Same incorrect silence response | Same silence response |

All six responses stopped at EOS. The original audio waveform hash is identical across the audio-only and joint paths, its peak is 0.6864, and both paths contain 517 valid audio feature frames. Muted controls contain exact zero samples. These checks confirm differing model responses to controlled inputs, not an assistant listening judgment. The raw captions and actual input shapes are saved.

**Disposition:** the isolated environment works and the audio-only route passes this one-source silence smoke test. The combined AV route fails basic sound-presence calibration, so it is **not qualified for merge ranking, full audiovisual scoring or held-out evaluation**. This is an evaluator failure, not evidence that an H3 merge lacks audio. The two-source delay/silence battery, full twelve-clip six-dimensional calibration and human agreement comparison remain incomplete. No labels are invented to fill them.

The largest observed allocation was 19,083,712,512 bytes (about 17.8 GiB), with no OOM. After completion the existing ComfyUI queue remained empty and GPU use returned to 3,300 MiB, leaving 28,845 MiB free. No unrelated process was stopped. The goal is active again under the user's authorization; next work is diagnosing token/feature mapping and documented separate versus interleaved audio/video input on this same calibration source. Optimizer and experimental-merge hashes are unchanged; no installed-node replacement, commit, push or version bump.

## Joint-input diagnosis and rejection of the Qwen2.5 quality configuration

The [second checkpoint](data/2026-09-09-h3-local-av-evaluator-calibration-checkpoint-02.json) records 24 new local inferences: ten routing traces, six prompt-compatibility cases and the eight-case structured control battery. All three GPU sessions and the independent control auditor finished. No new weights, packages, render jobs or held-out inputs were introduced.

Under [amendment 04](data/2026-09-09-h3-local-av-evaluator-amendment-04.json), a read-only hook captured the actual audio embeddings inserted at text-decoder prefill. All were finite. Every original-audio route produced embedding hash `3edbf1e816227f77381d4a142b5c8d8e3ebae5ca275904b28baf316041b78c5a`; every silent route produced a different hash, `209fcbac17f40f77a08977530bd1135288ea8db0c6ff10860448d8905507e129`. Each had 129 audio tokens; AV cases additionally had 2,400 video tokens and finite visual embeddings. The nonzero audio is genuinely present in the decoder input, not merely in an unused preprocessing field.

Separate video/audio slots and the reader's original time grid did not fix the false-silence response. Running joint AV first and repeating it last produced the same result, so this experiment does not support a prior-muted-request explanation. It does not rule out every possible backend/model issue, but it narrows the next action beyond guessing that the audio was dropped.

[Amendment 05](data/2026-09-09-h3-local-av-evaluator-amendment-05.json) bounded the prompt check to three declared variants. Helpful-system plus a joint see/hear question distinguished original impacts from muted audio. The official system message plus the previous audio-only question hallucinated a swinging-bag sound in exact silence; the official joint variant did not give a reliable audio assessment. All outputs remain saved. Selecting the small positive for a further frozen control test is calibration-only protocol development, not evidence of unbiased generalization.

[Amendment 06](data/2026-09-09-h3-local-av-evaluator-amendment-06.json) requested a flat six-score JSON schema and completed the original two-source control matrix. The corrected stopping IDs remain explicit. Results:

| Gate | Result |
| --- | --- |
| Complete structured output | 8/8 valid, all terminate at EOS |
| Exact input integrity | Source media rehashed; original/muted/delayed model waveforms independently reconstructed; video tensors unchanged within each source |
| Silence detection | Both source controls report no audible sound, speech or music |
| Fresh-conversation repeat | Both repeated Combat-only conditions preserve categorical observations and all six scores |
| +750 ms delay sensitivity | **Fails:** both sources score synchronization 3/4 before and after delay, still claiming aligned impacts |
| Overall admission | **Rejected for quality ranking and held-out evaluation** |

The muted conditions also assign synchronization 2/4 despite missing audible contacts and use wording that confuses absent sound with inability to confirm a visible punch. Those descriptions are retained as limitations, not corrected to improve apparent validity. The dedicated [control auditor](../../scripts/h3_local_av_control_audit.py) recomputes input hashes and the gates; it cannot certify quality from syntactically valid answers or from necessary numerical checks alone. The exact raw responses and all eight input records are pinned in its report.

**No further prompt search for this configuration.** The complete twelve-clip scoring sweep and held-out scoring are not justified by the failed timing gate. The next method to examine is a separate synchronization estimator, initially [Synchformer](https://github.com/v-iashin/Synchformer). [AVGen-Bench](https://github.com/microsoft/AVGen-Bench) uses it separately from audio production-quality and visual-quality measures; that is a research lead, not validation on these H3 clips. Its repository warns that decoder versions can affect results, so any local setup must first pin decoding/timestamps and pass source/delay/silence controls. No Synchformer weights or environment have been installed here.

New research-harness tests: **60 passed**, 0.19 seconds; whitespace checks pass. The original full merger suite was not rerun for these isolated helpers. After the runs, ComfyUI's queue remained empty and GPU use returned to 3,128 MiB, with 29,016 MiB free. Production hashes, installed nodes and release state are unchanged. The goal remains active: these are substantive negative calibration findings and a justified next method, not a claim of completed audiovisual qualification.

The subsequent [Synchformer calibration](2026-09-09-h3-synchformer.md) passed imposed-delay controls but failed to demonstrate useful discrimination across the twelve naturally differing rated clips. It remains a narrow diagnostic, not an admitted automatic merge-ranking signal.
