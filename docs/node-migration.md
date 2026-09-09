# 1.8.6: Inline fix and node migration

Inline now records the exact patch tuples appended by each stock LoRA loader
call instead of guessing ownership from strength object identity. Equal
strengths, repeated cached files, disjoint/overlapping targets and branch-only
calls retain separate slots. Unsupported/order-dependent patches stay in place.
Upstream models and shared attachments are not mutated by the inline node;
captured source references are removed from its output clone after stripping.

Captured CLIP bias targets are retained rather than silently skipped when only
their sibling weight appears in the loader alias map. Inline CLIP exports use
verified loader aliases; previously bare targets could silently fail to reload.
Restart ComfyUI after installing the fix so loaders execute with tracking enabled.
Unstamped third-party loaders retain best-effort capture; check the report.

## Removed nodes

| Removed class / display name | Replacement |
| --- | --- |
| `LoRAOptimizer` / LoRA Optimizer (Legacy) | `LoRAOptimizerSimple` / LoRA Optimizer, plus Optimizer Settings and optional Merge Settings |
| `WanVideoLoRAOptimizer` / WanVideo LoRA Optimizer | Native WAN loader returning `MODEL`, then the regular optimizer |
| `MergedLoRAToWanVideo` / Merged LoRA → WanVideo | Native `MODEL` merge/application path; no supported wrapper bridge |

These registrations are removed, not hidden aliases. Existing workflows using
them will report missing nodes until migrated. No personal workflow is rewritten
or deleted. The shared Python merge engine remains for the supported nodes.
`WANVIDEOMODEL` cannot be converted to `MODEL` just by changing socket types.

For the old AutoTuner bridge, set AutoTuner to `tuning_only`. Connect its unchanged
MODEL/CLIP and `tuner_data` to the current optimizer, with the same stack on both.
Leave `settings` unconnected for replay; connected Settings take priority.
Use Merge Selector for another rank. Do not apply the same merge twice by passing
already-merged weights into another full application.

The obsolete frontend bridge and bundled legacy example are removed and remain
recoverable from Git history. Other functional utility nodes and native WAN
support remain.

## Validation scope

Regression tests reproduce equal-strength attribution failures. The separate CPU
integration test uses real stock Load LoRA / Model Only nodes, safetensors and
ComfyUI patch application. It compares inline with file-stack merging and
save/reload for weighted sum, weighted average, SLERP, TIES and per-prefix modes,
including disabling the middle MODEL-only call. These are numerical correctness
checks, not claims of better-looking renders for every merge.

Validation on September 9, 2026, against an isolated copy excluding the unrelated
F08 search prototype: **1,064 Python tests and 69 subtests passed**, three CUDA-only
tests skipped; three JavaScript migrations and all ten stock-loader parity cases
passed. The existing real ComfyUI H3/native-export round trip also passed. Tests
used the user's `13_env_py313` environment in CPU mode, without a server restart.
Optimizer source SHA-256 in that copy:
`54041a1c5c414b61a3c052f3349b8b3268abf122ccf3d181aa2f1fa54e2f36ca`.

The first isolated-copy setup had a patch-format error and left the old source
in place; its failing regressions are not counted as validation of the fix.
The counts above are from the completed rerun after correcting that setup.
