"""Prepare all 18 held-out frame/waveform inspectors without exposing methods.

The frozen calibration entry points stay calibration-only. This separate gate
implements the predeclared held-out policy and reuses their unchanged decoder,
PTS, stereo-envelope and sheet math. No sampling, gain or timing changes.
"""
import argparse
import importlib.util
import json
from pathlib import Path

try:
    from . import h3_av_sync as original
    from .h3_benchmark import digest, save_new
    from .h3_character_benchmark import expected_graph
    from .h3_av_review import output_video
except ImportError:
    import h3_av_sync as original
    from h3_benchmark import digest, save_new
    from h3_character_benchmark import expected_graph
    from h3_av_review import output_video

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / 'docs/research/data/2026-09-08-h3-character-heldout-policy.json'
POLICY_SHA = '40da10ebc71044f09ded4f7761388a367ac145116f45a34ed7088291b99bb760'
ENGINE_SHA = '0db0ed94c6e97709ae774ade4944b768cc60586e3ef00117301f0d8adba5f0af'


def load(path):
    return json.loads(Path(path).read_text())


def engine():
    if digest(original.__file__) != ENGINE_SHA:
        raise ValueError('Frozen frame/waveform engine changed')
    name = ((original.__package__ + '.') if original.__package__ else '') + '_h3_heldout_renderer'
    spec = importlib.util.spec_from_file_location(name, original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Correct the legacy heading only in this private renderer instance.
    module.PAGE = module.PAGE.replace('Calibration diagnostic, not a blind rating page.',
                                      'Held-out diagnostic, not a human rating page.')
    return module


def indexed(rows, field):
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Repeated review/job identifier')
    return result


def prepare_scope(plan_path, observations_path, review_dir, policy_path=POLICY):
    """Validate the entire block before creating any diagnostic output."""
    if digest(policy_path) != POLICY_SHA:
        raise ValueError('Frozen held-out policy changed')
    policy = load(policy_path)
    for name, sha in policy['calibration_evidence_sha256'].items():
        if digest(Path(policy_path).parent / name) != sha:
            raise ValueError('Frozen calibration evidence changed')
    for name, sha in policy['review_helper_sha256'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError('Frozen review helper changed')
    plan_sha = digest(plan_path)
    if not any(plan_sha == p['sha256'] and Path(plan_path).resolve() == (REPO / p['path']).resolve()
               for p in policy['plans']):
        raise ValueError('Plan is outside the frozen held-out scope')
    plan, obs = load(plan_path), load(observations_path)
    key_path, public_path = review_dir / 'private-key.json', review_dir / 'public.json'
    key, public = load(key_path), load(public_path)
    verification_path = review_dir / 'media-verification.json'
    verification = load(verification_path)
    seed = obs['seed']
    if (seed not in policy['seeds'] or key['seed'] != seed
            or any(x['stage'] != 'heldout' or x['plan_sha256'] != plan_sha for x in (obs, key, public))
            or any(x['review_id'] != obs['review_id'] for x in (key, public, verification))
            or obs['heldout_policy_sha256'] != POLICY_SHA
            or obs['method_key_read_before_record'] is not False
            or obs['public_manifest_sha256'] != digest(public_path)
            or Path(obs['review_directory']).resolve() != review_dir.resolve()
            or verification['unique_clips'] != 18
            or verification['verifier_sha256'] != policy['review_helper_sha256']['h3_av_review.py']):
        raise ValueError('Held-out review provenance mismatch')
    jobs = indexed([j for j in plan['jobs'] if j['stage'] == 'heldout' and j['seed'] == seed], 'id')
    cases = indexed(key['cases'], 'blind_id')
    visible, observed = indexed(public['cases'], 'blind_id'), indexed(obs['observations'], 'blind_id')
    verified = indexed(verification['clips'], 'blind_id')
    ids = {f'C{i:02d}' for i in range(1, 19)}
    if (len(jobs) != 18 or any(set(x) != ids for x in (cases, visible, observed, verified))
            or len({c['job_id'] for c in cases.values()}) != 18
            or {c['job_id'] for c in cases.values()} != set(jobs)):
        raise ValueError('A complete disjoint 18-case block is required')
    provenance = dict(plan_sha256=plan_sha, review_id=obs['review_id'], stage='heldout', seed=seed,
        heldout_policy_sha256=POLICY_SHA, observations_sha256=digest(observations_path),
        review_key_sha256=digest(key_path), public_manifest_sha256=digest(public_path),
        media_verification_sha256=digest(verification_path), wrapper_sha256=digest(__file__),
        evidence_kind='assistant_frame_observations', human_ratings_supplied=False,
        identifier_kind='review_blind_id_not_execution_job_id')
    prepared = []
    for blind in sorted(ids):
        case = cases[blind]
        job = jobs[case['job_id']]
        if (any(case[k] != job[k] for k in ('pair', 'variant', 'stage', 'seed', 'prompt'))
                or visible[blind]['pair'] != job['pair'] or observed[blind]['pair'] != job['pair']
                or verified[blind]['video_and_audio_identical'] is not True
                or any(x['media_sha256'] != case['media_sha256'] for x in (visible[blind], verified[blind]))
                or digest(review_dir / f'{blind}.mp4') != case['media_sha256']):
            raise ValueError(f'Review identity mismatch for {blind}')
        run = Path(plan['root']) / job['id']
        history = load(run / 'history.json')
        expected = expected_graph(plan, job)
        if history['prompt'][2] != expected or load(run / 'prompt_api.json') != expected:
            raise ValueError(f'Frozen graph mismatch for {blind}')
        video = output_video(history)
        source_sha = digest(video)
        audit = load(run / 'media-audit/metrics.json')
        if (source_sha != case['original_sha256'] or source_sha != audit['video_sha256']
                or audit['decoded_frames'] != 124):
            raise ValueError(f'Original media/audit mismatch for {blind}')
        prepared.append(dict(blind_id=blind, video=video, source_sha256=source_sha))
    return prepared, provenance


def build(plan_path, observations_path, review_dir, out, policy_path=POLICY):
    if out.exists():
        raise FileExistsError(out)
    cases, provenance = prepare_scope(plan_path, observations_path, review_dir, policy_path)
    renderer = engine()
    out.mkdir(parents=True)
    completed = []
    for case in cases:
        blind = case['blind_id']
        target = out / blind
        try:
            result = renderer.inspect_video(case['video'], case['source_sha256'], target,
                dict(provenance, job_id=blind, blind_id=blind))
        except Exception:
            # ffmpeg exceptions can contain a method-named original path.
            raise RuntimeError(f'Diagnostic extraction failed for {blind}; retain partial output for private diagnosis') from None
        if result['frames'] != 124:
            raise ValueError(f'Unexpected decoded frame count for {blind}')
        sync = load(target / 'sync.json')
        artifacts = ['sync.json', 'inspect.html', *sync['frame_sheets']]
        completed.append(dict(blind_id=blind, frames=result['frames'],
            artifacts_sha256={name: digest(target / name) for name in artifacts}))
        # Never echo original paths, execution IDs or variant names.
        print(json.dumps(dict(prepared=blind, frames=result['frames'], sheets=result['sheets'])), flush=True)
    save_new(out / 'prepared.json', dict(provenance, candidates=completed,
        detailed_observations_saved=False, method_mapping_exposed=False,
        note='Preparation is not inspection or a quality label. All candidates still require detailed observation before unblinding.'))
    return dict(prepared=len(completed), frames=sum(c['frames'] for c in completed), directory=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.observations, args.review, args.out)))
