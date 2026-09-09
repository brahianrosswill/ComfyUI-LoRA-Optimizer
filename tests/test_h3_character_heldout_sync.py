"""Held-out exposure gates; actual decoder/PTS math stays in test_h3_av_sync."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import h3_character_heldout_sync as sync


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def block(tmp_path, monkeypatch):
    root, review = tmp_path / 'runs', tmp_path / 'review'
    review.mkdir()
    jobs, cases, clips = [], [], []
    for i in range(1, 19):
        blind, job_id = f'C{i:02d}', f'private-method-job-{i}'
        job = dict(id=job_id, stage='heldout', seed=41, pair='pair-a' if i <= 9 else 'pair-b',
                   variant=f'private-variant-{i}', prompt='fixture')
        jobs.append(job)
        video = review / f'{blind}.mp4'
        video.write_bytes(blind.encode())
        sha = sync.digest(video)
        cases.append(dict(blind_id=blind, job_id=job_id, media_sha256=sha, original_sha256=sha,
                          **{k: job[k] for k in ('stage', 'seed', 'pair', 'variant', 'prompt')}))
        clips.append(dict(blind_id=blind, media_sha256=sha, video_and_audio_identical=True))
        graph = {'fixture': job_id}
        write(root / job_id / 'history.json', dict(prompt=[None, None, graph], video=str(video)))
        write(root / job_id / 'prompt_api.json', graph)
        write(root / job_id / 'media-audit/metrics.json', dict(video_sha256=sha, decoded_frames=124))
    plan = tmp_path / 'plan.json'
    write(plan, dict(root=str(root), jobs=jobs))
    plan_sha = sync.digest(plan)
    policy = tmp_path / 'policy.json'
    verifier_sha = sync.digest(Path(sync.__file__).with_name('h3_av_review.py'))
    write(policy, dict(plans=[dict(path=str(plan), sha256=plan_sha)], seeds=[41, 42],
                       calibration_evidence_sha256={}, review_helper_sha256={'h3_av_review.py': verifier_sha}))
    policy_sha = sync.digest(policy)
    monkeypatch.setattr(sync, 'POLICY_SHA', policy_sha)
    shared = dict(review_id='test-review', plan_sha256=plan_sha, stage='heldout')
    write(review / 'private-key.json', dict(shared, seed=41, cases=cases))
    write(review / 'public.json', dict(shared, cases=[{k: c[k] for k in ('blind_id', 'pair', 'media_sha256')} for c in cases]))
    write(review / 'media-verification.json', dict(review_id='test-review', unique_clips=18,
                                                  verifier_sha256=verifier_sha, clips=clips))
    obs = tmp_path / 'observations.json'
    write(obs, dict(shared, seed=41, heldout_policy_sha256=policy_sha, method_key_read_before_record=False,
                    public_manifest_sha256=sync.digest(review / 'public.json'), review_directory=str(review),
                    observations=[dict(blind_id=c['blind_id'], pair=c['pair']) for c in cases]))
    monkeypatch.setattr(sync, 'expected_graph', lambda plan, job: {'fixture': job['id']})
    monkeypatch.setattr(sync, 'output_video', lambda history: Path(history['video']))
    return dict(plan_path=plan, observations_path=obs, review_dir=review, policy_path=policy)


def test_complete_scope_preserves_anonymous_identifiers_and_provenance(block):
    rows, provenance = sync.prepare_scope(**block)
    assert [r['blind_id'] for r in rows] == [f'C{i:02d}' for i in range(1, 19)]
    assert provenance['stage'] == 'heldout'
    assert provenance['human_ratings_supplied'] is False
    assert provenance['observations_sha256'] == sync.digest(block['observations_path'])
    assert provenance['review_key_sha256'] == sync.digest(block['review_dir'] / 'private-key.json')


@pytest.mark.parametrize('field,value', [
    ('stage', 'calibration'), ('seed', 42), ('review_id', 'other'),
    ('plan_sha256', 'other'), ('heldout_policy_sha256', 'other'),
    ('method_key_read_before_record', True), ('public_manifest_sha256', 'other'),
    ('review_directory', '/tmp/unrelated'), ('observations', []),
])
def test_mismatched_observations_rejected(block, field, value):
    obs = sync.load(block['observations_path'])
    obs[field] = value
    write(block['observations_path'], obs)
    with pytest.raises(ValueError):
        sync.prepare_scope(**block)


@pytest.mark.parametrize('filename', ['private-key.json', 'public.json', 'media-verification.json'])
def test_duplicate_or_missing_case_in_any_manifest_rejected(block, filename):
    path = block['review_dir'] / filename
    data = sync.load(path)
    field = 'clips' if filename == 'media-verification.json' else 'cases'
    data[field][-1] = data[field][0]
    write(path, data)
    with pytest.raises(ValueError):
        sync.prepare_scope(**block)


@pytest.mark.parametrize('kind', ['policy', 'plan', 'media', 'graph', 'audit', 'false_av', 'wrong_pair', 'wrong_variant'])
def test_changed_frozen_inputs_or_invalid_media_rejected(block, kind):
    job = sync.load(block['plan_path'])['jobs'][0]
    run = Path(sync.load(block['plan_path'])['root']) / job['id']
    if kind in ('policy', 'plan'):
        path = block[kind + '_path']
        path.write_text(path.read_text() + '\n')
    elif kind == 'media':
        (block['review_dir'] / 'C01.mp4').write_bytes(b'changed')
    elif kind == 'graph':
        write(run / 'prompt_api.json', {'changed': True})
    elif kind == 'audit':
        data = sync.load(run / 'media-audit/metrics.json')
        write(run / 'media-audit/metrics.json', dict(data, decoded_frames=123))
    elif kind == 'false_av':
        path = block['review_dir'] / 'media-verification.json'
        data = sync.load(path)
        data['clips'][0]['video_and_audio_identical'] = False
        write(path, data)
    else:
        path = block['review_dir'] / 'private-key.json'
        data = sync.load(path)
        data['cases'][0]['pair' if kind == 'wrong_pair' else 'variant'] = 'changed'
        write(path, data)
    with pytest.raises(ValueError):
        sync.prepare_scope(**block)


def test_private_renderer_changes_only_heading_not_frozen_module_or_math():
    original_page = sync.original.PAGE
    renderer = sync.engine()
    assert renderer is not sync.original
    assert 'Held-out diagnostic, not a human rating page.' in renderer.PAGE
    assert sync.original.PAGE == original_page
    for name in ('frame_times', 'audio_start', 'envelope', 'burst_candidates', 'make_sheets', 'inspect_video'):
        assert getattr(renderer, name).__code__.co_code == getattr(sync.original, name).__code__.co_code


def test_build_decodes_every_case_without_echoing_private_mapping(block, tmp_path, monkeypatch, capsys):
    calls = []
    def inspect(video, sha, out, provenance):
        calls.append(provenance)
        assert provenance['job_id'] == provenance['blind_id']
        write(out / 'sync.json', dict(provenance, frame_sheets=[]))
        (out / 'inspect.html').write_text('synthetic renderer fixture')
        return dict(frames=124, sheets=0)
    monkeypatch.setattr(sync, 'engine', lambda: SimpleNamespace(inspect_video=inspect))
    out = tmp_path / 'prepared'
    result = sync.build(**block, out=out)
    assert result['prepared'] == 18 and result['frames'] == 2232
    assert len(calls) == 18
    public_text = (out / 'prepared.json').read_text() + capsys.readouterr().out
    assert 'private-method-job' not in public_text and 'private-variant' not in public_text
    assert sync.load(out / 'prepared.json')['detailed_observations_saved'] is False
    with pytest.raises(FileExistsError):
        sync.build(**block, out=out)
    assert len(calls) == 18


def test_extraction_failure_does_not_expose_method_named_original(block, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('/private-method-job-1/variant.mp4')
    monkeypatch.setattr(sync, 'engine', lambda: SimpleNamespace(inspect_video=fail))
    with pytest.raises(RuntimeError, match='C01') as exc:
        sync.build(**block, out=tmp_path / 'failed')
    assert 'private-method-job' not in str(exc.value)
    assert exc.value.__suppress_context__
