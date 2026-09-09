"""Local reference/base/character comparison page; no ratings or model calls.

Unblinded qualification, not a preference study. Videos retain their original
timing and audio. Creator references are deliberately NOT time-synchronized to
our generations because the prompts/workflows and durations need not align.
"""
import argparse
import base64
import html
import json
from pathlib import Path

try:
    from .h3_av_review import output_video
    from .h3_benchmark import digest, save_new
    from .h3_identity_study import completed, expected_graph, SEEDS
except ImportError:
    from h3_av_review import output_video
    from h3_benchmark import digest, save_new
    from h3_identity_study import completed, expected_graph, SEEDS


def script_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build(plan_path, sources, out):
    plan = json.loads(plan_path.read_text())
    if out.exists():
        raise FileExistsError(out)
    media, records, case_media = {}, {}, {}
    def add_media(path):
        sha = digest(path)
        if sha not in media:
            media[sha] = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
            records[sha] = dict(path=str(path), sha256=sha, bytes=path.stat().st_size)
        return sha
    for job in plan["jobs"]:
        run = Path(plan["root"]) / job["id"]
        if not completed(run, expected_graph(plan, job)):
            raise ValueError(f"Incomplete qualification job: {job['id']}")
        history = json.loads((run / "history.json").read_text())
        case_media[(job["prompt"], job["seed"], job["variant"])] = add_media(output_video(history))
    refs = {name: add_media(sources / f"{name}-reference.mp4") for name in plan["adapters"]}
    sections = []
    def card(label, sha):
        return f'<section><h3>{html.escape(label)}</h3><video controls muted preload="metadata" data-media="{sha}"></video></section>'
    for name in plan["adapters"]:
        source = json.loads((sources / f"{name}-source.json").read_text())
        for suffix in ("source", "turn"):
            prompt = f"{name}_{suffix}"
            for seed in SEEDS:
                title = f"{name} · {suffix} · seed {seed}"
                panels = card("Creator reference (different workflow; independent playback)", refs[name])
                panels += card("Local base, no character LoRA", case_media[(prompt, seed, "base")])
                panels += card("Local character LoRA", case_media[(prompt, seed, "character")])
                sections.append(f'<article><h2>{html.escape(title)}</h2><p>Reference: {html.escape(source["source_url"])}</p>'
                    f'<div class="row">{panels}</div><details><summary>Exact local prompt</summary><pre>{html.escape(plan["prompt_text"][prompt])}</pre></details></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 identity qualification</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:16px system-ui;background:#17191d;color:#eee;margin:24px}article{border-top:1px solid #555;margin-top:32px;padding-top:12px}
.row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}video{width:100%;max-height:440px;background:#000}h3{font-size:14px}
pre{white-space:pre-wrap}p{overflow-wrap:anywhere}@media(max-width:850px){.row{grid-template-columns:1fr}}</style>
<h1>H3 character-only qualification</h1><p>Unblinded calibration, not merge results or human preference labels.
All local clips: 640×384, 124 frames, 24 fps, 20 steps, INT8 convrot / FP4, Turbo off.
Base and character clips share prompt and seed. Creator references do not share the exact workflow;
Series30 includes Turbo, enhancement and interpolation. No pixel-perfect reproduction is implied.</p>
<p>Videos start muted. Unmute one to listen; playing another pauses the previous video.
Check facial/detail consistency separately from pose, scene, motion and sound. Black padding in diagnostic contact sheets is not part of the videos.</p>
'''
    page += "".join(sections)
    page += '<script>const media=' + script_json(media) + ''';
const videos=[...document.querySelectorAll('video[data-media]')];
for(const video of videos){video.src=media[video.dataset.media];video.addEventListener('play',()=>{
for(const other of videos)if(other!==video)other.pause();});}
</script></html>'''
    out.mkdir(parents=True)
    with (out / "review.html").open("x") as stream:
        stream.write(page)
    save_new(out / "manifest.json", dict(plan_sha256=digest(plan_path), unblinded=True,
        qualification_only=True, media=list(records.values()), ratings=[], audio_listened=False))
    return dict(page=str(out / "review.html"), unique_media=len(media), cases=len(case_media))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.plan, args.sources, args.out)))
