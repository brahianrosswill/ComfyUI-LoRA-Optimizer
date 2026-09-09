"""Build a self-contained, method-blinded audiovisual review page.

No model judge or synthetic labels. Videos are stream-copy remuxed to remove
workflow metadata, embedded locally, and left at original loudness. The private
key is a separate file. Calibration and held-out pages must remain separate.
"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import random
import subprocess
import time

try:
    from .h3_benchmark import digest, save_new
except ImportError:
    from h3_benchmark import digest, save_new

OUTPUT_ROOT = Path("/media/unraid/comfyui/output")
DIMENSIONS = ("action", "temporal", "appearance", "audio", "synchronization", "overall")


def output_video(history, output_root=OUTPUT_ROOT):
    if history.get("status", {}).get("status_str") != "success":
        raise ValueError("Only successful outputs can enter review")
    outputs = history.get("outputs", {}).get("14", {}).get("images", [])
    if len(outputs) != 1 or outputs[0].get("type") != "output":
        raise ValueError("Expected one SaveVideo output")
    item = outputs[0]
    path = (output_root / item["subfolder"] / item["filename"]).resolve()
    if not path.is_relative_to(output_root.resolve()) or path.suffix != ".mp4":
        raise ValueError("Unexpected video output path")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def blinded_cases(cases, stage, seed):
    chosen = [dict(c) for c in cases if c["stage"] == stage and c["seed"] == seed]
    if not chosen:
        raise ValueError("No cases for requested split/seed")
    random.SystemRandom().shuffle(chosen)
    for index, case in enumerate(chosen, 1):
        case["blind_id"] = f"C{index:02d}"
    return chosen


def validate_labels(labels, key):
    """Reject unlabeled/partial/fabricated-shape data, not adjudicate honesty."""
    if labels.get("review_id") != key["review_id"] or labels.get("plan_sha256") != key["plan_sha256"]:
        raise ValueError("Labels belong to another frozen review")
    if not isinstance(labels.get("reviewer"), str) or not labels["reviewer"].strip():
        raise ValueError("A reviewer identity is required")
    by_id = {c["blind_id"]: c for c in key["cases"]}
    seen, results = set(), []
    for row in labels.get("ratings", []):
        cid = row.get("blind_id")
        if cid not in by_id or cid in seen:
            raise ValueError("Unknown or repeated clip ID")
        seen.add(cid)
        if row.get("full_video_reviewed") is not True or row.get("audio_listened") is not True:
            raise ValueError("Full video and audio review must both be affirmed")
        if row.get("media_sha256") != by_id[cid]["media_sha256"]:
            raise ValueError("Review media hash mismatch")
        scores = row.get("scores", {})
        if set(scores) != set(DIMENSIONS) or any(type(v) is not int or not 0 <= v <= 4 for v in scores.values()):
            raise ValueError("Every dimension needs an integer 0..4 score")
        results.append({**by_id[cid], "scores": scores, "notes": str(row.get("notes", ""))})
    if not results:
        raise ValueError("No audiovisual ratings supplied")
    return results


def build(plan_path, out, stage, seed):
    plan = json.loads(plan_path.read_text())
    cases = blinded_cases(plan["cases"], stage, seed)
    # Validate all requested cases before creating a partially usable page.
    originals = {}
    for case in cases:
        run = Path(plan["root"]) / case["job_id"]
        originals[case["job_id"]] = output_video(json.loads((run / "history.json").read_text()))
    out.mkdir(parents=True, exist_ok=False)
    review_id = hashlib.sha256(f"{digest(plan_path)}|{time.time_ns()}".encode()).hexdigest()[:20]
    key = dict(review_id=review_id, plan_sha256=digest(plan_path), stage=stage, seed=seed, cases=cases)
    media = {}
    for case in cases:
        job = case["job_id"]
        if job not in media:
            target = out / f"{case['blind_id']}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(originals[job]), "-map", "0:v:0",
                "-map", "0:a:0", "-map_metadata", "-1", "-map_chapters", "-1", "-c", "copy",
                "-movflags", "+faststart", "-n", str(target)], check=True)
            media[job] = dict(sha256=digest(target),
                             data="data:video/mp4;base64," + base64.b64encode(target.read_bytes()).decode())
        case.update(media_sha256=media[job]["sha256"], original_sha256=digest(originals[job]),
                    original_path=str(originals[job]))
    save_new(out / "private-key.json", key)
    public = dict(review_id=review_id, plan_sha256=key["plan_sha256"], stage=stage,
                  cases=[{k: c[k] for k in ("blind_id", "pair", "media_sha256")} for c in cases])
    cards = []
    for pair in dict.fromkeys(c["pair"] for c in cases):
        cards.append(f"<h2>{html.escape(pair.replace('_', ' + '))}</h2>")
        for c in (c for c in cases if c["pair"] == pair):
            cid = c["blind_id"]
            selects = "".join(f'<label>{d.capitalize()} <select data-dim="{d}"><option value="">Unrated</option>' +
                              ''.join(f'<option value="{i}">{i}</option>' for i in range(5)) + '</select></label>'
                              for d in DIMENSIONS)
            cards.append(f'<article data-id="{cid}"><h3>{cid}</h3>'
                f'<video controls preload="none" playsinline src="{media[c["job_id"]]["data"]}"></video>'
                f'<div class="scores">{selects}</div>'
                '<p><label><input type="checkbox" data-video> I reviewed the full motion</label> '
                '<label><input type="checkbox" data-audio> I listened to the full audio</label></p>'
                '<textarea placeholder="Specific faults, event timing, unwanted speech/music, or reasons for preference"></textarea></article>')
    page = PAGE.replace("__CARDS__", "\n".join(cards)).replace("__DATA__", json.dumps(public).replace("<", "\\u003c"))
    page = page.replace("__PROMPT__", html.escape(plan["prompt_text"][cases[0]["prompt"]]))
    with (out / "review.html").open("x") as stream:
        stream.write(page)
    return dict(review=str(out / "review.html"), private_key=str(out / "private-key.json"), cases=len(cases))


def verify_media(root):
    """Verify review fidelity using independent full video/audio decodes."""
    if (root / "media-verification.json").exists():
        raise FileExistsError(root / "media-verification.json")
    key = json.loads((root / "private-key.json").read_text())
    media = {digest(p): p for p in root.glob("C*.mp4")}
    rows, seen = [], set()
    def stream_hash(path, kind):
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{kind}:0",
            "-c", "rawvideo" if kind == "v" else "pcm_f32le", "-f", "hash", "-hash", "sha256", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout.strip()
    for case in key["cases"]:
        sha = case["media_sha256"]
        if sha in seen:
            continue
        seen.add(sha)
        target = media[sha]
        if digest(case["original_path"]) != case["original_sha256"]:
            raise ValueError("Original video changed since review construction")
        hashes = {}
        for kind in ("v", "a"):
            hashes[kind] = stream_hash(case["original_path"], kind)
            if stream_hash(target, kind) != hashes[kind]:
                raise ValueError(f"Review changed decoded {kind} samples: {case['blind_id']}")
        result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(target)],
                                check=True, stdout=subprocess.PIPE, text=True)
        probe = json.loads(result.stdout)
        tags = [probe.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in probe["streams"]]
        if any(k.lower() in ("comment", "prompt", "workflow", "description") for t in tags for k in t):
            raise ValueError("Review media contains potentially identifying workflow metadata")
        rows.append(dict(blind_id=case["blind_id"], media_sha256=sha, decoded_stream_hashes=hashes,
                         metadata_tags=tags, video_and_audio_identical=True))
        print("Verified decoded video/audio: " + case["blind_id"], flush=True)
    record = dict(review_id=key["review_id"], unique_clips=len(rows), clips=rows, listened=False,
                  verifier_sha256=digest(__file__),
                  note="Decode equality verifies review fidelity, not audiovisual quality.")
    save_new(root / "media-verification.json", record)
    return dict(review_id=key["review_id"], unique_clips=len(rows), verified=True)


PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>H3 blinded AV review</title>
<style>body{font:17px system-ui;max-width:1050px;margin:32px auto;padding:0 16px;background:#17191d;color:#eee}
article{padding:16px;margin:20px 0;border:1px solid #666;border-radius:8px}video{width:640px;max-width:100%;display:block}
label{display:inline-block;margin:8px 14px 8px 0}select,button,input,textarea{font:inherit}textarea{width:95%;min-height:60px}
pre{white-space:pre-wrap}button{padding:10px}#notice{white-space:pre-wrap}</style>
<h1>H3 blinded audiovisual review</h1>
<p>Methods are hidden. Compare within each adapter-pair group. Watch at normal speed with sound, one clip at a time.
Loudness is unchanged: quiet is not automatically worse. This page is local; it uploads nothing.</p>
<p>Rate 0 = fails badly, 1 = major faults, 2 = mixed, 3 = good with minor faults, 4 = convincing.
Action: requested action order, contact, physical response and ending. Temporal: stable anatomy, contact and continuous movement.
Appearance: coherent subjects, texture and lighting (not brightness alone). Audio: plausible ambience and impacts without unwanted speech/music.
Synchronization: audible impacts match visible contact. Overall: your audiovisual preference, not a calculated average.</p>
<details><summary>Exact target prompt</summary><pre>__PROMPT__</pre></details>
<p>Reviewer <input id="reviewer" placeholder="Name or stable pseudonym"></p>
__CARDS__
<button id="download">Download reviewed ratings</button><p id="notice"></p>
<script>
const spec=__DATA__;
const draftKey='h3-av-draft-'+spec.review_id;
try {
  const draft=JSON.parse(localStorage.getItem(draftKey)||'null');
  if(draft){
    document.querySelector('#reviewer').value=draft.reviewer||'';
    for(const el of document.querySelectorAll('article')){
      const row=draft.rows[el.dataset.id];if(!row)continue;
      el.querySelectorAll('select').forEach(s=>{s.value=row.scores[s.dataset.dim]??'';});
      el.querySelector('[data-video]').checked=row.video===true;
      el.querySelector('[data-audio]').checked=row.audio===true;
      el.querySelector('textarea').value=row.notes||'';
    }
  }
} catch(e) { /* Some browsers disable local storage for local HTML files. */ }
function saveDraft(){
  const rows={};
  for(const el of document.querySelectorAll('article'))rows[el.dataset.id]={
    scores:Object.fromEntries([...el.querySelectorAll('select')].map(s=>[s.dataset.dim,s.value])),
    video:el.querySelector('[data-video]').checked,audio:el.querySelector('[data-audio]').checked,
    notes:el.querySelector('textarea').value};
  try{localStorage.setItem(draftKey,JSON.stringify({reviewer:document.querySelector('#reviewer').value,rows}));}
  catch(e){document.querySelector('#notice').textContent='Browser draft storage unavailable. Download ratings before closing this page.';}
}
document.addEventListener('input',saveDraft);document.addEventListener('change',saveDraft);
document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>{
  document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause();});
}));
document.querySelector('#download').onclick=()=>{
  try {
    const reviewer=document.querySelector('#reviewer').value.trim();
    if(!reviewer)throw Error('Enter a reviewer name or pseudonym.');
    const ratings=[];
    for(const el of document.querySelectorAll('article')){
      const scores=Object.fromEntries([...el.querySelectorAll('select')].map(s=>[s.dataset.dim,s.value]));
      if(Object.values(scores).every(v=>v===''))continue;
      if(Object.values(scores).some(v=>v===''))throw Error(el.dataset.id+': fill every score, or leave the whole clip unrated.');
      if(!el.querySelector('[data-video]').checked || !el.querySelector('[data-audio]').checked)
        throw Error(el.dataset.id+': affirm full video AND audio review.');
      const item=spec.cases.find(c=>c.blind_id===el.dataset.id);
      ratings.push({blind_id:item.blind_id,media_sha256:item.media_sha256,
        full_video_reviewed:true,audio_listened:true,scores:Object.fromEntries(Object.entries(scores).map(([k,v])=>[k,Number(v)])),
        notes:el.querySelector('textarea').value});
    }
    if(!ratings.length)throw Error('No complete ratings yet.');
    const result={review_id:spec.review_id,plan_sha256:spec.plan_sha256,reviewer,created_at:new Date().toISOString(),ratings};
    const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download='h3-av-ratings-'+spec.review_id+'.json';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
    document.querySelector('#notice').textContent=ratings.length+' ratings exported. Return the JSON file to continue the study.';
  } catch(e){document.querySelector('#notice').textContent=e.message;}
};
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "validate", "verify"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=("calibration", "heldout"), default="calibration")
    parser.add_argument("--seed", type=int, default=2026090803)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--key", type=Path)
    args = parser.parse_args()
    if args.action == "build":
        if args.plan is None or args.out is None:
            parser.error("build requires --plan and --out")
        print(json.dumps(build(args.plan, args.out, args.stage, args.seed)))
    elif args.action == "verify":
        if args.out is None:
            parser.error("verify requires --out pointing to an existing review directory")
        print(json.dumps(verify_media(args.out)))
    else:
        if args.labels is None or args.key is None:
            parser.error("validate requires --labels and --key")
        rows = validate_labels(json.loads(args.labels.read_text()), json.loads(args.key.read_text()))
        print(json.dumps(dict(valid_ratings=len(rows), rows=rows)))


if __name__ == "__main__":
    main()
