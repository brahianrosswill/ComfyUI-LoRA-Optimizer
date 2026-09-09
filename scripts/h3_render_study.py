"""Submit one controlled benchmark to an EXISTING ComfyUI API; never starts it.

No cancellation, history clearing, custom-node installation, or canvas editing.
prepare is local-only; submit explicitly queues one job behind existing work.
"""

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener
import uuid


PROMPTS = {
    "cup": "integrated_multimodal_description: [Shot 1] ASTROCINEMAV01K2T. Live-action, a medium shot of an adult man in a blue work shirt sitting at a wooden kitchen table. A red ceramic cup rests in front of him. He reaches with his right hand, grasps the handle, lifts the cup to his lips, takes one sip, and sets it back in its original position. One continuous shot at normal speed. His left hand stays on the table. Warm light from a shaded lamp and cool window fill reveal natural skin and fabric texture. The camera remains stationary.\n\noverall_soundscape: Quiet room ambience, a small sipping sound, and one ceramic tap when the cup returns to the table. No speech.\n\nnon_diegetic_music: N/A",
    "boxing": "integrated_multimodal_description: [Shot 1] ASTROCINEMAV01K2T. Live-action, a medium-wide side view of an adult boxer in a grey tracksuit and red boxing gloves beside a suspended heavy punching bag in an empty gym. At normal real-time speed he throws two distinct straight punches into the bag, returns both gloves to his guard after each punch, then takes one step backward and stops. The bag swings away from each impact and back toward him. His feet remain grounded. No injuries or blood. One continuous stationary shot, warm overhead lamps, cool window fill, natural fabric texture.\n\noverall_soundscape: Two distinct dull impacts synchronized with glove contact, shoe squeaks, light chain rattling, and quiet gym ambience. No speech.\n\nnon_diegetic_music: N/A",
    "padwork": "integrated_multimodal_description: [Shot 1] ASTROCINEMAV01K2T. Live-action, a stationary full-body side view of two adult athletes facing each other in an empty gym. On the left, an athlete wears a navy training shirt, black trousers and red boxing gloves. On the right, a trainer in a light grey shirt holds two black training pads. At normal real-time speed the athlete delivers one left straight punch to the raised pad, returns that glove to the guard, then lifts the right knee and delivers one right front kick to the lower pad. The trainer braces against each contact. The athlete puts the right foot back on the floor and both people settle into their original stance. Their clothing and positions remain consistent. One continuous shot, warm overhead lamps and cool window fill, natural fabric texture. No injuries or blood.\n\noverall_soundscape: One short padded thump at the glove contact, then a deeper padded thump at the foot contact. Quiet gym ambience, brief shoe squeaks and clothing rustle. No speech.\n\nnon_diegetic_music: N/A",
}


def request(base, path, body=None):
    data = json.dumps(body, allow_nan=False).encode() if body is not None else None
    req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        response = build_opener(ProxyHandler({})).open(req, timeout=60)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read(16000).decode(errors='replace')}") from exc
    with response:
        return json.load(response)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def graph(prompt, seed, loras, prefix, profile="remote", width=None, height=None):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    result = {
        "1": node("UNETLoader", unet_name="MiniMax-H3/minimax_h3_fl2va_pruned_int8_convrot.safetensors", weight_dtype="default"),
        "2": node("CLIPLoader", clip_name="MiniMax-H3/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", type="minimax", device="default"),
        "3": node("VAELoader", vae_name="MiniMaw-H3/minimax_h3_video_vae_fp16.safetensors"),
        "4": node("VAELoader", vae_name="MiniMaw-H3/minimax_h3_audio_vae_fp32.safetensors"),
        "5": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=prompt, width=832, height=480, length=124),
        "6": node("RandomNoise", noise_seed=seed),
        "7": node("KSamplerSelect", sampler_name="res_multistep"),
        "8": node("BasicScheduler", model=["1", 0], scheduler="simple", steps=20, denoise=1.),
        "9": node("BasicGuider", model=["1", 0], conditioning=["5", 0]),
        "10": node("SamplerCustomAdvanced", noise=["6", 0], guider=["9", 0], sampler=["7", 0], sigmas=["8", 0], latent_image=["5", 1]),
        "11": node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": node("CreateVideo", images=["11", 0], audio=["12", 0], fps=24.),
        "14": node("SaveVideo", video=["13", 0], filename_prefix=prefix, format="mp4", **{
            "format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16.}),
    }
    previous = "1"
    for index, (filename, strength) in enumerate(loras, 20):
        key = str(index)
        result[key] = node("LoraLoaderModelOnly", model=[previous, 0], lora_name=filename, strength_model=float(strength))
        previous = key
    for key in ("8", "9"):
        result[key]["inputs"]["model"] = [previous, 0]
    if profile == "local":
        result["1"]["inputs"]["unet_name"] = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        result["2"]["inputs"]["clip_name"] = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        result["3"]["inputs"]["vae_name"] = "minimax_h3_video_vae_fp16.safetensors"
        result["4"]["inputs"]["vae_name"] = "minimax_h3_audio_vae_fp32.safetensors"
    result["5"]["inputs"]["width"] = width or (640 if profile == "local" else 832)
    result["5"]["inputs"]["height"] = height or (384 if profile == "local" else 480)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "watch"))
    parser.add_argument("--base")
    parser.add_argument("--profile", choices=("remote", "local"), default="remote")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prompt", choices=PROMPTS, default="cup")
    parser.add_argument("--seed", type=int, default=2026090801)
    parser.add_argument("--lora", nargs=2, action="append", default=[], metavar=("FILENAME", "STRENGTH"))
    args = parser.parse_args()
    if args.action == "prepare":
        args.base = args.base or ("http://127.0.0.1:8189" if args.profile == "local" else "http://192.168.1.12:8188")
        width = args.width or (640 if args.profile == "local" else 832)
        height = args.height or (384 if args.profile == "local" else 480)
        if any(n < 128 or n % 32 for n in (width, height)):
            raise ValueError("Use dimensions at least 128 and divisible by 32")
        args.out.mkdir(parents=True, exist_ok=False)
        prefix = "h3_autotuner_study_20260908/" + args.out.name
        manifest = {"base": args.base, "client_id": "h3-study-" + uuid.uuid4().hex,
            "prompt_name": args.prompt, "seed": args.seed, "loras": args.lora,
            "output_prefix": prefix, "turbo": False, "steps": 20,
            "profile": args.profile, "width": width, "height": height, "length": 124, "quality_claim": None}
        save(args.out / "manifest.json", manifest)
        save(args.out / "prompt_api.json", graph(PROMPTS[args.prompt], args.seed, args.lora, prefix,
                                                args.profile, width, height))
        (args.out / "prompt.txt").write_text(PROMPTS[args.prompt])
        print(str(args.out))
        return
    manifest = json.loads((args.out / "manifest.json").read_text())
    base = manifest["base"]
    if args.action == "submit":
        if (args.out / "submission.json").exists():
            raise RuntimeError("Already submitted; use a new experiment directory for a rerun")
        queue = request(base, "/queue")
        save(args.out / "queue_before.json", {k: [x[1] for x in v] for k, v in queue.items()})
        save(args.out / "system_stats.json", request(base, "/system_stats"))
        submitted_at = time.time()
        result = request(base, "/prompt", {"prompt": json.loads((args.out / "prompt_api.json").read_text()),
                                          "client_id": manifest["client_id"]})
        save(args.out / "submission.json", {**result, "submitted_at": submitted_at})
        print(json.dumps(result))
        return
    submitted = json.loads((args.out / "submission.json").read_text())
    prompt_id = submitted["prompt_id"]
    if args.action == "watch":
        from websockets.sync.client import connect
        ws = connect(base.replace("http", "ws", 1) + "/ws?clientId=" + manifest["client_id"],
                     proxy=None, max_size=16 * 1024**2)
        try:
            history = request(base, "/history/" + prompt_id)
            while True:
                if prompt_id in history:
                    break
                try:
                    message = ws.recv(timeout=30)
                except TimeoutError:
                    print("Waiting for benchmark " + prompt_id, flush=True)
                    history = request(base, "/history/" + prompt_id)
                    continue
                if isinstance(message, str):
                    message = json.loads(message)
                    if message.get("data", {}).get("prompt_id") == prompt_id:
                        if message.get("type") != "progress_state":
                            print(json.dumps(message), flush=True)
                        if message.get("type") in ("execution_success", "execution_error", "execution_interrupted") or (
                            message.get("type") == "executing" and message["data"].get("node") is None
                        ):
                            history = request(base, "/history/" + prompt_id)
        finally:
            ws.close()
    else:
        history = request(base, "/history/" + prompt_id)
    if prompt_id in history:
        save(args.out / "history.json", history[prompt_id])
        print(json.dumps({"prompt_id": prompt_id, "status": history[prompt_id].get("status"),
                          "outputs": history[prompt_id].get("outputs")}))
    else:
        queue = request(base, "/queue")
        print(json.dumps({"prompt_id": prompt_id, "state": next((k for k,v in queue.items() if any(x[1] == prompt_id for x in v)), "not_in_queue_or_history")}))


if __name__ == "__main__":
    main()
