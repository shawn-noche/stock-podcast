#!/usr/bin/env python3
"""Turns pending episode scripts into finished, published audio.

Reads every JSON file in pending/, generates per-line TTS audio via the
OpenAI API, stitches it into a single MP3 with a synthesized intro/outro
sting, writes show notes, updates the episode manifest, and moves the
processed script into state/processed/.

Runs inside GitHub Actions, where normal internet access is available and
OPENAI_API_KEY is provided as a repository secret.
"""
import glob
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import requests

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PENDING_DIR = os.path.join(ROOT, "pending")
PROCESSED_DIR = os.path.join(ROOT, "state", "processed")
AUDIO_OUT_DIR = os.path.join(ROOT, "docs", "episodes", "audio")
NOTES_OUT_DIR = os.path.join(ROOT, "docs", "episodes", "notes")
MANIFEST_PATH = os.path.join(ROOT, "state", "episodes.json")

OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def tts_line(text, voice, out_path, api_key):
    resp = requests.post(
        OPENAI_TTS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": config.TTS_MODEL,
            "voice": voice,
            "input": text,
            "format": config.TTS_FORMAT,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        die(f"TTS request failed ({resp.status_code}): {resp.text[:500]}")
    with open(out_path, "wb") as f:
        f.write(resp.content)


def make_tone(path, freq, duration=0.16, volume=0.25):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}",
            "-af", f"volume={volume},afade=t=in:d=0.02,afade=t=out:st={max(duration-0.03,0)}:d=0.03",
            "-ar", "44100", path,
        ],
        check=True,
    )


def make_sting(out_path, work_dir, ascending=True):
    notes = [523.25, 659.25, 783.99] if ascending else [783.99, 659.25, 523.25]
    tone_paths = []
    for i, freq in enumerate(notes):
        p = os.path.join(work_dir, f"tone_{ascending}_{i}.mp3")
        make_tone(p, freq)
        tone_paths.append(p)
    concat_list = os.path.join(work_dir, f"sting_{ascending}_list.txt")
    with open(concat_list, "w") as f:
        for p in tone_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c", "copy", out_path,
        ],
        check=True,
    )


def make_silence(path, duration=0.35):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(duration), path,
        ],
        check=True,
    )


def concat_mp3s(paths, out_path, work_dir):
    # Each line/sting/silence clip is its own independently-encoded MP3
    # with its own internal timestamps starting at 0. Concatenating those
    # with "-c copy" (stream copy, no re-encoding) just splices the raw
    # compressed frames together without fixing up timing, which is what
    # produced the repeated "non monotonically increasing dts" warnings --
    # and can cause audible clicks/glitches at the seams between clips.
    # Decoding every clip and re-encoding once (no "-c copy") lets ffmpeg
    # regenerate clean, continuous timestamps for the whole file, and we
    # fold the loudness normalization into this same pass instead of a
    # separate second encode.
    concat_list = os.path.join(work_dir, "final_list.txt")
    with open(concat_list, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", concat_list,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a", "libmp3lame", "-b:a", "128k",
            out_path,
        ],
        check=True,
    )


def probe_duration_seconds(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def format_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return []


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def process_episode(script_path, api_key):
    with open(script_path) as f:
        ep = json.load(f)

    date_str = ep.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = f"{date_str}-{ep.get('episode_type', 'episode')}"
    print(f"Processing {slug} from {script_path}")

    work_dir = os.path.join("/tmp", f"build_{slug}_{uuid.uuid4().hex[:8]}")
    os.makedirs(work_dir, exist_ok=True)

    lines = ep["lines"]
    line_paths = []
    silence_path = os.path.join(work_dir, "silence.mp3")
    make_silence(silence_path)

    intro_path = os.path.join(work_dir, "intro.mp3")
    outro_path = os.path.join(work_dir, "outro.mp3")
    make_sting(intro_path, work_dir, ascending=True)
    make_sting(outro_path, work_dir, ascending=False)

    line_paths.append(intro_path)
    line_paths.append(silence_path)
    for i, line in enumerate(lines):
        speaker = line["speaker"].lower()
        voice = config.HOST_VOICES.get(speaker)
        if not voice:
            die(f"Unknown speaker '{speaker}' in {script_path}; add it to HOST_VOICES in config.py")
        line_mp3 = os.path.join(work_dir, f"line_{i:04d}.mp3")
        tts_line(line["text"], voice, line_mp3, api_key)
        line_paths.append(line_mp3)
        line_paths.append(silence_path)
    line_paths.append(outro_path)

    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(NOTES_OUT_DIR, exist_ok=True)
    final_filename = f"{slug}.mp3"
    final_path = os.path.join(AUDIO_OUT_DIR, final_filename)
    concat_mp3s(line_paths, final_path, work_dir)

    duration_seconds = probe_duration_seconds(final_path)
    file_size = os.path.getsize(final_path)

    notes_filename = f"{slug}.txt"
    notes_path = os.path.join(NOTES_OUT_DIR, notes_filename)
    transcript_lines = [f"{line['speaker'].capitalize()}: {line['text']}" for line in lines]
    with open(notes_path, "w") as f:
        f.write(ep.get("title", slug) + "\n\n")
        f.write(ep.get("description", "") + "\n\n")
        f.write("--- Transcript ---\n\n")
        f.write("\n\n".join(transcript_lines))

    manifest = load_manifest()
    guid = ep.get("guid") or str(uuid.uuid4())
    manifest = [e for e in manifest if e.get("guid") != guid]  # replace if re-processed
    manifest.append({
        "guid": guid,
        "title": ep.get("title", slug),
        "description": ep.get("description", ""),
        "date": date_str,
        "pub_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "episode_type": ep.get("episode_type", "episode"),
        "tickers": ep.get("tickers", []),
        "audio_filename": final_filename,
        "notes_filename": notes_filename,
        "duration_seconds": duration_seconds,
        "duration_formatted": format_duration(duration_seconds),
        "file_size_bytes": file_size,
    })
    manifest.sort(key=lambda e: e["pub_datetime_utc"], reverse=True)
    save_manifest(manifest)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, os.path.basename(script_path))
    os.replace(script_path, processed_path)

    print(f"Done: {final_filename} ({format_duration(duration_seconds)}, {file_size} bytes)")


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        die("OPENAI_API_KEY environment variable is not set")

    pending_files = sorted(glob.glob(os.path.join(PENDING_DIR, "*.json")))
    if not pending_files:
        print("No pending episodes found. Nothing to do.")
        return

    for script_path in pending_files:
        process_episode(script_path, api_key)


if __name__ == "__main__":
    main()
