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
import re
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

# A flat pause length between every single line is what made rapid
# back-and-forth exchanges -- a speaker change straight into a short,
# reactive reply -- sound like the incoming host was cutting the first one
# off, even though there is always real silence between the two clips (see
# the "won't let him finish" listener feedback on the 2026-09-20 CTOS
# episode: pocketsphinx word-level analysis of that exact published file
# confirmed zero literal audio overlap and a clean ~0.7s gap at every line
# boundary, so the complaint was about how a short, instant-sounding reply
# READS after only a beat of silence, not a bug in the gap itself).
#
# This gives a noticeably longer beat specifically before a short or
# reactive-sounding reply, and only there. It is pure post-processing on
# top of whatever script the model wrote, so unlike the writing-prompt
# pacing guidance (which the model doesn't always follow -- confirmed on
# the CTOS episode, which still came back 100% strictly alternating even
# after one automatic corrective rewrite), this fix applies identically to
# every episode no matter how the model wrote it.
REACTIVE_OPENERS = (
    "right", "exactly", "that's", "yeah", "okay", "no,", "correct,",
    "true,", "totally,", "yep,", "well,",
)
SHORT_LINE_WORD_THRESHOLD = 12
LONG_PAUSE_SECONDS = 1.3
MICRO_PAUSE_SECONDS = 0.15

# 2026-09-21, ORN episode: Shawn reported the hosts "not letting each other
# finish," with a concrete example -- Alex's very first line got cut off
# mid-word at "it just pos-" and Jordan's reply started almost immediately
# after. Downloading and directly analyzing that exact published file
# confirmed this wasn't the pacing/perception issue from the CTOS episode:
# word-level timing showed Alex's line stopping before its last sentence
# ("...that made at least one analyst cut their price target." never got
# spoken at all), and Jordan's very next line ALSO got cut short before its
# own last sentence -- real dropped content, not a pause problem. A
# whole-episode word-count-vs-duration check across several published
# episodes ruled out a global slowdown/speedup explanation; this looks like
# gpt-4o-mini-tts (an autoregressive, LLM-driven TTS model) occasionally
# stopping generation before it has spoken all of a longer, multi-sentence
# input -- a known category of failure for this kind of TTS model, distinct
# from older parametric TTS engines that always render 100% of their input.
#
# Two independent, code-only defenses, neither of which depends on the
# writer model behaving any differently (that's a separate API call and had
# nothing to do with this bug):
#   1. Send each SENTENCE to the TTS API as its own request, not a whole
#      multi-sentence line -- a short, single-sentence input is much less
#      likely to get cut off than a paragraph.
#   2. After every TTS call, check the resulting clip's duration against a
#      generous floor for how fast that many words could plausibly be
#      spoken, and retry if it's implausibly short. A final whole-episode
#      version of the same check runs again in process_episode() as a last
#      resort before anything is allowed to publish.
MIN_PLAUSIBLE_WPM = 260
MAX_TTS_RETRIES = 2

SENTENCE_ABBREVIATIONS = (
    "U.S.", "U.K.", "Mr.", "Mrs.", "Ms.", "Dr.", "Inc.", "Corp.", "Ltd.",
    "vs.", "etc.", "Jr.", "Sr.", "St.", "Co.",
)


def _sounds_like_a_quick_reply(text):
    words = text.split()
    if not words:
        return False
    if len(words) <= SHORT_LINE_WORD_THRESHOLD:
        return True
    first_two = " ".join(words[:2]).lower().strip(".,")
    return any(first_two.startswith(o.strip(",")) for o in REACTIVE_OPENERS)


def split_into_sentences(text):
    """Splits one line of dialogue into individual sentences, each sent to
    the TTS API on its own -- see the module comment above MIN_PLAUSIBLE_WPM
    for why. Careful not to split on decimal numbers (9.5, 379.2) or common
    abbreviations (U.S., Inc.), since a stock-podcast script is full of
    both.
    """
    text = text.strip()
    if not text:
        return []

    protected = text
    placeholders = []
    for abbr in SENTENCE_ABBREVIATIONS:
        if abbr in protected:
            token = f"\x00ABBR{len(placeholders)}\x00"
            placeholders.append((token, abbr))
            protected = protected.replace(abbr, token)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x00DEC\x00" + m.group(2), protected)

    parts = re.split(r"(?<=[.!?])\s+", protected)

    sentences = []
    for part in parts:
        for token, abbr in placeholders:
            part = part.replace(token, abbr)
        part = part.replace("\x00DEC\x00", ".")
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences if sentences else [text]


def _min_plausible_duration(text):
    word_count = len(text.split())
    return word_count / (MIN_PLAUSIBLE_WPM / 60.0)


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


def tts_sentence(text, voice, out_path, api_key):
    """Wraps tts_line() with the duration-floor retry described above the
    MIN_PLAUSIBLE_WPM constant. Returns the clip's final duration."""
    min_duration = _min_plausible_duration(text)
    last_duration = None
    for attempt in range(1, MAX_TTS_RETRIES + 1):
        tts_line(text, voice, out_path, api_key)
        last_duration = probe_duration_seconds(out_path)
        if last_duration >= min_duration:
            return last_duration
        print(
            f"WARNING: TTS clip came back {last_duration:.1f}s for "
            f"{len(text.split())} words (floor: {min_duration:.1f}s) -- "
            f"looks truncated. Retrying ({attempt}/{MAX_TTS_RETRIES}): "
            f"{text[:80]!r}",
            file=sys.stderr,
        )
    print(
        f"WARNING: still only {last_duration:.1f}s after {MAX_TTS_RETRIES} "
        f"attempts (floor: {min_duration:.1f}s) -- using it anyway rather "
        f"than failing the whole episode over one clip: {text[:80]!r}",
        file=sys.stderr,
    )
    return last_duration


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


def make_silence(path, duration=0.7):
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
    if ep.get("tickers"):
        # Disambiguate same-day-and-type episodes (e.g. several manual
        # weekday_deep_dive backlog runs in one day) with the ticker, and
        # fall back to a short random suffix if that's still not unique.
        ticker_part = re.sub(r"[^A-Za-z0-9]+", "", ep["tickers"][0]).upper()
        if ticker_part:
            slug = f"{slug}-{ticker_part}"
    if os.path.exists(os.path.join(AUDIO_OUT_DIR, f"{slug}.mp3")):
        slug = f"{slug}-{uuid.uuid4().hex[:6]}"
    print(f"Processing {slug} from {script_path}")

    work_dir = os.path.join("/tmp", f"build_{slug}_{uuid.uuid4().hex[:8]}")
    os.makedirs(work_dir, exist_ok=True)

    lines = ep["lines"]
    line_paths = []
    silence_path = os.path.join(work_dir, "silence.mp3")
    make_silence(silence_path)
    silence_long_path = os.path.join(work_dir, "silence_long.mp3")
    make_silence(silence_long_path, duration=LONG_PAUSE_SECONDS)
    silence_micro_path = os.path.join(work_dir, "silence_micro.mp3")
    make_silence(silence_micro_path, duration=MICRO_PAUSE_SECONDS)

    intro_path = os.path.join(work_dir, "intro.mp3")
    outro_path = os.path.join(work_dir, "outro.mp3")
    make_sting(intro_path, work_dir, ascending=True)
    make_sting(outro_path, work_dir, ascending=False)

    line_paths.append(intro_path)
    line_paths.append(silence_path)
    total_word_count = 0
    for i, line in enumerate(lines):
        speaker = line["speaker"].lower()
        voice = config.HOST_VOICES.get(speaker)
        if not voice:
            die(f"Unknown speaker '{speaker}' in {script_path}; add it to HOST_VOICES in config.py")
        total_word_count += len(line["text"].split())

        # Each line is sent to TTS one sentence at a time -- see the
        # module comment above MIN_PLAUSIBLE_WPM for why -- with only a
        # very small breathing gap between sentences of the SAME line,
        # since that's one host continuing one thought, not a real pause.
        sentences = split_into_sentences(line["text"])
        for j, sentence in enumerate(sentences):
            clip_path = os.path.join(work_dir, f"line_{i:04d}_{j:02d}.mp3")
            tts_sentence(sentence, voice, clip_path, api_key)
            line_paths.append(clip_path)
            if j < len(sentences) - 1:
                line_paths.append(silence_micro_path)

        # Pick the pause that follows this line based on what's coming
        # next, not a flat value -- see the module-level comment on
        # _sounds_like_a_quick_reply for why.
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if next_line is not None and next_line["speaker"].lower() != speaker \
                and _sounds_like_a_quick_reply(next_line["text"]):
            line_paths.append(silence_long_path)
        else:
            line_paths.append(silence_path)
    line_paths.append(outro_path)

    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    os.makedirs(NOTES_OUT_DIR, exist_ok=True)
    final_filename = f"{slug}.mp3"
    final_path = os.path.join(AUDIO_OUT_DIR, final_filename)
    concat_mp3s(line_paths, final_path, work_dir)

    duration_seconds = probe_duration_seconds(final_path)
    file_size = os.path.getsize(final_path)

    # Last-resort safety net for the whole episode, on top of the per-
    # sentence retries above: if the finished file is shorter than even
    # implausibly fast speech could explain for this many total words,
    # something was still dropped somewhere in the pipeline. Refuse to
    # publish rather than let a truncated episode reach listeners -- this
    # script stays in pending/ (not moved to state/processed/ below) so
    # the next run retries it from scratch.
    min_plausible_total = total_word_count / (MIN_PLAUSIBLE_WPM / 60.0)
    if duration_seconds < min_plausible_total:
        os.remove(final_path)  # don't leave a broken file behind in docs/
        die(
            f"{script_path}: finished audio is {duration_seconds:.0f}s, "
            f"shorter than {min_plausible_total:.0f}s -- the fastest "
            f"{total_word_count} words could plausibly be spoken. Some "
            "dialogue was likely dropped during audio generation. Refusing "
            "to publish; this episode will be retried on the next run."
        )

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
