#!/usr/bin/env python3
"""Researches and writes today's episode script using the Claude API
(with its server-side web search tool), then drops the result into
pending/ for generate_audio.py to turn into audio.

This runs as TWO separate Claude API calls rather than one combined call:

1. RESEARCH: has the web_search tool, produces a short plain-text research
   brief (not the episode itself).
2. WRITE: has NO tools at all, takes the research brief as input, and
   writes the final two-host JSON episode script.

Why split it up: the original single-call design asked the model to
search AND write a ~3000-word script in one continuous exchange. That
combination turned out to be fragile in two ways that kept causing failed
runs -- (a) after using up its search budget the model sometimes kept
retrying refused searches many more times instead of stopping, and each
retry re-sends the whole growing conversation as input tokens, which is
what caused one run to cost $3.23; and (b) by the time a long research
phase finished, there often wasn't enough of the max_tokens budget left
for the model to finish writing the full script, so it got cut off
mid-way with no usable output. Splitting the work in two makes each call
much easier to bound: the research call only has to produce a short brief
(so truncation is far less likely and a failed attempt is cheap to retry),
and the write call structurally CANNOT run away on search, because it
doesn't have the search tool at all.

Runs inside GitHub Actions, which has normal internet access.
"""
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone

import anthropic

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PENDING_DIR = os.path.join(ROOT, "pending")
USED_STOCKS_PATH = os.path.join(ROOT, "state", "used_stocks.json")
CANDIDATE_STOCKS_PATH = os.path.join(ROOT, "state", "candidate_stocks.json")

ANTHROPIC_MODEL = "claude-sonnet-5"
MAX_ATTEMPTS = 3

SCHEMA_INSTRUCTIONS = """
Respond with ONLY a single JSON object, no markdown code fences, no commentary
before or after it. It must match exactly this shape:

{
  "episode_type": "<one of: weekday_deep_dive, saturday_recap, sunday_preview>",
  "date": "<YYYY-MM-DD, today's date>",
  "title": "<a punchy, specific episode title, not generic>",
  "tickers": ["<TICKER>", ...],
  "description": "<2-3 sentence show notes / RSS description, written for a listener deciding whether to play the episode>",
  "lines": [
    {"speaker": "alex", "text": "..."},
    {"speaker": "jordan", "text": "..."}
  ]
}

The "lines" array is the full two-host conversation, in order. Write it as a
natural, energetic back-and-forth between two co-hosts named Alex and
Jordan -- not a lecture, not a script one person reads. Use contractions,
let them react to and build on what the other just said, disagree a little
where reasonable, and vary sentence length.

IMPORTANT for how the audio will sound: each line is recorded separately by
a text-to-speech voice and then stitched together with a short pause
between lines -- there is no way to make two lines play at once. So do NOT
write ultra-short one-or-two-word interjections that depend on quick timing
to land ("Right." "Exactly." "Totally." as their own line), and do NOT have
one host cut off or finish the other's sentence -- that reads fine on paper
but with a pause inserted between every line it sounds like the hosts are
talking over each other. Instead, give each line a complete, clear thought
of at least a full sentence, and have hosts respond to what the other
FINISHED saying, not interrupt mid-thought. Reactions and quick agreement
are fine as long as they're a full beat ("That's exactly the surprising
part to me.") rather than a bare one-word interjection standing alone.

ALSO IMPORTANT, and just as easy to get wrong even with full-sentence lines:
do NOT write this as a rigid back-and-forth where the mic switches to the
other host after literally every single line, all episode long, like a
tennis rally or a formal interview. Real co-hosts don't take turns with
perfect regularity. Several times per episode, have ONE host speak for TWO
lines in a row (occasionally three) to fully develop an explanation, walk
through a set of numbers, or tell a story, before handing off -- especially
when covering earnings figures or a multi-part point. If you look at the
"speaker" values in your own output in order, they should NOT simply
alternate alex/jordan/alex/jordan for the entire episode; there should be
several places where the same speaker appears twice (or three times) back
to back.

Relatedly: do not have almost every line open by immediately agreeing with
or countering what the other host just said ("Right, ...", "Exactly, ...",
"That's a fair point, ..."). That pattern, repeated on nearly every line,
is exactly what makes two co-hosts sound like they're cutting each other
off even when each line is a complete sentence and there's a pause between
them -- it reads as constant instant rebuttal rather than a conversation
with room to breathe. Vary how lines start: let some lines simply continue
the thought, ask a plain follow-up question, introduce a new angle, or sit
with a number for a beat, instead of reflexively reacting to the prior
line. As a rough guide, well under half of all lines should open with a
quick agree/rebuttal word like that.

Do not use stage directions or sound effect cues, only spoken words.

The disclaimer below must appear, spoken in full by one of the hosts, near
the very end of the episode, immediately before the final sign-off line(s):

\"\"\"%s\"\"\"
""" % config.DISCLAIMER


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_used_stocks():
    if os.path.exists(USED_STOCKS_PATH):
        with open(USED_STOCKS_PATH) as f:
            return json.load(f)
    return []


def save_used_stocks(entries):
    os.makedirs(os.path.dirname(USED_STOCKS_PATH), exist_ok=True)
    with open(USED_STOCKS_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def load_candidate_stocks():
    # A pre-approved list of genuinely under-the-radar tickers to work
    # through in order (state/candidate_stocks.json -- edit it any time to
    # add or remove names). Working from a known list instead of asking
    # the model to discover a pick each time skips the most expensive,
    # least predictable part of the research (open-ended screening), so
    # the model can go straight to reading earnings reports and news for
    # a specific company. If the file is missing or every candidate has
    # already been covered, we fall back to letting the model pick its
    # own stock, so the show never gets stuck.
    if os.path.exists(CANDIDATE_STOCKS_PATH):
        with open(CANDIDATE_STOCKS_PATH) as f:
            return json.load(f)
    return []


def next_candidate(candidates, used_tickers):
    for c in candidates:
        ticker = (c.get("ticker") or "").strip().upper()
        if ticker and ticker not in used_tickers:
            return {"ticker": ticker, "company": c.get("company", ticker)}
    return None


def episode_type_for_weekday(weekday):
    # Monday=0 ... Sunday=6
    if weekday == 5:
        return "saturday_recap"
    if weekday == 6:
        return "sunday_preview"
    return "weekday_deep_dive"


# ---------------------------------------------------------------------------
# Stage 1: research (has web_search, produces a short plain-text brief)
# ---------------------------------------------------------------------------

def build_research_prompt(episode_type, today_str, avoid_tickers, assigned_candidate=None):
    if episode_type == "weekday_deep_dive":
        avoid_str = ", ".join(avoid_tickers) if avoid_tickers else "(none yet)"

        if assigned_candidate:
            pick_instruction = f"""1. You have been assigned to cover {assigned_candidate['company']} (ticker:
   {assigned_candidate['ticker']}) today. Start with a quick search to confirm
   it's still a fitting pick: genuinely small-to-mid cap (roughly
   $300M-$10B), not oversaturating financial media this week, not a meme
   stock riding pure hype, and not one of these tickers the show has
   already covered: {avoid_str}. If it still fits, proceed with it for the
   rest of your research. If it clearly no longer fits (acquired,
   delisted, ballooned into mega-cap territory, or having an unusually
   hyped/heavily-covered week), pick a different genuinely
   under-the-radar small-to-mid-cap US stock instead, and note in your
   brief that you swapped picks and why."""
        else:
            pick_instruction = f"""1. Identify ONE US-listed, small-to-mid cap stock (roughly $300M-$10B market
   cap as a guideline, not a hard rule) that is genuinely under-the-radar --
   NOT a mega-cap, NOT a stock that is already saturating financial media
   this week, NOT a meme stock riding pure hype. Do not pick any of these
   tickers, which the show has already covered recently: {avoid_str}."""

        return f"""You are the RESEARCHER for today's ({today_str}) episode of "Under the
Radar," a daily podcast about overlooked, under-the-radar publicly traded
stocks. Your ONLY job right now is research -- someone else will turn your
notes into the actual episode script, so do NOT write any dialogue.

You have a budget of AT MOST {MAX_SEARCHES} web searches total -- be
economical. Combine what you need into broad, well-targeted queries rather
than many narrow ones. IMPORTANT: if a search is ever refused because
you've used up this budget, do NOT try again -- immediately stop searching
and write up your brief using only what you've already gathered.
Repeatedly re-attempting a blocked search wastes a huge amount of money for
no benefit, so treat a refusal as a hard stop, not a retry signal.

Do this research using web search:

{pick_instruction}
2. Find that company's investor relations page and read its two most recent
   quarterly earnings reports or earnings press releases. Note revenue,
   revenue growth rate, margins, segment breakdown if disclosed, guidance,
   and any notable management commentary or surprises.
3. Search for recent news about the company from the last few weeks beyond
   the earnings reports (product news, contracts, insider activity,
   analyst commentary, sector context).

When you're done, write a RESEARCH BRIEF as plain text (not JSON, no
dialogue) using exactly these headings:

TICKER: <the ticker>
COMPANY: <company name>
WHAT IT DOES / HOW IT MAKES MONEY: <plain-language explanation of the
  business model and revenue streams>
RECENT EARNINGS: <what stood out in the two most recent reports -- revenue,
  growth, margins, guidance, surprises>
RECENT NEWS: <notable news from the last few weeks and what it means>
WHY UNDER THE RADAR: <why this stock is flying under the radar right now,
  and what could change that>

Keep it factual and reasonably concise -- this is working notes for a
writer, not the finished episode, so plain prose or bullet points under
each heading is fine."""

    if episode_type == "saturday_recap":
        return f"""You are the RESEARCHER for this Saturday's ({today_str}) weekly recap
episode of "Under the Radar," a podcast about the stock market. Your ONLY
job right now is research -- someone else will turn your notes into the
actual episode script, so do NOT write any dialogue.

You have a budget of AT MOST {MAX_SEARCHES} web searches total -- be
economical, use broad well-targeted queries rather than many narrow ones.
IMPORTANT: if a search is ever refused because you've used up this budget,
do NOT try again -- immediately stop searching and write up your brief
using only what you've already gathered. Repeatedly re-attempting a
blocked search wastes a huge amount of money for no benefit.

Use web search to research the past week (Monday through Friday) in the US
stock market: major index performance, the most significant market-moving
stories, notable earnings from the week, and any macro/economic data
releases that mattered. Also briefly check how the market reacted to
under-the-radar-style stocks generally this week if there's anything
notable.

When you're done, write a RESEARCH BRIEF as plain text (not JSON, no
dialogue) using exactly these headings:

TICKERS: <any specific tickers central to the week, or "none">
INDEX PERFORMANCE: <how major indices did this week and why>
BIGGEST STORIES: <the most significant market-moving stories of the week>
NOTABLE EARNINGS: <earnings that mattered this week>
MACRO / ECONOMIC DATA: <economic releases that mattered>
UNDER-THE-RADAR ANGLE: <how smaller/overlooked stocks fared this week, if
  notable>

Keep it factual and reasonably concise -- this is working notes for a
writer, not the finished episode."""

    # sunday_preview
    return f"""You are the RESEARCHER for this Sunday's ({today_str}) week-ahead preview
episode of "Under the Radar," a podcast about the stock market. Your ONLY
job right now is research -- someone else will turn your notes into the
actual episode script, so do NOT write any dialogue.

You have a budget of AT MOST {MAX_SEARCHES} web searches total -- be
economical, use broad well-targeted queries rather than many narrow ones.
IMPORTANT: if a search is ever refused because you've used up this budget,
do NOT try again -- immediately stop searching and write up your brief
using only what you've already gathered. Repeatedly re-attempting a
blocked search wastes a huge amount of money for no benefit.

Use web search to find what's coming up in the next week: scheduled major
earnings releases, economic data releases (e.g. CPI, jobs report, Fed
meetings), and any other notable calendar events for US markets.

When you're done, write a RESEARCH BRIEF as plain text (not JSON, no
dialogue) using exactly these headings:

TICKERS: <any specific tickers worth mentioning, or "none">
EARNINGS TO WATCH: <major companies reporting next week>
ECONOMIC DATA TO WATCH: <releases/events next week and why they matter>
OTHER NOTABLE EVENTS: <anything else worth flagging>

Keep it short and factual -- this is working notes for a writer, not the
finished episode."""


# ---------------------------------------------------------------------------
# Stage 2: writing (no tools at all -- cannot run away on search)
# ---------------------------------------------------------------------------

def build_writing_prompt(episode_type, today_str, research_notes):
    host_names = "Alex and Jordan"

    if episode_type == "weekday_deep_dive":
        length_guidance = ("Target length: 15-25 minutes of spoken dialogue, roughly "
                            "2600-3800 words total across both hosts.")
        structure = """It must, in this rough order:
   - Cold open that hooks the listener on why this stock is worth 20 minutes
     of their time.
   - Explain in plain, accessible language what the company actually does
     and, specifically, how it makes money -- assume the listener has never
     heard of it.
   - Walk through what stood out in the two most recent earnings reports:
     revenue trends, growth, margins, guidance, surprises.
   - Cover the recent news and what it means going forward.
   - Discuss explicitly why this stock is flying under the radar right now,
     and what could change that.
   - Close with the required disclaimer (verbatim, spoken by one host) and a
     sign-off."""
    elif episode_type == "saturday_recap":
        length_guidance = "Target length: ~20 minutes, roughly 2800-3400 words."
        structure = """Recap the week: what happened, why it mattered, and any threads worth
   remembering, based on the research notes below. This episode has no
   single featured ticker unless the notes name specific stocks central to
   the recap. Close with the required disclaimer (verbatim, spoken by one
   host) and a sign-off."""
    else:
        length_guidance = "Keep this SHORT: target 5-10 minutes, roughly 900-1500 words."
        structure = """Preview what to watch for in the week ahead, based on the research notes
   below, in a brief, upbeat tone. Close with the required disclaimer
   (verbatim, spoken by one host) and a sign-off."""

    return f"""You are the WRITER for today's ({today_str}) episode of "Under the Radar,"
a podcast about the stock market. The two co-hosts are {host_names}. A
researcher has already done the legwork -- your job is to turn their notes
into a natural, engaging two-host conversation. Do not invent facts beyond
what's in the notes below, but you have full freedom in how to phrase and
pace the conversation.

{length_guidance}

{structure}

RESEARCH NOTES:
\"\"\"
{research_notes}
\"\"\"

{SCHEMA_INSTRUCTIONS}"""


def extract_json(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        die(f"Could not find a JSON object in model output. Raw output:\n{text[:2000]}")
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        die(f"Model output was not valid JSON ({e}). Candidate:\n{candidate[:2000]}")


# Rough cost model for logging only (Claude Sonnet 5 + web search pricing,
# as of when this was written -- check platform.claude.com/docs/en/about-claude/pricing
# if these ever look off).
PRICE_INPUT_PER_MTOK = 2.00
PRICE_OUTPUT_PER_MTOK = 10.00
PRICE_PER_1000_SEARCHES = 10.00
MAX_SEARCHES = 8

# Safety net for the research call: the model is instructed to stop
# cleanly once it hits MAX_SEARCHES, but if it ever ignores that and keeps
# re-attempting refused searches anyway, each retry re-sends the whole
# growing conversation as input tokens. If total search *attempts*
# (including refused ones) blows past this ceiling, we abort the
# connection outright. Because this only guards the (short) research call
# now, not a call that also has to write the full script, an abort here is
# cheap to retry.
HARD_SEARCH_ATTEMPT_CEILING = MAX_SEARCHES + 6


class RunawaySearchLoop(Exception):
    pass


class TruncatedResponse(Exception):
    pass


def log_usage_and_cost(usage, label):
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    searches = usage.server_tool_use.web_search_requests if usage.server_tool_use else 0
    cost = (
        input_tokens / 1_000_000 * PRICE_INPUT_PER_MTOK
        + output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOK
        + searches / 1000 * PRICE_PER_1000_SEARCHES
    )
    print(
        f"[{label}] {input_tokens} input tokens, {output_tokens} output tokens, "
        f"{searches} billed web searches -> approx ${cost:.3f}"
    )
    return cost


def call_claude(prompt, api_key, label, use_search, max_tokens):
    # Streaming keeps the connection actively fed with data the whole time
    # instead of sitting idle waiting for one big response -- idle
    # connections like that get silently dropped by network infrastructure
    # in between (this is what caused the very first run to fail with a
    # RemoteDisconnected error).
    client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=900.0)

    tools = None
    if use_search:
        tools = [{"type": "web_search_20260318", "name": "web_search", "max_uses": MAX_SEARCHES}]

    last_error = None
    total_cost = 0.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[{label}] Calling Claude API (attempt {attempt}/{MAX_ATTEMPTS})...")
            text_parts = []
            search_attempts = 0
            stream_kwargs = dict(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if tools:
                stream_kwargs["tools"] = tools
            with client.messages.stream(**stream_kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                        text_parts.append(event.delta.text)
                    elif event.type == "content_block_start" and getattr(event.content_block, "type", None) == "server_tool_use":
                        search_attempts += 1
                        print(f"[{label}]   ...search attempt {search_attempts}")
                        if search_attempts > HARD_SEARCH_ATTEMPT_CEILING:
                            # Close the connection immediately -- stop paying
                            # for further generation on a run that's ignoring
                            # its search budget instead of writing.
                            stream.close()
                            raise RunawaySearchLoop(
                                f"exceeded {HARD_SEARCH_ATTEMPT_CEILING} search attempts "
                                "without stopping; aborted to cap cost"
                            )
                final_message = stream.get_final_message()
            total_cost += log_usage_and_cost(final_message.usage, label)
            if final_message.stop_reason == "max_tokens":
                raise TruncatedResponse(
                    "response was truncated at max_tokens before finishing; "
                    "discarding this attempt"
                )
            return "".join(text_parts), total_cost
        except (RunawaySearchLoop, TruncatedResponse) as e:
            last_error = e
            print(f"[{label}]   attempt {attempt} aborted: {e}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(5)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError) as e:
            last_error = e
            print(f"[{label}]   attempt {attempt} failed with a transient error: {e}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(10 * attempt)
        except anthropic.APIStatusError as e:
            die(f"[{label}] Claude API request failed ({e.status_code}): {e.response.text[:2000]}")

    die(f"[{label}] Claude API request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def validate_episode(ep):
    required = ["episode_type", "date", "title", "tickers", "description", "lines"]
    for key in required:
        if key not in ep:
            die(f"Model output missing required field '{key}'")
    if not isinstance(ep["lines"], list) or not ep["lines"]:
        die("Model output 'lines' must be a non-empty list")
    for line in ep["lines"]:
        if line.get("speaker", "").lower() not in config.HOST_VOICES:
            die(f"Unexpected speaker '{line.get('speaker')}'; must be one of {list(config.HOST_VOICES)}")
        if not line.get("text"):
            die("A line is missing 'text'")


# Quick-agree/rebuttal openers that, used on nearly every line, are what
# made past episodes sound like the hosts were constantly cutting each
# other off (see the "ALSO IMPORTANT" pacing guidance in SCHEMA_INSTRUCTIONS
# above). This isn't an exhaustive list, just the common cases, for a cheap
# sanity check logged to the run -- it never fails the build, since a
# script that trips it is still usable, just worth a listen.
QUICK_OPENERS = ("right", "exactly", "that's", "yeah", "okay", "no,")


def log_pacing_diagnostics(ep):
    lines = ep["lines"]
    speakers = [l["speaker"].lower() for l in lines]
    same_speaker_runs = sum(1 for i in range(1, len(speakers)) if speakers[i] == speakers[i - 1])
    quick_opener_count = 0
    for line in lines:
        first_words = " ".join(line["text"].split()[:2]).lower().strip(",.")
        if any(first_words.startswith(o) for o in QUICK_OPENERS):
            quick_opener_count += 1
    pct = 100 * quick_opener_count / len(lines) if lines else 0
    print(
        f"[pacing] {len(lines)} lines, {same_speaker_runs} same-speaker-in-a-row "
        f"transitions, {quick_opener_count} ({pct:.0f}%) open with a quick "
        f"agree/rebuttal word"
    )
    if same_speaker_runs == 0:
        print(
            "[pacing] WARNING: every single line alternates speaker with no "
            "exceptions -- this rigid ping-pong pattern is a likely cause of "
            "hosts sounding like they're talking over each other, even with "
            "no literal audio overlap and a clean pause between lines."
        )
    if pct > 50:
        print(
            f"[pacing] WARNING: {pct:.0f}% of lines open with an instant "
            "agree/rebuttal word -- consider this a soft signal the episode "
            "may read as rapid-fire rather than a relaxed conversation."
        )


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        die("ANTHROPIC_API_KEY environment variable is not set")

    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    # Manual runs (Actions tab -> "Run workflow") can force a specific
    # episode type via the FORCE_EPISODE_TYPE env var, e.g. to produce
    # extra weekday_deep_dive episodes and build up a backlog on a day
    # that would otherwise auto-select a recap/preview. Scheduled runs
    # leave this unset and fall back to the normal day-of-week logic.
    valid_types = {"weekday_deep_dive", "saturday_recap", "sunday_preview"}
    forced_type = (os.environ.get("FORCE_EPISODE_TYPE") or "").strip()
    if forced_type and forced_type not in valid_types:
        die(f"FORCE_EPISODE_TYPE={forced_type!r} is not one of {sorted(valid_types)}")
    episode_type = forced_type or episode_type_for_weekday(now.weekday())

    used_stocks = load_used_stocks()
    # Avoid repeating anything covered in the last ~40 entries (kept short
    # so this list doesn't bloat the prompt).
    avoid_tickers = [e["ticker"] for e in used_stocks[-40:]]

    assigned_candidate = None
    if episode_type == "weekday_deep_dive":
        # Check against the FULL history (not just the last 40) so the
        # curated list never repeats a ticker, however long the show runs.
        all_used_tickers = {e["ticker"].upper() for e in used_stocks if e.get("ticker")}
        candidates = load_candidate_stocks()
        assigned_candidate = next_candidate(candidates, all_used_tickers)
        if assigned_candidate:
            print(f"Assigned candidate from state/candidate_stocks.json: "
                  f"{assigned_candidate['ticker']} ({assigned_candidate['company']})")
        elif candidates:
            print("All candidates in state/candidate_stocks.json have been covered; "
                  "falling back to open-ended pick.")
        else:
            print("No state/candidate_stocks.json found; falling back to open-ended pick.")

    print(f"=== Stage 1: research (episode_type={episode_type}, date={today_str}) ===")
    research_prompt = build_research_prompt(episode_type, today_str, avoid_tickers, assigned_candidate)
    # Research output is just a short brief, not a full script, so a much
    # smaller max_tokens is plenty and keeps a truncation-triggered retry
    # cheap.
    research_notes, research_cost = call_claude(
        research_prompt, api_key, label="research", use_search=True, max_tokens=4000
    )
    print("--- research brief ---")
    print(research_notes)
    print("--- end research brief ---")

    print(f"=== Stage 2: writing (episode_type={episode_type}, date={today_str}) ===")
    writing_prompt = build_writing_prompt(episode_type, today_str, research_notes)
    raw_text, writing_cost = call_claude(
        writing_prompt, api_key, label="write", use_search=False, max_tokens=16000
    )
    print(f"Total cost for this episode: approx ${research_cost + writing_cost:.3f}")

    episode = extract_json(raw_text)
    validate_episode(episode)
    log_pacing_diagnostics(episode)

    episode["episode_type"] = episode_type
    episode["date"] = today_str

    if episode_type == "weekday_deep_dive" and episode.get("tickers"):
        used_stocks.append({"ticker": episode["tickers"][0], "date": today_str})
        save_used_stocks(used_stocks)

    os.makedirs(PENDING_DIR, exist_ok=True)
    # Include a short random suffix so multiple episodes of the same type
    # produced on the same calendar day (e.g. running several manual
    # weekday_deep_dive backlog builds in one day) never collide on
    # filename -- each is its own file all the way through audio/manifest.
    out_path = os.path.join(PENDING_DIR, f"{today_str}-{episode_type}-{uuid.uuid4().hex[:6]}.json")
    with open(out_path, "w") as f:
        json.dump(episode, f, indent=2)
    print(f"Wrote {out_path}: '{episode['title']}' ({len(episode['lines'])} lines, tickers={episode.get('tickers')})")


if __name__ == "__main__":
    main()
