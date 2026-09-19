#!/usr/bin/env python3
"""Researches and writes today's episode script using the Claude API
(with its server-side web search tool), then drops the result into
pending/ for generate_audio.py to turn into audio.

This is the one step in the pipeline that needs real judgment and real
research: picking a genuinely under-the-radar stock, reading its investor
relations page and two most recent earnings reports, pulling recent news,
and writing natural two-host dialogue. Runs inside GitHub Actions, which
has normal internet access.
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
where reasonable, and vary sentence length. Do not use stage directions or
sound effect cues, only spoken words.

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


def build_prompt(episode_type, today_str, avoid_tickers, assigned_candidate=None):
    host_names = "Alex and Jordan"
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
   under-the-radar small-to-mid-cap US stock instead, and briefly
   mention in the episode that you swapped picks and why."""
        else:
            pick_instruction = f"""1. Identify ONE US-listed, small-to-mid cap stock (roughly $300M-$10B market
   cap as a guideline, not a hard rule) that is genuinely under-the-radar --
   NOT a mega-cap, NOT a stock that is already saturating financial media
   this week, NOT a meme stock riding pure hype. Do not pick any of these
   tickers, which the show has already covered recently: {avoid_str}."""

        return f"""You are producing today's ({today_str}) episode of "Under the Radar," a
daily podcast about overlooked, under-the-radar publicly traded stocks. The
two co-hosts are {host_names}. Target length: 15-25 minutes of spoken
dialogue, roughly 2600-3800 words total across both hosts.

You have a budget of AT MOST {MAX_SEARCHES} web searches total -- be
economical. Combine what you need into broad, well-targeted queries rather
than many narrow ones (e.g. one query per: candidate/pick, IR page +
earnings, recent news), and stop searching as soon as you have enough to
write a great episode. IMPORTANT: if a search is ever refused because
you've used up this budget, do NOT try again -- immediately stop searching
and write the episode using only the information you've already gathered.
Repeatedly re-attempting a blocked search wastes a huge amount of money for
no benefit, so treat a refusal as a hard stop, not a retry signal.

Do this research using web search before writing anything:

{pick_instruction}
2. Find that company's investor relations page and read its two most recent
   quarterly earnings reports or earnings press releases. Note revenue,
   revenue growth rate, margins, segment breakdown if disclosed, guidance,
   and any notable management commentary or surprises.
3. Search for recent news about the company from the last few weeks beyond
   the earnings reports (product news, contracts, insider activity,
   analyst commentary, sector context).
4. Write the two-host episode script. It must, in this rough order:
   - Cold open that hooks the listener on why this stock is worth 20 minutes
     of their time.
   - Explain in plain, accessible language what the company actually does
     and, specifically, how it makes money (its business model and revenue
     streams) -- assume the listener has never heard of it.
   - Walk through what stood out in the two most recent earnings reports:
     revenue trends, growth, margins, guidance, surprises.
   - Cover the recent news you found and what it means going forward.
   - Discuss explicitly why this stock is flying under the radar right now,
     and what could change that.
   - Close with the required disclaimer (verbatim, spoken by one host) and a
     sign-off.

{SCHEMA_INSTRUCTIONS}"""

    if episode_type == "saturday_recap":
        return f"""You are producing this Saturday's ({today_str}) weekly recap episode of
"Under the Radar," a podcast about the stock market. Co-hosts:
{host_names}. Target length: ~20 minutes, roughly 2800-3400 words.

You have a budget of AT MOST {MAX_SEARCHES} web searches total -- be
economical, use broad well-targeted queries rather than many narrow ones.
IMPORTANT: if a search is ever refused because you've used up this budget,
do NOT try again -- immediately stop searching and write the episode using
only what you've already gathered. Repeatedly re-attempting a blocked
search wastes a huge amount of money for no benefit.

Use web search to research the past week (Monday through Friday) in the US
stock market: major index performance, the most significant market-moving
stories, notable earnings from the week, and any macro/economic data
releases that mattered. Also briefly revisit how the market reacted to
under-the-radar-style stocks generally this week if there's anything
notable.

Write a natural two-host conversation recapping the week: what happened,
why it mattered, and any threads worth remembering. Close with the required
disclaimer (verbatim, spoken by one host) and a sign-off.

Note: this episode has no single featured ticker; set "tickers" to an empty
list unless specific stocks are central to the recap, in which case list
them.

{SCHEMA_INSTRUCTIONS}"""

    # sunday_preview
    return f"""You are producing this Sunday's ({today_str}) week-ahead preview episode of
"Under the Radar," a podcast about the stock market. Co-hosts: {host_names}.
Keep this SHORT: target 5-10 minutes, roughly 900-1500 words.

You have a budget of AT MOST {MAX_SEARCHES} web searches total -- be
economical, use broad well-targeted queries rather than many narrow ones.
IMPORTANT: if a search is ever refused because you've used up this budget,
do NOT try again -- immediately stop searching and write the episode using
only what you've already gathered. Repeatedly re-attempting a blocked
search wastes a huge amount of money for no benefit.

Use web search to find what's coming up in the next week: scheduled major
earnings releases, economic data releases (e.g. CPI, jobs report, Fed
meetings), and any other notable calendar events for US markets.

Write a brief, upbeat two-host conversation previewing what to watch for in
the week ahead. Close with the required disclaimer (verbatim, spoken by one
host) and a sign-off.

Note: set "tickers" to any specific tickers mentioned, or an empty list if
none are central.

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
# A real weekday deep dive genuinely needs on the order of 8-9 searches
# (find the pick, read two earnings reports, check recent news -- several
# distinct lookups). An earlier, tighter budget of 5 caused every single
# run to hit the limit, get refused, and then the model kept trying
# anyway instead of stopping -- which is what caused a run to balloon to
# 1.4M input tokens and ~$3.23 for one episode. Budgeting enough searches
# for it to actually finish in one pass is the real fix: if it never gets
# refused, it never has a reason to retry.
MAX_SEARCHES = 8

# Safety net: the model is instructed to stop cleanly once it hits
# MAX_SEARCHES, but if it ever ignores that and keeps re-attempting refused
# searches anyway, each retry re-sends the whole growing conversation as
# input tokens. If total search *attempts* (including refused ones) blows
# past this ceiling, we abort the connection outright rather than let cost
# run away unbounded. Set with real headroom above the ~9 attempts a normal
# run uses, so this only ever fires on genuinely runaway behavior, not on
# ordinary research.
HARD_SEARCH_ATTEMPT_CEILING = MAX_SEARCHES + 6


class RunawaySearchLoop(Exception):
    pass


class TruncatedResponse(Exception):
    pass


def log_usage_and_cost(usage):
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    searches = usage.server_tool_use.web_search_requests if usage.server_tool_use else 0
    cost = (
        input_tokens / 1_000_000 * PRICE_INPUT_PER_MTOK
        + output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOK
        + searches / 1000 * PRICE_PER_1000_SEARCHES
    )
    print(
        f"Usage: {input_tokens} input tokens, {output_tokens} output tokens, "
        f"{searches} billed web searches -> approx ${cost:.3f} for this episode"
    )


def call_claude(prompt, api_key):
    # This request can involve several web searches plus writing a long
    # script, which can take minutes. Streaming keeps the connection
    # actively fed with data the whole time instead of sitting idle
    # waiting for one big response -- idle connections like that get
    # silently dropped by network infrastructure in between (this is what
    # caused the very first run to fail with a RemoteDisconnected error).
    client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=900.0)

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Calling Claude API (attempt {attempt}/{MAX_ATTEMPTS})...")
            text_parts = []
            search_attempts = 0
            with client.messages.stream(
                model=ANTHROPIC_MODEL,
                max_tokens=32000,
                tools=[
                    {
                        "type": "web_search_20260318",
                        "name": "web_search",
                        "max_uses": MAX_SEARCHES,
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta" and getattr(event.delta, "type", None) == "text_delta":
                        text_parts.append(event.delta.text)
                    elif event.type == "content_block_start" and getattr(event.content_block, "type", None) == "server_tool_use":
                        search_attempts += 1
                        print(f"  ...search attempt {search_attempts}")
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
            log_usage_and_cost(final_message.usage)
            if final_message.stop_reason == "max_tokens":
                # The response got cut off before finishing -- almost
                # always mid-way through writing the final episode JSON,
                # since that's the longest single piece of text in the
                # whole exchange. Whatever text we captured is incomplete
                # and not worth trying to parse or salvage; treat this as
                # retryable, the same as a runaway search loop, rather
                # than silently returning broken output that will just
                # fail JSON parsing later with no attempts left.
                raise TruncatedResponse(
                    "response was truncated at max_tokens before finishing "
                    "(likely mid-script); discarding this attempt"
                )
            return "".join(text_parts)
        except (RunawaySearchLoop, TruncatedResponse) as e:
            last_error = e
            print(f"  attempt {attempt} aborted: {e}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(5)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError) as e:
            last_error = e
            print(f"  attempt {attempt} failed with a transient error: {e}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(10 * attempt)
        except anthropic.APIStatusError as e:
            die(f"Claude API request failed ({e.status_code}): {e.response.text[:2000]}")

    die(f"Claude API request failed after {MAX_ATTEMPTS} attempts: {last_error}")


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

    prompt = build_prompt(episode_type, today_str, avoid_tickers, assigned_candidate)
    print(f"Requesting script from Claude for episode_type={episode_type}, date={today_str}...")
    raw_text = call_claude(prompt, api_key)
    episode = extract_json(raw_text)
    validate_episode(episode)

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
