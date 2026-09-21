"""Central configuration for the podcast pipeline. Edit these values to
rebrand the show, change voices, or point at a different host.
"""

SHOW_TITLE = "Under the Radar"
SHOW_SUBTITLE = "Small stocks nobody's talking about yet"
SHOW_DESCRIPTION = (
    "A daily deep dive into overlooked, under-the-radar stocks, with a weekly "
    "recap every Saturday and a look ahead every Sunday. Two hosts, one "
    "under-covered ticker at a time. Not financial advice."
)
SHOW_AUTHOR = "Under the Radar Podcast"
OWNER_NAME = "Shawn"
OWNER_EMAIL = "shawnoche@gmail.com"
SHOW_LANGUAGE = "en-us"

# GitHub Pages base URL (project site: https://<user>.github.io/<repo>/)
GITHUB_USER = "shawn-noche"
REPO_NAME = "stock-podcast"
BASE_URL = f"https://{GITHUB_USER}.github.io/{REPO_NAME}/"

ITUNES_CATEGORY = ("Business", "Investing")
ITUNES_EXPLICIT = "false"

# Two conversational hosts and the OpenAI TTS voice each one uses.
# Valid voices as of writing: alloy, echo, fable, onyx, nova, shimmer.
HOST_VOICES = {
    "alex": "onyx",
    "jordan": "nova",
}

# gpt-4o-mini-tts was used originally, but it has a confirmed, still-open
# OpenAI bug where it silently drops the end of sentences/clips (see the
# OpenAI developer forum threads on "gpt-4o-mini-tts truncates final
# sentences") -- this is what was causing episodes to sound like the hosts
# were cutting each other off. tts-1 is an older, non-autoregressive model
# with no reports of this truncation behavior, and costs about the same per
# minute as gpt-4o-mini-tts (tts-1-hd would roughly double the cost for
# higher fidelity we don't need here).
TTS_MODEL = "tts-1"
TTS_FORMAT = "mp3"

DISCLAIMER = (
    "Quick reminder before we get into it: everything in this episode is for "
    "informational and entertainment purposes only. It is not financial "
    "advice, and you should always do your own research or talk to a "
    "licensed financial advisor before making investment decisions."
)

# A short, FIXED spoken welcome played at the very start of every single
# episode (after the musical intro sting, before that day's actual
# content) -- added 2026-09-22 because a first-time listener could land on
# any random episode, not necessarily episode 1, and had no way to know
# what the show even is. This is plain fixed text, not something the
# writer model generates per episode, so it's always present, never drifts
# in quality, and never gets cut for time by the model. generate_audio.py
# runs it through the exact same TTS pipeline as every other line (so it
# gets the same sentence-splitting and truncation-floor protection) --
# see the "lines = ..." line near the top of process_episode().
SHOW_INTRO_LINES = [
    {"speaker": "alex", "text": "Hey, welcome to Under the Radar. I'm Alex."},
    {"speaker": "jordan", "text": "And I'm Jordan. If this is your first time with us, here's the idea. Every weekday we take one small, overlooked stock that almost nobody's talking about, and we spend real time figuring out whether it deserves more attention."},
    {"speaker": "alex", "text": "Saturdays we zoom out and recap the whole week in the market, and Sundays we give you a quick preview of what's coming up next week."},
    {"speaker": "jordan", "text": "All right, let's get into today's show."},
]
