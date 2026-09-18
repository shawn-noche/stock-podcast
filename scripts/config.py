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

TTS_MODEL = "gpt-4o-mini-tts"
TTS_FORMAT = "mp3"

DISCLAIMER = (
    "Quick reminder before we get into it: everything in this episode is for "
    "informational and entertainment purposes only. It is not financial "
    "advice, and you should always do your own research or talk to a "
    "licensed financial advisor before making investment decisions."
)
