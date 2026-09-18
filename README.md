# Under the Radar — automated stock podcast

Fully automated podcast pipeline. A daily deep dive into overlooked,
under-the-radar stocks, plus a Saturday weekly recap and a Sunday
week-ahead preview. Two AI hosts, generated end to end with no manual
production work. Everything runs inside GitHub Actions on a daily
schedule — once set up, it needs no ongoing involvement from anyone.

## How it works

Every step below runs inside a single GitHub Actions job
(`.github/workflows/produce-episode.yml`), triggered daily on a cron
schedule (or manually via "Run workflow"):

1. **Research & script** (`scripts/research_and_write.py`) — calls the
   Claude API with its web search tool to: pick the day's episode type from
   the calendar (Mon–Fri deep dive / Saturday recap / Sunday preview); for a
   weekday deep dive, pick one genuinely under-the-radar stock (avoiding
   anything in `state/used_stocks.json`), find its investor relations page,
   read its two most recent earnings reports, pull recent news, and write a
   natural two-host dialogue script that explains what the company does,
   how it makes money, what the earnings showed, and why it's flying under
   the radar. Saves the result as JSON under `pending/`.
2. **Audio production** (`scripts/generate_audio.py`) — calls the OpenAI
   TTS API to voice each line with one of two distinct voices, stitches the
   dialogue together with ffmpeg, adds a synthesized intro/outro sting, and
   loudness-normalizes the mix. Writes show notes/transcript to
   `docs/episodes/notes/`.
3. **Feed rebuild** (`scripts/build_rss.py`) — updates
   `state/episodes.json` (the episode manifest) and regenerates
   `docs/feed.xml`.
4. **Publish** — the job commits the finished MP3 + updated feed back to
   `main`. GitHub Pages serves everything under `docs/` at
   `https://<user>.github.io/stock-podcast/`. `docs/feed.xml` is the public
   RSS feed to submit to Spotify for Podcasters and Apple Podcasts Connect.

## Repo layout

```
.github/workflows/produce-episode.yml   The whole pipeline, run daily by GitHub Actions
scripts/config.py                       Show branding, voices, category, etc.
scripts/research_and_write.py           Claude API research + script writing -> pending/
scripts/generate_audio.py               OpenAI TTS + ffmpeg assembly + manifest update
scripts/build_rss.py                    Rebuilds docs/feed.xml from the manifest
scripts/generate_cover.py               One-off cover art generator (run manually, not in CI)
pending/                                Scripts waiting to be produced (JSON)
state/episodes.json                     Episode manifest (source of truth for feed.xml)
state/used_stocks.json                  History of covered tickers (avoid repeats)
state/processed/                        Archived scripts after production
docs/                                   Published site (GitHub Pages root): feed.xml, cover.jpg, audio, notes
```

## Episode script format (what research_and_write.py produces)

```json
{
  "episode_type": "weekday_deep_dive",
  "date": "2026-09-21",
  "title": "Episode title",
  "tickers": ["ABCD"],
  "description": "2-3 sentence show notes / RSS description.",
  "lines": [
    {"speaker": "alex", "text": "..."},
    {"speaker": "jordan", "text": "..."}
  ]
}
```

## One-time setup

- [ ] Repo created as **public** (required for free GitHub Pages)
- [ ] `ANTHROPIC_API_KEY` added under Settings → Secrets and variables →
      Actions (console.anthropic.com API key; pay-as-you-go, separate from
      a claude.ai subscription)
- [ ] `OPENAI_API_KEY` added the same way
- [ ] Settings → Pages → Source: "Deploy from a branch" → `main` / `/docs`
- [ ] Run the workflow once manually (Actions tab → Produce Episode → Run
      workflow) to confirm everything works before relying on the schedule
- [ ] Public feed (`.../feed.xml`) submitted to Spotify for Podcasters and
      Apple Podcasts Connect

## Cost (estimated, at ~130 min of finished audio/week)

- OpenAI TTS: ~$8–10/month
- Claude API (research + writing, incl. web search): ~$6–8/month
- GitHub Actions + Pages: free
- **Total: roughly $15–20/month**

## Changing things later

- Show name, host voices, category, disclaimer text: edit `scripts/config.py`.
- Publish time: edit the `cron` line in `.github/workflows/produce-episode.yml`
  (currently `0 10 * * *`, i.e. 10:00 UTC / ~6:00am ET).
- Research/writing instructions per episode type: edit the prompts in
  `scripts/research_and_write.py`.
