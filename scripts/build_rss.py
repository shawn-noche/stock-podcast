#!/usr/bin/env python3
"""Regenerates docs/feed.xml from state/episodes.json.

Run this after generate_audio.py so the public RSS feed reflects the
latest episode list. Pure stdlib, no dependencies.
"""
import json
import os
import xml.sax.saxutils as sx
from datetime import datetime, timezone
from email.utils import format_datetime

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "state", "episodes.json")
FEED_PATH = os.path.join(ROOT, "docs", "feed.xml")
COVER_FILENAME = "cover.jpg"


def esc(s):
    return sx.escape(s or "")


def build_item(ep):
    audio_url = config.BASE_URL + "episodes/audio/" + ep["audio_filename"]
    pub_dt = datetime.fromisoformat(ep["pub_datetime_utc"])
    pub_date_rfc822 = format_datetime(pub_dt)
    return f"""    <item>
      <title>{esc(ep['title'])}</title>
      <description>{esc(ep['description'])}</description>
      <itunes:summary>{esc(ep['description'])}</itunes:summary>
      <pubDate>{pub_date_rfc822}</pubDate>
      <enclosure url="{esc(audio_url)}" length="{ep['file_size_bytes']}" type="audio/mpeg" />
      <guid isPermaLink="false">{esc(ep['guid'])}</guid>
      <itunes:duration>{esc(ep['duration_formatted'])}</itunes:duration>
      <itunes:explicit>{config.ITUNES_EXPLICIT}</itunes:explicit>
      <itunes:episodeType>full</itunes:episodeType>
    </item>"""


def main():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []

    manifest.sort(key=lambda e: e["pub_datetime_utc"], reverse=True)
    items_xml = "\n".join(build_item(ep) for ep in manifest)

    cat_main, cat_sub = config.ITUNES_CATEGORY
    now_rfc822 = format_datetime(datetime.now(timezone.utc))
    cover_url = config.BASE_URL + COVER_FILENAME

    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{esc(config.SHOW_TITLE)}</title>
    <link>{esc(config.BASE_URL)}</link>
    <atom:link href="{esc(config.BASE_URL)}feed.xml" rel="self" type="application/rss+xml" />
    <language>{esc(config.SHOW_LANGUAGE)}</language>
    <description>{esc(config.SHOW_DESCRIPTION)}</description>
    <itunes:summary>{esc(config.SHOW_DESCRIPTION)}</itunes:summary>
    <itunes:author>{esc(config.SHOW_AUTHOR)}</itunes:author>
    <itunes:owner>
      <itunes:name>{esc(config.OWNER_NAME)}</itunes:name>
      <itunes:email>{esc(config.OWNER_EMAIL)}</itunes:email>
    </itunes:owner>
    <itunes:image href="{esc(cover_url)}" />
    <image>
      <url>{esc(cover_url)}</url>
      <title>{esc(config.SHOW_TITLE)}</title>
      <link>{esc(config.BASE_URL)}</link>
    </image>
    <itunes:category text="{esc(cat_main)}">
      <itunes:category text="{esc(cat_sub)}" />
    </itunes:category>
    <itunes:explicit>{config.ITUNES_EXPLICIT}</itunes:explicit>
    <lastBuildDate>{now_rfc822}</lastBuildDate>
{items_xml}
  </channel>
</rss>
"""
    os.makedirs(os.path.dirname(FEED_PATH), exist_ok=True)
    with open(FEED_PATH, "w") as f:
        f.write(feed)
    print(f"Wrote {FEED_PATH} with {len(manifest)} episode(s)")


if __name__ == "__main__":
    main()
