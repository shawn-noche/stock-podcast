#!/usr/bin/env python3
"""Deterministic research data fetcher for the weekday deep-dive episode.

Pulls a company's most recent earnings-related SEC filings and a handful
of recent news headlines using plain HTTP requests -- no LLM involved, no
search budget to manage, no runaway-search failure mode possible at all.

This replaces Claude's own web search for the "read the two most recent
earnings reports" part of research, which was the single largest source of
both cost and unreliability in this project: search results ballooning the
token count on every round, and the model not reliably stopping when told
it had used up its budget. Fetching the real documents with code instead
removes that failure category structurally rather than trying to police it
with tighter prompts.

Everything here is free, public data with no API key required:
- SEC EDGAR (https://www.sec.gov/os/webmaster-faq#developers) -- requires
  only a descriptive User-Agent header identifying who's asking.
- Google News RSS -- for recent headlines only (not full article text,
  which keeps this simple and avoids paywalls/scraping fragility).

If anything here fails, callers should catch DataFetchFailed and fall back
to the older web-search-based research path for that one episode rather
than failing the whole run -- this is new code hitting live external
services for the first time in production, so a safety net matters more
than saving a few cents on any single episode.
"""
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

USER_AGENT = "UnderTheRadarPodcast/1.0 (contact: shawnoche@gmail.com)"
REQUEST_TIMEOUT = 20
# Be a good citizen of a free, shared, unauthenticated API -- SEC asks
# automated tools not to hammer it. We only make a handful of requests per
# episode, so this costs a couple of seconds total, not a meaningful delay.
REQUEST_DELAY = 0.2


class DataFetchFailed(Exception):
    pass


def _get(url):
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise DataFetchFailed(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        raise DataFetchFailed(f"network error fetching {url}: {e.reason}") from e
    time.sleep(REQUEST_DELAY)
    return data


_TICKER_MAP_CACHE = None


def _load_ticker_map():
    global _TICKER_MAP_CACHE
    if _TICKER_MAP_CACHE is not None:
        return _TICKER_MAP_CACHE
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DataFetchFailed(f"company_tickers.json was not valid JSON: {e}")
    mapping = {}
    for entry in data.values():
        t = (entry.get("ticker") or "").strip().upper()
        cik = entry.get("cik_str")
        if t and cik is not None:
            mapping[t] = int(cik)
    _TICKER_MAP_CACHE = mapping
    return mapping


def get_cik_for_ticker(ticker):
    mapping = _load_ticker_map()
    cik = mapping.get(ticker.strip().upper())
    if cik is None:
        raise DataFetchFailed(f"ticker {ticker!r} not found in SEC's company_tickers.json")
    return cik


def get_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    raw = _get(url)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise DataFetchFailed(f"submissions JSON for CIK {cik} was not valid: {e}")


def _iter_recent_filings(submissions):
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    items_list = recent.get("items", [])
    n = len(forms)
    for i in range(n):
        yield {
            "form": forms[i],
            "filingDate": filing_dates[i] if i < len(filing_dates) else None,
            "accessionNumber": accession_numbers[i] if i < len(accession_numbers) else None,
            "primaryDocument": primary_documents[i] if i < len(primary_documents) else None,
            "items": items_list[i] if i < len(items_list) else "",
        }


def find_earnings_filings(submissions, limit=2):
    """Prefer 8-Ks that specifically report results of operations -- SEC
    item code 2.02 is how EDGAR tags an earnings-release 8-K. Falls back
    to plain 10-Q/10-K filings if no such 8-K exists (some smaller
    companies fold earnings straight into the quarterly report instead of
    filing a separate press release)."""
    earnings_8ks = [
        f for f in _iter_recent_filings(submissions)
        if f["form"] == "8-K" and "2.02" in (f["items"] or "").split(",")
    ]
    if earnings_8ks:
        earnings_8ks.sort(key=lambda f: f["filingDate"] or "", reverse=True)
        return earnings_8ks[:limit], "8-K (results of operations)"

    quarterlies = [
        f for f in _iter_recent_filings(submissions)
        if f["form"] in ("10-Q", "10-K")
    ]
    if quarterlies:
        quarterlies.sort(key=lambda f: f["filingDate"] or "", reverse=True)
        return quarterlies[:limit], "10-Q/10-K"

    return [], None


def _accession_no_dashes(accession):
    return accession.replace("-", "")


def list_filing_documents(cik, accession):
    """The documents inside one filing's accession folder, so we can find
    the actual earnings-release exhibit rather than just the 8-K cover
    page (which is often just a couple of boilerplate sentences pointing
    at the real exhibit)."""
    acc_nodash = _accession_no_dashes(accession)
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/index.json"
    raw = _get(url)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DataFetchFailed(f"filing index JSON for {accession} was not valid: {e}")
    return data.get("directory", {}).get("item", [])


def _strip_html(raw_bytes):
    text = raw_bytes.decode("utf-8", errors="replace")
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_relevant_slice(text, max_chars):
    # For a 10-Q/10-K fallback document, the actual narrative (revenue,
    # margins, guidance commentary) lives in "Item 2: Management's
    # Discussion and Analysis," which tends to appear well into the
    # document, after the raw financial statements. Start there if we can
    # find it instead of just taking the first max_chars, which is usually
    # cover page + table of contents + legal boilerplate.
    match = re.search(r"item\s*2\.?\s*management.?s\s+discussion", text, re.I)
    if match and len(text) - match.start() > 2000:
        text = text[match.start():]
    return text[:max_chars]


def fetch_filing_text(cik, filing, max_chars=25000):
    """Fetches the best available document text for one filing: for an
    8-K, prefers an EX-99* exhibit (the actual press release) over the
    8-K cover page itself; otherwise fetches primaryDocument directly."""
    accession = filing["accessionNumber"]
    acc_nodash = _accession_no_dashes(accession)
    doc_name = filing["primaryDocument"]

    if filing["form"] == "8-K":
        try:
            docs = list_filing_documents(cik, accession)
            exhibit = next(
                (d for d in docs if re.search(r"ex.?99", d.get("name", ""), re.I)),
                None,
            )
            if exhibit:
                doc_name = exhibit["name"]
        except DataFetchFailed:
            pass  # fine, just use the primary document instead

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc_name}"
    raw = _get(url)
    text = _extract_relevant_slice(_strip_html(raw), max_chars)
    return text, url


def fetch_news_headlines(ticker, company, limit=8):
    """Free, no-key headline search via Google News RSS. Headlines only,
    not full article text -- news here is supplementary color, not the
    main source material, so this stays fast and doesn't hit paywalls."""
    query = urllib.parse.quote(f"{company} OR {ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        raw = _get(url)
    except DataFetchFailed:
        return []  # nice-to-have; don't fail the whole run over a news hiccup
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    items = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else ""
        if title:
            items.append((title, source, pub_date))
    return items


def fetch_research_bundle(ticker, company):
    """Top-level entry point. Returns a dict with everything the writer
    needs, or raises DataFetchFailed if the core earnings data (not the
    supplementary news) can't be assembled -- callers should fall back to
    the web-search research path on that exception."""
    cik = get_cik_for_ticker(ticker)
    submissions = get_submissions(cik)
    filings, source_kind = find_earnings_filings(submissions, limit=2)
    if not filings:
        raise DataFetchFailed(
            f"no 8-K/10-Q/10-K filings found for {ticker} (CIK {cik}) -- "
            "likely delisted, acquired, or too new a listing"
        )

    filing_texts = []
    for f in filings:
        text, url = fetch_filing_text(cik, f)
        filing_texts.append({
            "form": f["form"],
            "filingDate": f["filingDate"],
            "url": url,
            "text": text,
        })

    news = fetch_news_headlines(ticker, company)

    return {
        "ticker": ticker,
        "company": company,
        "cik": cik,
        "source_kind": source_kind,
        "filings": filing_texts,
        "news": news,
    }
