"""Web access for characters: DuckDuckGo search and reading a page as plain text."""
import asyncio
import re
import ssl

import aiohttp
import truststore

from .log import log

MAX_PAGE_CHARS = 6000
# the OS certificate store; Python's own copy can hold expired roots on Windows
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _search(query, n):
    from ddgs import DDGS
    return DDGS().text(query, max_results=n) or []


async def web_search(query, n=5):
    query = str(query or "").strip()
    if not query:
        return "Give a search query."
    try:
        hits = await asyncio.to_thread(_search, query, n)
    except Exception as e:
        log(f"[web] search for {query!r} failed: {e!r}", level="warn")
        return f"The search failed ({type(e).__name__}). Try again later or with other words."
    if not hits:
        return f"No results for {query!r}."
    lines = [f"Results for {query!r}:"]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. {h.get('title', '').strip()}\n   {h.get('href', '')}\n   {h.get('body', '').strip()}")
    return "\n".join(lines)


def html_to_text(html):
    try:
        import lxml.html
        doc = lxml.html.fromstring(html)
        title = (doc.findtext(".//title") or "").strip()
        for bad in doc.xpath("//head|//script|//style|//noscript|//nav|//footer|//header|//form|//svg"):
            bad.drop_tree()
        text = doc.text_content()
    except Exception:
        title, text = "", re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return (title + "\n\n" if title else "") + text


async def read_webpage(session, url):
    url = str(url or "").strip()
    if not re.match(r"https?://", url, re.I):
        return "Give a full http(s) link."
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20), ssl=SSL_CONTEXT,
                               headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CharacterBot/1.0"}) as r:
            if r.status != 200:
                return f"The page answered HTTP {r.status}."
            ctype = r.headers.get("Content-Type", "")
            if "html" not in ctype and not ctype.startswith("text/"):
                return f"That link is not a web page ({ctype or 'unknown type'})."
            raw = await r.content.read(-1)
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        return f"Couldn't open the page: {type(e).__name__}."
    text = html_to_text(raw[:3_000_000].decode(r.charset or "utf-8", "replace"))
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[page cut off]"
    return text or "The page has no readable text."
