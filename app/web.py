"""Web access for characters: DuckDuckGo search and reading a page as plain text."""
import asyncio
import re
import ssl
from dataclasses import dataclass
from typing import Optional

import aiohttp
import truststore

from .log import log

MAX_PAGE_CHARS = 6000
MAX_IMAGE_BYTES = 12 * 1024 * 1024
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CharacterBot/1.0"
URL_RE = re.compile(r"https?://(?:[^\s<>()\"']|\([^\s<>()\"']*\))+", re.I)   # keeps wiki/Foo_(bar)
# pages that answer 200 but only say they don't exist
SOFT_404_RE = re.compile(r"\b404\b|page not found|not be found|doesn'?t exist|does not exist|no longer available",
                         re.I)
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


@dataclass
class Link:
    url: str
    ok: Optional[bool]      # True works, False dead (404, no such site), None couldn't tell (403, timeout)
    kind: str = "unknown"   # page | image | file | dead | unknown
    reason: str = ""
    title: str = ""
    text: str = ""
    image: Optional[str] = None   # data URI for the vision model


async def _read(r, limit):
    buf = bytearray()
    async for chunk in r.content.iter_chunked(64 * 1024):
        buf += chunk
        if len(buf) >= limit:
            break
    return bytes(buf[:limit])


async def fetch_link(session, url, max_side=896):
    """Open a link once and say what's behind it."""
    if not re.match(r"https?://[^\s/]+\.[^\s/]+", url or "", re.I):
        return Link(url, False, "dead", "that isn't a full web address")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=15), ssl=SSL_CONTEXT,
                               headers={"User-Agent": UA}, allow_redirects=True) as r:
            if r.status in (404, 410):
                return Link(url, False, "dead", f"HTTP {r.status}, the page doesn't exist")
            if r.status >= 400:
                return Link(url, None, "unknown", f"HTTP {r.status}")
            ctype = r.headers.get("Content-Type", "").lower()
            if ctype.startswith("image/"):
                from .media import encode_image
                data = await _read(r, MAX_IMAGE_BYTES)
                img = encode_image(data, max_side)
                return Link(url, bool(img), "image" if img else "dead", "" if img else "the image is broken",
                            image=img)
            if "html" not in ctype and not ctype.startswith("text/"):
                return Link(url, True, "file", ctype or "unknown type")
            raw = await _read(r, 3_000_000)
            text = html_to_text(raw.decode(r.charset or "utf-8", "replace"))
    except aiohttp.ClientConnectorCertificateError:
        return Link(url, None, "unknown", "the site's certificate is invalid")
    except (aiohttp.ClientConnectorError, aiohttp.InvalidURL):
        return Link(url, False, "dead", "the site doesn't exist or can't be reached")
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
        return Link(url, None, "unknown", type(e).__name__)
    title, _, body = text.partition("\n\n") if "\n\n" in text else ("", "", text)
    if SOFT_404_RE.search(title) or (len(body) < 400 and SOFT_404_RE.search(body[:200])):
        return Link(url, False, "dead", f'the page says "{(title or body)[:60]}"')
    return Link(url, True, "page", title=title, text=body)


def links_in(text):
    """http(s) links in a message, without trailing punctuation or markdown."""
    out = []
    for u in URL_RE.findall(text or ""):
        u = u.rstrip(".,!?;:*~>]'\"")
        if u not in out:
            out.append(u)
    return out


def remove_link(text, url):
    """Take a link out of a message, keeping the words of a markdown [text](link)."""
    u = re.escape(url)
    text = re.sub(r"\[([^\]]*)\]\(\s*<?" + u + r">?\s*\)", r"\1", text)
    text = re.sub(r"<" + u + r">|" + u, "", text)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\(\s*\)", "", text)).strip()


def preview(link, chars=300):
    """One line for the chat history: what a posted link shows."""
    if link.kind == "page":
        snippet = re.sub(r"\s+", " ", link.text)[:chars]
        return f"[link {link.url}: {link.title or 'web page'} — {snippet}{'…' if len(link.text) > chars else ''}]"
    if link.ok is False:
        return f"[link {link.url} doesn't work: {link.reason}]"
    return ""


async def read_webpage(session, url, max_side=896):
    url = str(url or "").strip()
    if not re.match(r"https?://", url, re.I):
        return "Give a full http(s) link."
    link = await fetch_link(session, url, max_side)
    if link.kind == "image":   # shown to the vision model
        return [{"type": "text", "text": f"[system] {url} is a picture:"},
                {"type": "image_url", "image_url": {"url": link.image}}]
    if link.kind != "page":
        return f"Couldn't read the page: {link.reason or link.kind}."
    text = (link.title + "\n\n" if link.title else "") + link.text
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n[page cut off]"
    return text or "The page has no readable text."
