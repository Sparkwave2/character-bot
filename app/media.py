"""Image download + downscale to base64 data URIs for the vision model."""
import base64
import io
import re

import aiohttp
from PIL import Image

from .log import log

URL_RE = re.compile(r"https?://(?:[^\s<>()\"']|\([^\s<>()\"']*\))+", re.I)
IMG_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|bmp)(\?|$)", re.I)
MAX_BYTES = 12 * 1024 * 1024


def encode_image(data, max_side=896):
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)  # first frame of GIFs
        im = im.convert("RGB")
    except Exception:
        return None
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def fetch_image(session, url, max_side=896, require_image_type=False):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20),
                               headers={"User-Agent": "Mozilla/5.0 CharacterBot"}) as r:
            if r.status != 200:
                log(f"[media] {url[:100]} answered HTTP {r.status}", level="warn")
                return None
            ctype = r.headers.get("Content-Type", "")
            if require_image_type and not ctype.startswith("image/"):
                return None
            if int(r.headers.get("Content-Length", 0) or 0) > MAX_BYTES:
                return None
            # content.read(n) only returns what has arrived so far; read to the end
            buf = bytearray()
            async for chunk in r.content.iter_chunked(64 * 1024):
                buf += chunk
                if len(buf) > MAX_BYTES:
                    return None
            data = bytes(buf)
    except (aiohttp.ClientError, TimeoutError, ValueError) as e:
        log(f"[media] failed to fetch {url[:100]}: {e!r}", level="warn")
        return None
    img = encode_image(data, max_side)
    if img is None:
        log(f"[media] couldn't decode {url[:100]} ({len(data)} bytes)", level="warn")
    return img


async def images_from_message(session, message, max_side=896, fetch_links=True, limit=4):
    """Collect images from attachments, embeds and pasted links."""
    urls, seen = [], set()

    def push(u, strict):
        if u and u not in seen:
            seen.add(u)
            urls.append((u, strict))

    for a in message.attachments:
        if (a.content_type or "").startswith("image/") or IMG_EXT_RE.search(a.filename or ""):
            push(a.url, False)
    for st in getattr(message, "stickers", []):
        if getattr(st, "url", None) and str(getattr(st, "format", "")).lower().find("lottie") < 0:
            push(str(st.url), True)
    if fetch_links:
        for e in message.embeds:
            for part in (e.image, e.thumbnail):
                u = getattr(part, "proxy_url", None) or getattr(part, "url", None)
                if u:
                    push(u, True)
        for u in URL_RE.findall(message.content or ""):
            push(u, not IMG_EXT_RE.search(u))  # unknown links must answer with image/*
    out = []
    if urls:
        log(f"[media] message has {len(urls)} image candidate(s)")
    for u, strict in urls[:limit * 2]:
        img = await fetch_image(session, u, max_side, require_image_type=strict)
        if img:
            out.append(img)
        if len(out) >= limit:
            break
    if urls:
        log(f"[media] got {len(out)} of {len(urls)} image(s)")
    return out
