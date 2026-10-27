"""GIF search through the KLIPY API (https://klipy.com, a free Tenor replacement)."""
import asyncio

import aiohttp

API = "https://api.klipy.com/api/v1"


class GifError(Exception):
    pass


async def search_gif(session, key, query, customer_id="characterbot"):
    """First GIF for `query` -> (url, title). The URL is a direct .gif, which Discord shows inline."""
    if not key:
        raise GifError("no KLIPY API key is set (Behavior page)")
    params = {"q": query, "per_page": 8, "customer_id": customer_id, "locale": "en"}
    try:
        async with session.get(f"{API}/{key}/gifs/search", params=params,
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json(content_type=None)
            if r.status in (401, 403):
                raise GifError("KLIPY rejected the API key")
            if r.status != 200 or not isinstance(data, dict) or not data.get("result"):
                raise GifError(f"KLIPY answered HTTP {r.status}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise GifError(f"couldn't reach KLIPY: {e!r}")
    for item in (data.get("data") or {}).get("data") or []:
        if item.get("type") == "ad":
            continue
        files = item.get("file") or {}
        for size in ("md", "hd", "sm"):
            url = ((files.get(size) or {}).get("gif") or {}).get("url")
            if url:
                return url, item.get("title") or query
    raise GifError(f'no GIF found for "{query}"')
