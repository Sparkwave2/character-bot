"""ElevenLabs text to speech, returned as 48 kHz stereo PCM ready for Discord."""
import json
import re

import aiohttp
import numpy as np

from .log import log

API = "https://api.elevenlabs.io/v1"
RATE = 24000          # pcm_24000 is available on every ElevenLabs plan
VOICE_ID_RE = re.compile(r"^[A-Za-z0-9]{20}$")


class TTSError(Exception):
    pass


class ElevenLabs:
    def __init__(self, cfg):
        self.cfg = cfg
        self._voices = {}   # lower name -> id

    @property
    def key(self):
        return (self.cfg["voice"].get("elevenlabs_key") or "").strip()

    async def list_voices(self, session):
        """[(name, id, kind)] of the voices in the account; kind is one of VOICE_KINDS."""
        async with session.get(f"{API}/voices", headers={"xi-api-key": self.key},
                               timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status != 200:
                raise await api_error(r)
            data = await r.json()
        out = [(v["name"], v["voice_id"], voice_kind(v)) for v in data.get("voices", [])]
        self._voices = {n.lower(): i for n, i, _ in out}
        return out

    async def resolve(self, session, voice):
        """A voice ID, or a voice name from the account."""
        voice = voice.strip()
        if VOICE_ID_RE.match(voice):
            return voice
        if voice.lower() not in self._voices:
            await self.list_voices(session)
        if voice.lower() not in self._voices:
            raise TTSError(f"there is no ElevenLabs voice called {voice!r}")
        return self._voices[voice.lower()]

    async def speak(self, session, voice, text):
        if not self.key:
            raise TTSError("no ElevenLabs API key set")
        vid = await self.resolve(session, voice)
        body = {"text": text, "model_id": self.cfg["voice"].get("tts_model") or "eleven_flash_v2_5"}
        async with session.post(f"{API}/text-to-speech/{vid}", params={"output_format": f"pcm_{RATE}"},
                                json=body, headers={"xi-api-key": self.key},
                                timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                raise await api_error(r)
            raw = await r.read()
        log(f"[voice] ElevenLabs returned {len(raw) / 2 / RATE:.1f} s of audio", level="debug")
        return to_discord_pcm(raw)


# (label, usable on the free plan) in the order the picker lists them
VOICE_KINDS = {"premade": ("Built-in", True), "generated": ("Designed", True), "cloned": ("Cloned", True),
               "professional": ("Professional clone", True), "library": ("Library (paid plan)", False)}


def voice_kind(v):
    """Library voices are copies of someone else's voice; free plans can't use them through the API."""
    if (v.get("sharing") or {}).get("status") in ("copied", "copied_disabled"):
        return "library"
    cat = v.get("category") or ""
    if v.get("is_owner") is False and cat != "premade":
        return "library"
    return cat if cat in VOICE_KINDS else "library" if cat in ("famous", "high_quality") else "cloned"


ERROR_HINTS = {
    "paid_plan_required": "this is a Voice Library voice, and free ElevenLabs plans can't use those through "
                          "the API. Choose a Built-in or Designed voice with Pick… on the Characters page, "
                          "or upgrade your ElevenLabs plan",
    "quota_exceeded": "your ElevenLabs account has used up its characters for this month",
    "invalid_api_key": "the ElevenLabs API key is wrong (Voice page)",
    "voice_not_found": "ElevenLabs has no voice with that ID; choose one with Pick… on the Characters page",
}


async def api_error(r):
    text = await r.text()
    try:
        detail = json.loads(text).get("detail") or {}
    except (ValueError, AttributeError):
        detail = {}
    if not isinstance(detail, dict):
        detail = {"message": str(detail)}
    code = detail.get("code") or detail.get("status") or ("invalid_api_key" if r.status == 401 else "")
    if code in ERROR_HINTS:
        return TTSError(ERROR_HINTS[code])
    return TTSError(f"ElevenLabs answered HTTP {r.status}: {detail.get('message') or text[:200]}")


def to_discord_pcm(raw, rate=RATE):
    """16-bit mono at `rate` -> 16-bit stereo at 48 kHz."""
    a = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32)
    if not len(a):
        return b""
    n = int(len(a) * 48000 / rate)
    a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    a = np.clip(a, -32768, 32767).astype(np.int16)
    return np.repeat(a, 2).tobytes()


SPOKEN_JUNK = [
    (re.compile(r"\*\*|__"), ""),                     # **bold** keeps its words
    (re.compile(r"\*[^*\n]{1,200}\*"), " "),         # *roleplay actions*
    (re.compile(r"<a?:\w+:\d+>"), " "),              # custom discord emoji
    (re.compile(r"https?://\S+"), " link "),
    (re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]+"), " "),
    (re.compile(r"[*_~`#>|]+"), " "),
    (re.compile(r"\s+"), " "),
]


def speakable(text):
    for rx, sub in SPOKEN_JUNK:
        text = rx.sub(sub, text)
    return text.strip()
