"""Text to speech, returned as 48 kHz stereo PCM ready for Discord: ElevenLabs (online) or
Kokoro (local, runs on the GPU when it can). Both are streamed, so playback starts while the
rest of the reply is still being made."""
import asyncio
import json
import os
import re
import threading
import urllib.request

import aiohttp
import numpy as np

from .config import ROOT
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

    async def stream(self, session, voice, text):
        """Yields Discord PCM as ElevenLabs sends it."""
        if not self.key:
            raise TTSError("no ElevenLabs API key set")
        vid = await self.resolve(session, voice)
        body = {"text": text, "model_id": self.cfg["voice"].get("tts_model") or "eleven_flash_v2_5"}
        async with session.post(f"{API}/text-to-speech/{vid}/stream", params={"output_format": f"pcm_{RATE}"},
                                json=body, headers={"xi-api-key": self.key},
                                timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                raise await api_error(r)
            carry = b""
            async for chunk in r.content.iter_any():
                data = carry + chunk
                even = len(data) // 2 * 2
                carry = data[even:]
                if even:
                    yield to_discord_pcm(data[:even])


KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
# first letter of a Kokoro voice = language, second = female/male
KOKORO_LANGS = {"a": ("en-us", "American English"), "b": ("en-gb", "British English"), "e": ("es", "Spanish"),
                "f": ("fr-fr", "French"), "h": ("hi", "Hindi"), "i": ("it", "Italian"), "j": ("ja", "Japanese"),
                "p": ("pt-br", "Portuguese"), "z": ("cmn", "Chinese")}
KOKORO_VOICES = (
    "af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
    "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa "
    "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis ef_dora em_alex em_santa "
    "ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola "
    "jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo pf_dora pm_alex pm_santa "
    "zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang").split()


def kokoro_label(voice):
    lang = KOKORO_LANGS.get(voice[:1], ("", "?"))[1]
    return f"{voice[3:].capitalize()}   ·   {lang}, {'female' if voice[1:2] == 'f' else 'male'}"


class LocalTTS:
    """Kokoro-82M through onnxruntime. Loads on first use and stays loaded (~0.5 GB of VRAM on the GPU)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.kokoro = None
        self.loaded_as = None
        self._lock = threading.Lock()

    def _settings(self):
        v = self.cfg["voice"]
        return (v.get("local_tts_device") or "auto",
                v.get("local_tts_dir") or os.path.join(ROOT, "models", "kokoro"))

    def voice_for(self, char):
        v = (getattr(char, "local_voice", "") or self.cfg["voice"].get("local_voice") or "af_heart").strip()
        return v if v in KOKORO_VOICES else "af_heart"

    def _download(self, folder):
        os.makedirs(folder, exist_ok=True)
        for name in KOKORO_FILES:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                continue
            log(f"[voice] downloading {name} for the local voice")
            tmp = path + ".part"
            urllib.request.urlretrieve(KOKORO_URL + name, tmp)
            os.replace(tmp, path)

    def load(self):
        """Blocking. Downloads the model (~350 MB) the first time."""
        with self._lock:
            device, folder = self._settings()
            if self.kokoro is not None and self.loaded_as[:2] == (device, folder):
                return
            try:
                import onnxruntime as ort
                from kokoro_onnx import Kokoro
            except ImportError:
                raise TTSError("the local voice needs kokoro-onnx (see the README: Voice chat)")
            ort.set_default_logger_severity(3)   # its warnings are harmless and noisy
            from .stt import _add_cuda_dlls
            self._download(folder)
            _add_cuda_dlls()
            model, voices = (os.path.join(folder, n) for n in KOKORO_FILES)
            want_gpu = device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
            providers = ([("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})] if want_gpu else [])                 + ["CPUExecutionProvider"]
            k = Kokoro.from_session(ort.InferenceSession(model, providers=providers), voices)
            k.create("Hello.", voice="af_heart", lang="en-us")   # missing CUDA DLLs only show up on first use
            used = "GPU" if "CUDAExecutionProvider" in k.sess.get_providers() else "CPU"
            self.kokoro, self.loaded_as = k, (device, folder, used)
            log(f"[voice] local voice (Kokoro) ready on the {used}")

    def synth(self, voice, text):
        """Blocking: text -> Discord PCM."""
        self.load()
        speed = float(self.cfg["voice"].get("local_speed") or 1.0)
        audio, rate = self.kokoro.create(text, voice=voice, speed=speed, lang=KOKORO_LANGS[voice[0]][0])
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        return to_discord_pcm(pcm, rate)

    async def stream(self, voice, text):
        """One sentence (or a few short ones) at a time, so the first one plays almost at once."""
        for part in speech_chunks(text):
            yield await asyncio.to_thread(self.synth, voice, part)


SENTENCE_END_RE = re.compile(r"(?<=[.!?…。！？])\s+")


def speech_chunks(text, first=1, size=220):
    """Sentences grouped into chunks: the first chunk is short so it can start playing early."""
    out, cur = [], ""
    for s in SENTENCE_END_RE.split(text.strip()):
        cur = f"{cur} {s}".strip()
        if (len(out) < first and len(cur) >= 12) or len(cur) >= size:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


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
