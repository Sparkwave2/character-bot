"""Voice chat: a character joins a voice channel, listens with Whisper and answers with ElevenLabs
or the local voice (Kokoro).

Discord encrypts voice end to end (DAVE). discord.py handles that for sending, but the
voice_recv extension predates it, so the reader is patched below to strip the DAVE layer
from received audio as well."""
import asyncio
import threading
import time

import aiohttp
import davey
import discord
from discord.ext import voice_recv
from discord.ext.voice_recv.reader import AudioReader
from discord.ext.voice_recv.rtp import OPUS_SILENCE

from .agent import run_turn
from .log import log
from .tools import tools_tokens
from .tts import TTSError, speakable

BYTES_PER_SEC = 48000 * 2 * 2      # Discord PCM: 48 kHz, 16-bit, stereo
MIN_SPEECH = 0.4                   # seconds; shorter bursts are coughs and clicks
MAX_SPEECH = 30                    # seconds; longer speech is cut and transcribed in parts
TALK_OVER = 0.4                    # seconds of someone talking over the character before it stops
CHARS_PER_SEC = 14                 # rough speaking rate, to guess how much of a reply was said

VOICE_RULES = """
You're now in voice channel "{channel}". What people say is transcribed (may contain mistakes) and your reply
is spoken aloud: talk naturally in 1-3 short sentences, with no emojis, markdown, links or *actions*."""


def _patch_reader_for_dave():
    original = AudioReader.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        transport = self.decryptor.decrypt_rtp
        vc = self.voice_client

        def decrypt(packet):
            data = transport(packet)
            conn = vc._connection
            dave = getattr(conn, "dave_session", None)
            if not dave or not getattr(conn, "dave_protocol_version", 0) or data == OPUS_SILENCE:
                return data
            uid = vc._ssrc_to_id.get(packet.ssrc)
            if uid is None:
                return OPUS_SILENCE
            try:
                return dave.decrypt(uid, davey.MediaType.audio, data)
            except Exception:
                return data if dave.can_passthrough(uid) else OPUS_SILENCE

        self.decryptor.decrypt_rtp = decrypt

    AudioReader.__init__ = init


_patch_reader_for_dave()


class StreamSource(discord.AudioSource):
    """PCM that is still arriving: plays what's there, and silence while the next part is made."""
    FRAME = 3840                       # 20 ms of Discord PCM

    def __init__(self):
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.finished = False
        self.started = False
        self.added = self.played = 0    # bytes

    def add(self, pcm):
        with self.lock:
            self.buf += pcm
            self.added += len(pcm)

    def finish(self):
        self.finished = True

    def read(self):
        with self.lock:
            if len(self.buf) >= self.FRAME or (self.finished and self.buf):
                out = bytes(self.buf[:self.FRAME]).ljust(self.FRAME, b"\0")
                self.played += min(len(self.buf), self.FRAME)
                del self.buf[:self.FRAME]
                return out
        return b"" if self.finished else b"\0" * self.FRAME

    def is_opus(self):
        return False


def log_timing(name, t, engine):
    """Where the time went between someone going quiet and the character starting to talk."""
    parts = []
    if "ended" in t:
        parts += [f"pause detection {t['pause']:.1f}", f"Whisper {t['whisper']:.1f}"]
    if t.get("queue", 0) >= 0.1:
        parts.append(f"waiting for a text-chat reply {t['queue']:.1f}")
    w = t.get("work") or {}
    extra = f", {w['calls']} model calls" if w.get("calls", 1) > 1 else ""
    parts.append(f"model {t['model']:.1f} ({w.get('prompt_n', 0)} prompt tokens read, "
                 f"{w.get('cache_n', 0)} from cache, {w.get('predicted_n', 0)} written{extra})")
    parts.append(f"{engine} {t['voice']:.1f}")
    total = f" {time.monotonic() - t['ended']:.1f} s after the speaker stopped" if "ended" in t else ""
    log(f"[voice] {name} started talking{total}: " + ", ".join(parts))


class Listener(voice_recv.AudioSink):
    """Runs on the voice reader thread; hands decoded PCM to the session."""

    def __init__(self, session):
        super().__init__()
        self.session = session

    def wants_opus(self):
        return False

    def write(self, user, data):
        if user is not None and not user.bot and data.pcm:
            self.session.feed(user, data.pcm)

    def cleanup(self):
        pass


class VoiceSession:
    """One character in one voice channel (a bot can be in one voice channel per server)."""

    def __init__(self, bot, char, vc, text_channel=None):
        self.bot, self.core, self.char, self.vc = bot, bot.core, char, vc
        self.text_channel = text_channel   # where /join was used; notices go there
        self.loop = asyncio.get_running_loop()
        self.buffers = {}               # user id -> [pcm bytearray, last packet time, user]
        self.lock = threading.Lock()
        self.reply_task = None
        self.busy = False               # generating or speaking
        self.again = False              # someone spoke meanwhile; reply once more afterwards
        self.timing = None              # how long hearing the last utterance took, for the log
        self.source = None              # what the character is saying right now (StreamSource)
        self.said = None                # (memory entry, spoken text) of that reply
        self.cut = False                # the reply was interrupted; stop making the rest of it
        self.closed = False
        self.watch_task = asyncio.create_task(self._watch())
        vc.listen(Listener(self))

    @property
    def settings(self):
        return self.core.cfg["voice"]

    @property
    def channel(self):
        return self.vc.channel

    @property
    def channel_label(self):
        return f"{self.channel.name} (voice)"

    # ---- hearing ------------------------------------------------------------
    def feed(self, user, pcm):
        now = time.monotonic()
        with self.lock:
            b = self.buffers.get(user.id)
            if b is None:
                b = self.buffers[user.id] = [bytearray(), now, user]
            b[0] += pcm
            b[1] = now
            talking_over = len(b[0]) >= TALK_OVER * BYTES_PER_SEC
        if talking_over and self.source and not self.cut and self.settings.get("interruptible", True):
            self.cut = True
            self.loop.call_soon_threadsafe(self._interrupt, user)

    # ---- being interrupted ---------------------------------------------------
    def _interrupt(self, user):
        """Someone talks over the character: it stops at once (the rest of the reply is dropped) and
        answers what they say once it's transcribed. Only the part it really said stays in its memory."""
        src, said = self.source, self.said
        if not src:
            return
        if said:
            entry, text = said
            total = src.added if src.finished else max(src.added, len(text) / CHARS_PER_SEC * BYTES_PER_SEC)
            keep = int(len(text) * min(1.0, src.played / max(total, 1)))
            spoken = text[:keep].rsplit(" ", 1)[0] if 0 < keep < len(text) else text[:keep]
            if keep < len(text):
                entry.text = spoken + "— [interrupted]"
                self.core.memory.dirty = True
            log(f"[voice] {getattr(user, 'display_name', user)} interrupted {self.char.name} after: "
                f"{spoken or '(nothing)'}")
        self.vc.stop()

    async def _watch(self):
        """Discord stops sending packets when someone stops talking; a gap ends the utterance."""
        while not self.closed:
            await asyncio.sleep(0.1)
            gap = float(self.settings.get("silence_seconds", 0.8))
            now, done = time.monotonic(), []
            with self.lock:
                for uid, (buf, last, user) in list(self.buffers.items()):
                    if now - last >= gap or len(buf) >= MAX_SPEECH * BYTES_PER_SEC:
                        del self.buffers[uid]
                        done.append((user, bytes(buf)))
            for user, pcm in done:
                if len(pcm) >= MIN_SPEECH * BYTES_PER_SEC:
                    asyncio.create_task(self._hear(user, pcm, now - gap))

    async def _hear(self, user, pcm, ended):
        t = time.monotonic()
        try:
            text = await asyncio.to_thread(self.core.stt.transcribe, pcm, f"Talking with {self.char.name}.")
        except Exception as e:
            log(f"[voice] transcription failed: {e}", level="warn")
            return
        if self.closed or len(text) < 2:
            return
        self.timing = {"ended": ended, "pause": t - ended, "whisper": time.monotonic() - t}
        name = getattr(user, "display_name", None) or user.name
        self.bot.names[user.id] = name
        log(f"[voice] {name}: {text}")
        mem = self.core.memory
        mem.add(self.char.name, mem.new_entry(channel_id=self.channel.id, channel_name=self.channel_label,
                                              author=name, author_id=user.id, text=text, ts=time.time()))
        self.schedule()

    # ---- talking ------------------------------------------------------------
    def schedule(self):
        if self.busy:
            self.again = True
            return
        self.busy = True               # no extra wait: the pause that ended the speech was enough
        self.reply_task = asyncio.create_task(self._reply_now())

    async def _reply_now(self):
        try:
            await self._reply()
        except Exception as e:
            log(f"[voice] {self.char.name} couldn't answer: {e!r}", level="warn")
        finally:
            self.busy = False
            if self.again and not self.closed:
                self.again = False
                self.schedule()

    async def _reply(self):
        llm, b, lc = self.core.llm, self.bot.behavior, self.core.cfg["llm"]
        if not llm.ready:
            log(f"[voice] {self.char.name} can't answer: model not loaded")
            return
        tools = self.bot.tools_for(voice=True, char=self.char)
        system = self.bot._system_prompt(self.channel, self.char, present=[self.char.name], tools=tools) + \
            VOICE_RULES.format(channel=self.channel.name)
        budget = int(lc["ctx_size"]) - int(lc["max_tokens"]) - 256 - tools_tokens(tools)
        timing, self.timing = self.timing or {}, None
        t = time.monotonic()
        async with self.bot.gen_lock:
            timing["queue"] = time.monotonic() - t   # a text-chat reply was being written
            t = time.monotonic()
            msgs = self.core.memory.build_messages(self.char.name, system, self.channel.id, budget,
                                                   int(b["image_token_cost"]), int(b["max_images_in_context"]))
            log(f"[voice] {self.char.name} is thinking in {self.channel_label}")
            llm.peak_tokens, llm.work = 0, {}
            res = await run_turn(llm, msgs, tools, self.char.name,
                                 on_react=lambda a: _no_reactions(),
                                 handlers=self.bot._local_tools(self.char))
            self.bot.note_usage(self.char.name)
            timing["model"], timing["work"] = time.monotonic() - t, dict(llm.work)
        self.bot._remember_actions(self.char, self.channel.id, self.channel_label, res)
        text = "" if res.silent else speakable(res.text)
        if text and self.again and self.settings.get("interruptible", True):
            log(f"[voice] {self.char.name} drops its answer: someone said something new while it was thinking")
            return
        if text and not self.closed:
            mem = self.core.memory
            entry = mem.new_entry(channel_id=self.channel.id, channel_name=self.channel_label,
                                  author=self.char.name, char_name=self.char.name, text=res.text, ts=time.time())
            mem.add(self.char.name, entry)
            log(f"[voice] {self.char.name}: {text}")
            self.said = (entry, text)
            try:
                await self.say(text, timing)
            finally:
                self.said = None
        if res.leave:
            await self.bot.leave_voice(self.channel.guild, f"{self.char.name} decided to leave {self.channel_label}.",
                                       f"👋 **{self.char.name}** left voice chat.")
        elif res.transform and not self.closed:
            new = self.core.characters.get(res.transform)
            if new:
                old, self.char = self.char, new        # its voice changes with it
                mem = self.core.memory
                mem.add(new.name, mem.new_entry(kind="event", channel_id=self.channel.id,
                                                channel_name=self.channel_label,
                                                text=f"{old.name} transformed into {new.name}."))
                log(f"[voice] {old.name} transformed into {new.name}")
                await self.bot.announce(self.text_channel, f"✨ **{old.name}** transformed into **{new.name}**.")
                self.again = True                       # the new form speaks next

    def _speech(self, text):
        if self.settings.get("local_tts"):
            return self.core.local_tts.stream(self.core.local_tts.voice_for(self.char), text), "local voice"
        engine = self.core.tts

        async def online():
            async for pcm in engine.stream(await self.core.llm.session(), self.char.voice, text):
                yield pcm
        return online(), "ElevenLabs " + (self.settings.get("tts_model") or "")

    async def say(self, text, timing=None):
        """Plays the reply while it is still being turned into speech."""
        source, done, t = StreamSource(), asyncio.Event(), time.monotonic()
        speech, engine = self._speech(text)
        self.cut = False
        try:
            async for pcm in speech:
                if self.closed or self.cut or not self.vc.is_connected():
                    break
                source.add(pcm)
                if not source.started:
                    source.started = True
                    if timing is not None:
                        timing["voice"] = time.monotonic() - t
                        log_timing(self.char.name, timing, engine)
                    if self.vc.is_playing():
                        self.vc.stop()
                    self.source = source
                    self.vc.play(source, after=lambda e: self.loop.call_soon_threadsafe(done.set))
        except (TTSError, OSError, asyncio.TimeoutError, aiohttp.ClientError) as e:
            log(f"[voice] {self.char.name} can't speak: {e}", level="warn")
        finally:
            source.finish()
        try:
            if source.started:
                await done.wait()
        finally:
            self.source = None

    async def close(self):
        self.closed = True
        self.watch_task.cancel()
        if self.reply_task:
            self.reply_task.cancel()
        try:
            if self.vc.is_listening():
                self.vc.stop_listening()
            await self.vc.disconnect(force=True)
        except Exception:
            pass


async def _no_reactions():
    return "You can't react in voice chat."
