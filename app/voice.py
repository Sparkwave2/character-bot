"""Voice chat: a character joins a voice channel, listens with Whisper and answers with ElevenLabs.

Discord encrypts voice end to end (DAVE). discord.py handles that for sending, but the
voice_recv extension predates it, so the reader is patched below to strip the DAVE layer
from received audio as well."""
import asyncio
import io
import threading
import time

import davey
import discord
from discord.ext import voice_recv
from discord.ext.voice_recv.reader import AudioReader
from discord.ext.voice_recv.rtp import OPUS_SILENCE

from .agent import run_turn
from .log import log
from .bot import now_note
from .tools import tools_tokens
from .tts import TTSError, speakable

BYTES_PER_SEC = 48000 * 2 * 2      # Discord PCM: 48 kHz, 16-bit, stereo
MIN_SPEECH = 0.4                   # seconds; shorter bursts are coughs and clicks
MAX_SPEECH = 30                    # seconds; longer speech is cut and transcribed in parts

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
                    asyncio.create_task(self._hear(user, pcm))

    async def _hear(self, user, pcm):
        try:
            text = await asyncio.to_thread(self.core.stt.transcribe, pcm, f"Talking with {self.char.name}.")
        except Exception as e:
            log(f"[voice] transcription failed: {e}", level="warn")
            return
        if self.closed or len(text) < 2:
            return
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
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()   # still waiting: wait for the rest of what they're saying
        self.reply_task = asyncio.create_task(self._reply_later())

    async def _reply_later(self):
        try:
            await asyncio.sleep(float(self.settings.get("reply_delay", 0.6)))
        except asyncio.CancelledError:
            return
        self.busy = True
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
        tools = self.bot.tools_for(voice=True)
        system = self.bot._system_prompt(self.channel, self.char, present=[self.char.name], tools=tools) + \
            VOICE_RULES.format(channel=self.channel.name)
        budget = int(lc["ctx_size"]) - int(lc["max_tokens"]) - 256 - tools_tokens(tools)
        async with self.bot.gen_lock:
            msgs = self.core.memory.build_messages(self.char.name, system, self.channel.id, budget,
                                                   int(b["image_token_cost"]), int(b["max_images_in_context"]))
            msgs.append(now_note())
            log(f"[voice] {self.char.name} is thinking in {self.channel_label}")
            llm.peak_tokens = 0
            res = await run_turn(llm, msgs, tools, self.char.name,
                                 on_react=lambda a: _no_reactions(),
                                 handlers=self.bot._local_tools(self.char))
            self.bot.note_usage(self.char.name)
        self.bot._remember_actions(self.char, self.channel.id, self.channel_label, res)
        text = "" if res.silent else speakable(res.text)
        if text and not self.closed:
            mem = self.core.memory
            mem.add(self.char.name, mem.new_entry(channel_id=self.channel.id, channel_name=self.channel_label,
                                                  author=self.char.name, char_name=self.char.name,
                                                  text=res.text, ts=time.time()))
            log(f"[voice] {self.char.name}: {text}")
            await self.say(text)
        if res.leave:
            await self.bot.leave_voice(self.channel.guild, f"{self.char.name} decided to leave {self.channel_label}.",
                                       f"👋 **{self.char.name}** left voice chat.")

    async def say(self, text):
        try:
            pcm = await self.core.tts.speak(await self.core.llm.session(), self.char.voice, text)
        except (TTSError, OSError, asyncio.TimeoutError) as e:
            log(f"[voice] {self.char.name} can't speak: {e}", level="warn")
            return
        if not pcm or self.closed or not self.vc.is_connected():
            return
        done = asyncio.Event()
        if self.vc.is_playing():
            self.vc.stop()
        self.vc.play(discord.PCMAudio(io.BytesIO(pcm)),
                     after=lambda e: self.loop.call_soon_threadsafe(done.set))
        await done.wait()

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
