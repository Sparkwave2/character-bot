"""Owns the asyncio loop (in a background thread) and every service.
The GUI only talks to this object."""
import asyncio
import threading

import discord

from .bot import CharacterBot
from .characters import CharacterStore
from .config import Config
from .draw import ImageGen
from .engine import Engine
from .games import GameWatcher
from .llm import LlamaServer
from .localchat import LocalChat
from .log import log
from .memory import MemoryBank
from .models import Downloader
from .stt import Transcriber
from .tts import ElevenLabs, LocalTTS


class Core:
    def __init__(self):
        self.cfg = Config()
        self.characters = CharacterStore()
        self.memory = MemoryBank(history_limit=int(self.cfg["behavior"]["history_limit"]))
        self.memory.set_links(self.characters.groups())   # linked forms share one memory
        # later changes (the Characters page) are applied on the core loop, where the memory is used
        self.characters.on_reload = lambda groups: self.loop.call_soon_threadsafe(self.memory.set_links, groups)
        self.engine = Engine(self.cfg)
        self.llm = LlamaServer(self.cfg, self.engine)
        self.games = GameWatcher(self)
        self.stt = Transcriber(self.cfg)
        self.tts = ElevenLabs(self.cfg)
        self.local_tts = LocalTTS(self.cfg)
        self.draw = ImageGen(self.cfg)
        self.downloader = Downloader(self.cfg)
        self.gen_lock = asyncio.Lock()   # one character generates at a time (Discord, voice and local chat)
        self.context_used = {}           # char lower -> (tokens of its last reply's biggest call, time)
        self.localchat = LocalChat(self)
        self.bot = None
        self.bot_task = None
        self.want_loaded = False   # user intent; game watcher respects it
        self.discord_status = "disconnected"
        self.on_discord = lambda: None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self.games.run())
        self.loop.create_task(self._autosave())
        self.loop.run_forever()

    def submit(self, coro):
        """Run a coroutine on the core loop from any thread."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _autosave(self):
        while True:
            await asyncio.sleep(30)
            if self.memory.dirty:
                try:
                    self.memory.save()
                except OSError as e:
                    log(f"[memory] save failed: {e}")

    # ---- model --------------------------------------------------------------
    async def load_model(self):
        self.want_loaded = True
        if self.games.active_game:
            self.llm.suspended_by = self.games.active_game
            log(f"[llm] {self.games.active_game} is running; model will load when it closes")
            return
        if not self.engine.server_path():
            log("[engine] llama.cpp is not installed yet; installing it first")
            if not await self.install_engine():
                return
        try:
            await self.llm.start()
        except RuntimeError as e:
            log(f"[llm] {e}")

    async def install_engine(self, force=False):
        """Install or update the managed llama.cpp. Restarts the model if it was running."""
        old_path = self.engine.server_path()
        try:
            # the new build goes into its own folder, so the model keeps running meanwhile
            new_path = await self.engine.install(await self.llm.session(), force=force)
        except Exception as e:
            self.engine.progress = (0.0, f"Install failed: {e}")
            log(f"[engine] install failed: {e}")
            return False
        if new_path != old_path and self.llm.state != self.llm.STOPPED:
            log("[engine] restarting the model on the new build")
            await self.llm.stop()
            if self.want_loaded and not self.llm.suspended_by:
                try:
                    await self.llm.start()
                except RuntimeError as e:
                    log(f"[llm] {e}")
        self.engine.cleanup()
        return True

    async def startup_engine(self):
        """Detect hardware, then update the managed build if enabled."""
        await asyncio.to_thread(self.engine.hardware)
        if not self.engine.installed or self.cfg["llm"].get("server_path"):
            return
        await self.check_engine_update()
        if self.engine.update_available and self.cfg["llm"].get("auto_update", True):
            await self.install_engine()

    async def check_engine_update(self):
        try:
            tag = await self.engine.check_update(await self.llm.session())
        except Exception as e:
            log(f"[engine] update check failed: {e}")
            return
        if self.engine.update_available:
            log(f"[engine] llama.cpp {tag} is available (installed: {self.engine.manifest.get('tag')})")
        elif self.engine.installed:
            log(f"[engine] llama.cpp is up to date ({tag})")

    async def download_model(self, model):
        """Downloads a model from the picker and makes it the one that loads (restarting it if loaded)."""
        model_path, mmproj_path = await self.downloader.download(await self.llm.session(), model)
        c = self.cfg["llm"]
        c["model_path"], c["mmproj_path"] = model_path, mmproj_path
        self.cfg.save()
        log(f"[models] {model.name} is now the model")
        if self.llm.state != self.llm.STOPPED:
            await self.llm.stop()
            await self.load_model()

    def context_report(self):
        """Context meter rows for every summoned character, on Discord and in the local chat."""
        rows = self.localchat.context_report()
        if self.bot and self.bot.is_ready():
            rows = self.bot.context_report() + rows
        return rows

    async def unload_model(self):
        self.want_loaded = False
        self.llm.suspended_by = None
        await self.llm.stop()

    # ---- discord ------------------------------------------------------------
    async def connect_discord(self):
        if self.bot and not self.bot.is_closed():
            return
        token = self.cfg["discord"]["token"].strip()
        if not token:
            log("[discord] no bot token set")
            return
        self.bot = CharacterBot(self)
        self.bot.on_change = self.on_discord
        self._set_discord("connecting")

        async def runner():
            try:
                await self.bot.start(token)
            except discord.LoginFailure:
                log("[discord] login failed: invalid token")
            except discord.PrivilegedIntentsRequired:
                log("[discord] enable the MESSAGE CONTENT intent for your bot in the Developer Portal")
            except Exception as e:
                log(f"[discord] stopped: {e!r}")
            finally:
                self._set_discord("disconnected")

        self.bot_task = self.loop.create_task(runner())
        for _ in range(60):
            await asyncio.sleep(0.5)
            if self.bot.is_ready():
                self._set_discord("connected")
                return
            if self.bot_task.done():
                return

    async def disconnect_discord(self):
        if self.bot:
            await self.bot.close()
        self._set_discord("disconnected")

    def _set_discord(self, s):
        self.discord_status = s
        try:
            self.on_discord()
        except Exception:
            pass

    async def shutdown(self):
        try:
            self.memory.save()
        except OSError:
            pass
        if self.bot and not self.bot.is_closed():
            await self.bot.close()
        await self.draw.stop()
        await self.localchat.close()
        await self.llm.close()
