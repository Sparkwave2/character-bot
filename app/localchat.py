"""A chat channel that lives in the app window instead of Discord. Characters are summoned into it and
talk exactly like they do on Discord (same memory, prompt and commands), minus the commands that need
Discord (reactions, profile pictures, GIFs). The transcript and its attachments are kept in ./localchat."""
import asyncio
import getpass
import json
import os
import re
import threading
import time
from types import SimpleNamespace

import aiohttp

from .chat import ChatLogic, safe_filename
from .config import ROOT
from .log import log
from .media import encode_image

LOCAL_DIR = os.path.join(ROOT, "localchat")
LOCAL_FILES = os.path.join(LOCAL_DIR, "files")
TRANSCRIPT = os.path.join(LOCAL_DIR, "chat.json")
USER_AVATAR = os.path.join(LOCAL_DIR, "you.png")    # your picture in the local chat (Chat page)
LOCAL_ID = 1                 # the channel id in the characters' memory (Discord ids are far bigger)
MAX_MESSAGES = 500
MAX_GIF = 15_000_000
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
COMMAND_RE = re.compile(r"^\s*/(summon|dismiss|gohome|forget|who|clear)\b[ \t]*([^\n]*)", re.I)


def is_image(path):
    return path.lower().endswith(IMAGE_EXT)


def _readable(path, limit=6000):
    """A small text file's contents (after a line break); '' for anything else."""
    try:
        with open(path, "rb") as f:
            data = f.read(limit + 1)
        text = data[:limit].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    if "\x00" in text:
        return ""
    return "\n" + text + ("\n[... the rest is cut off]" if len(data) > limit else "")


class LocalChat(ChatLogic):
    log_tag = "chat"
    excluded_tools = ("add_reaction", "view_profile_picture")

    def __init__(self, core):
        self.core, self.cfg = core, core.cfg
        self.channel = SimpleNamespace(id=LOCAL_ID, name="local-chat",
                                       guild=SimpleNamespace(name="Character Bot (local chat)"))
        self.members = {}        # char lower -> {"name"}
        self.recent, self.phase, self.tasks, self.followup = {}, {}, {}, {}
        self.messages = []       # what the window shows: {kind: msg|notice, author, char, text, files, ts}
        self.cleared = 0         # bumped when the transcript is replaced, so the window redraws it all
        self.typing = None       # name of the character writing right now
        self._entries = {}       # message id -> memory Entry, made the first time it's needed
        self._session = None
        self._lock = threading.Lock()
        self.load()

    @property
    def user_name(self):
        return (self.cfg["localchat"].get("user_name") or "").strip() or getpass.getuser() or "User"

    @property
    def http_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def present(self, channel_id):
        return self.members if channel_id == LOCAL_ID else {}

    # ---- transcript ---------------------------------------------------------
    def load(self):
        try:
            with open(TRANSCRIPT, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.messages = data.get("messages", [])
        self.members = {n.lower(): {"name": n} for n in data.get("members", [])}

    def save(self):
        with self._lock:
            os.makedirs(LOCAL_DIR, exist_ok=True)
            tmp = TRANSCRIPT + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"messages": self.messages[-MAX_MESSAGES:],
                           "members": [i["name"] for i in self.members.values()]}, f, ensure_ascii=False)
            os.replace(tmp, TRANSCRIPT)

    def _post(self, author, text="", char=False, files=(), kind="msg"):
        msg = {"id": time.time_ns(), "kind": kind, "author": author, "char": char, "text": text, "files": list(files),
               "ts": time.time()}
        self.messages.append(msg)
        if len(self.messages) > MAX_MESSAGES + 50:
            del self.messages[:-MAX_MESSAGES]
            self.cleared += 1
        self.save()
        return msg

    def notice(self, text):
        """A line from the app itself; characters never see these."""
        self._post("", text, kind="notice")

    def _store(self, name, data):
        os.makedirs(LOCAL_FILES, exist_ok=True)
        name, _ = safe_filename(name)
        fname = f"{int(time.time() * 1000)}_{name}"
        with open(os.path.join(LOCAL_FILES, fname), "wb") as f:
            f.write(data)
        return fname

    def _entry(self, msg, with_images=True):
        """The memory entry for a transcript message (images are read back from disk when needed)."""
        key = msg.get("id") or msg["ts"]
        e = self._entries.get(key)
        if e:
            return e
        side = int(self.behavior["max_image_side"])
        images, names = [], []
        for fname in msg["files"]:
            path = os.path.join(LOCAL_FILES, fname)
            if is_image(fname):
                if with_images and os.path.isfile(path):
                    with open(path, "rb") as f:
                        uri = encode_image(f.read(), side)
                    if uri:
                        images.append(uri)
            else:
                names.append(fname.split("_", 1)[-1])
                if not msg["char"]:          # characters can read the text files people send
                    names[-1] += _readable(path)
        text = (msg["text"] + "\n" + "\n".join(f"[attached file: {n}]" for n in names)).strip()
        e = self.core.memory.new_entry(channel_id=LOCAL_ID, channel_name=self.channel.name, author=msg["author"],
                                       char_name=msg["author"] if msg["char"] else None, text=text,
                                       images=images, ts=msg["ts"])
        self._entries[key] = e
        return e

    # ---- what the window calls ----------------------------------------------
    async def send(self, text, files=()):
        """`files` is [(name, bytes)] attached by the user."""
        text = (text or "").strip()
        m = COMMAND_RE.match(text)
        if m and not files:
            return await self.command(m.group(1).lower(), m.group(2).strip())
        if not text and not files:
            return
        msg = self._post(self.user_name, text, files=[self._store(n, d) for n, d in files])
        if text.startswith("//"):        # out of character: shown, but the characters don't see it
            return
        e = self._entry(msg)
        if self.behavior.get("link_previews", True):
            seen = await self._link_previews(text)
            if seen:
                e.text = (e.text + "\n" + seen).strip()
        if self.members:
            self._witness(LOCAL_ID, e)
            self.schedule(self.channel)
        else:
            self.recent.setdefault(LOCAL_ID, []).append(e)

    async def command(self, cmd, arg):
        if cmd == "summon":
            return await self.summon(arg) if arg else self.notice("Usage: /summon <character>")
        if cmd == "dismiss":
            if arg.lower() == "all":
                return await self.gohome()
            return await self.dismiss(arg) if arg else self.notice("Usage: /dismiss <character>")
        if cmd == "gohome":
            return await self.gohome()
        if cmd == "forget":
            return await self.forget()
        if cmd == "clear":
            return self.clear()
        names = [i["name"] for i in self.members.values()]
        self.notice(f"**In #{self.channel.name}:** {', '.join(names)}" if names else
                    "Nobody is here. Use the **Summon** button or `/summon <character>`.")

    async def summon(self, name):
        char = self.core.characters.get(name)
        if not char:
            return self.notice(f"No character named **{name}**.")
        if char.name.lower() in self.members:
            return self.notice(f"**{char.name}** is already here.")
        other = self.form_here(LOCAL_ID, char)
        if other:
            return self.notice(f"**{char.name}** is already here as **{other}** (they're forms of the same character).")
        self.members[char.name.lower()] = {"name": char.name}
        self.notice(f"✨ **{char.name}** has been summoned.")
        self._backfill(char.name)
        ev = self.core.memory.new_entry(kind="event", channel_id=LOCAL_ID, channel_name=self.channel.name,
                                        text=f"{self.user_name} summoned {char.name} into #{self.channel.name}.")
        self._witness(LOCAL_ID, ev)
        log(f"[chat] {char.name} summoned into the local chat")
        self.schedule(self.channel, force=char.name)

    def _backfill(self, char_name):
        n = int(self.behavior["backfill_messages"])
        msgs = [m for m in self.messages if m["kind"] == "msg" and not m["text"].startswith("//")][-n:] if n > 0 else []
        rec = self.recent.setdefault(LOCAL_ID, [])
        for i, m in enumerate(msgs):
            e = self._entry(m, with_images=i >= len(msgs) - 3)
            self.core.memory.add(char_name, e)
            if e not in rec:
                rec.append(e)
        self.recent[LOCAL_ID] = sorted(rec, key=lambda x: x.ts)[-30:]

    async def dismiss(self, name):
        info = self.members.get(name.lower())
        if not info:
            return self.notice(f"**{name}** isn't here.")
        self.notice(f"👋 Dismissed **{info['name']}**.")
        await self.remove_character(self.channel, info["name"],
                                    f"{self.user_name} dismissed {info['name']} from #{self.channel.name}.")

    async def gohome(self):
        names = [i["name"] for i in self.members.values()]
        if not names:
            return self.notice("Nobody is here.")
        self.notice("👋 Dismissed " + ", ".join(f"**{n}**" for n in names) + ".")
        for n in names:
            await self.remove_character(self.channel, n, f"{self.user_name} dismissed {n} from #{self.channel.name}.")

    async def forget(self):
        """Like on Discord: everyone goes home first, then their memory is wiped."""
        names = [i["name"] for i in self.members.values()]
        if not names:
            return self.notice("Nobody is here.")
        task = self.tasks.pop(LOCAL_ID, None)
        if task:
            task.cancel()
        for n in names:
            await self.remove_character(self.channel, n, f"{n} was sent home from #{self.channel.name}.")
            self.core.memory.forget(n)
        self.recent.pop(LOCAL_ID, None)
        self.core.memory.save()
        self.notice("🧹 " + ", ".join(f"**{n}**" for n in names) + " went home and forgot everything.")
        log(f"[chat] {', '.join(names)} went home and forgot everything")

    def clear(self):
        """Empties the window (the characters' memory is untouched)."""
        self.messages, self._entries = [], {}
        self.recent.pop(LOCAL_ID, None)
        self.cleared += 1
        self.save()

    def context_report(self):
        size = int(self.cfg["llm"]["ctx_size"])
        out = []
        for info in list(self.members.values()):
            char = self.core.characters.get(info["name"])
            if char:
                last = self.core.context_used.get(char.name.lower())
                out.append((char.name, self.channel.name, self.context_estimate(self.channel, char),
                            last[0] if last else None, size))
        return out

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ---- what ChatLogic needs -----------------------------------------------
    async def _send(self, channel, info, char, text="", file=None):
        msg = self._post(char.name, text, char=True)
        self._witness(LOCAL_ID, self._entry(msg))
        self.schedule(self.channel)       # another character may answer this one

    async def post_file(self, channel, info, char, name, data):
        msg = self._post(char.name, char=True, files=[self._store(name, data)])
        self._witness(LOCAL_ID, self._entry(msg))

    async def post_gif(self, channel, info, char, url, title):
        """The window can't show a link as a GIF, so the GIF is downloaded and posted as a picture."""
        data = b""
        try:
            async with self.http_session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    async for chunk in r.content.iter_chunked(1 << 16):   # read() alone stops at what has arrived
                        data += chunk
                        if len(data) > MAX_GIF:
                            break
        except (aiohttp.ClientError, asyncio.TimeoutError):
            data = b""
        if not data or len(data) > MAX_GIF:
            log(f"[gifs] couldn't download the GIF for {char.name}; posting its link", level="warn")
            return await self._send(channel, info, char, url)
        await self.post_file(channel, info, char, (re.sub(r"[^\w\-]+", "_", title).strip("_")[:40] or "gif") + ".gif",
                             data)

    async def _typing(self, channel, char=None):
        self.typing = char.name if char else None
        try:
            await asyncio.Event().wait()
        finally:
            self.typing = None

    async def swap_form(self, channel, old, new):
        if not self.members.pop(old.name.lower(), None):
            return False
        self.members[new.name.lower()] = {"name": new.name}
        self.save()
        return True

    async def remove_character(self, channel, name, reason):
        info = self.members.pop(name.lower(), None)
        if not info:
            return
        self.save()
        ev = self.core.memory.new_entry(kind="event", channel_id=LOCAL_ID, channel_name=self.channel.name,
                                        text=reason)
        self.core.memory.add(name, ev)
        self._witness(LOCAL_ID, ev)
        log(f"[chat] {name} left the local chat")

    async def announce(self, channel, text):
        self.notice(text)
