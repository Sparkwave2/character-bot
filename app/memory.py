"""Per-character rolling memory.

Every character keeps one memory that spans all channels it has been in, so
history carries over when it is summoned somewhere else. Each entry is tagged
with its channel. When a prompt is built, entries are taken newest-first until
the context budget is full and everything older is dropped for good: the
context "scrolls" automatically.
"""
import json
import os
import re
import threading
import time
from collections import deque

from .config import ROOT

MEMORY_PATH = os.path.join(ROOT, "memory.json")
GIF_LINK_RE = re.compile(r"https?://\S*klipy\S*", re.I)
MEDIA_ONLY_RE = re.compile(r"[\s.]*(\[image\][\s.]*)*")


def est_tokens(text):
    return int(len(text) / 3.5) + 4


class Entry:
    __slots__ = ("ref", "msg_id", "channel_id", "channel_name", "author", "author_id",
                 "char_name", "text", "images", "reply_ref", "kind", "ts")

    def __init__(self, **kw):
        self.ref = kw.get("ref", 0)
        self.msg_id = kw.get("msg_id")          # discord message id (int) or None for events
        self.channel_id = kw.get("channel_id")
        self.channel_name = kw.get("channel_name", "")
        self.author = kw.get("author", "")      # display name
        self.author_id = kw.get("author_id")
        self.char_name = kw.get("char_name")    # set if written by one of our characters
        self.text = kw.get("text", "")
        self.images = kw.get("images", [])      # list of data URIs (not persisted)
        self.reply_ref = kw.get("reply_ref")
        self.kind = kw.get("kind", "msg")       # "msg" or "event"
        self.ts = kw.get("ts", time.time())

    def to_json(self):
        d = {k: getattr(self, k) for k in self.__slots__ if k != "images"}
        d["n_images"] = len(self.images)
        return d

    @classmethod
    def from_json(cls, d):
        e = cls(**{k: v for k, v in d.items() if k in cls.__slots__})
        e.images = []
        if d.get("n_images"):
            e.text = (e.text + " " + "[image]" * d["n_images"]).strip()
        return e


class MemoryBank:
    def __init__(self, path=MEMORY_PATH, history_limit=400):
        self.path = path
        self.history_limit = history_limit
        self.memories = {}        # char name lower -> deque[Entry]
        self.by_msg = {}          # message id -> Entry (shared objects)
        self.by_ref = {}          # ref -> Entry
        self.next_ref = 1
        self.dirty = False
        self._lock = threading.Lock()
        self.load()

    # ---- persistence --------------------------------------------------------
    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self.next_ref = data.get("next_ref", 1)
        shared = {}
        for name, entries in data.get("characters", {}).items():
            dq = deque()
            for d in entries:
                key = (d.get("msg_id"), d.get("ref"))
                e = shared.get(key) or Entry.from_json(d)
                shared[key] = e
                dq.append(e)
                self._index(e)
            self.memories[name] = dq

    def save(self):
        with self._lock:
            data = {"next_ref": self.next_ref,
                    "characters": {n: [e.to_json() for e in dq] for n, dq in self.memories.items()}}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self.dirty = False

    # ---- mutation -----------------------------------------------------------
    def _index(self, e):
        if e.msg_id:
            self.by_msg[e.msg_id] = e
        if e.ref:
            self.by_ref[e.ref] = e

    def new_entry(self, **kw):
        existing = self.by_msg.get(kw.get("msg_id")) if kw.get("msg_id") else None
        if existing:
            return existing
        e = Entry(ref=self.next_ref, **kw)
        self.next_ref += 1
        self._index(e)
        return e

    def add(self, char_name, entry):
        dq = self.memories.setdefault(char_name.lower(), deque())
        if any(x is entry for x in list(dq)[-50:]):
            return
        # keep chronological order (backfill may insert older messages)
        if dq and entry.ts < dq[-1].ts:
            items = sorted(list(dq) + [entry], key=lambda x: x.ts)
            dq.clear()
            dq.extend(items)
        else:
            dq.append(entry)
        while len(dq) > self.history_limit:
            dq.popleft()
        self.dirty = True

    def remove_message(self, msg_id):
        e = self.by_msg.pop(msg_id, None)
        if not e:
            return
        for dq in self.memories.values():
            try:
                dq.remove(e)
            except ValueError:
                pass
        self.dirty = True

    def forget(self, char_name):
        self.memories.pop(char_name.lower(), None)
        self.dirty = True

    def entries(self, char_name):
        return list(self.memories.get(char_name.lower(), ()))

    # ---- prompt building ----------------------------------------------------
    def build_messages(self, char_name, system_prompt, current_channel_id, budget,
                       image_cost, max_images):
        """Return OpenAI-style message list fitting in `budget` tokens.
        Drops entries that no longer fit from the memory permanently."""
        dq = self.memories.get(char_name.lower(), deque())
        remaining = budget - est_tokens(system_prompt)
        picked, images_left = [], max_images
        cut_index = 0
        items = list(dq)
        for i in range(len(items) - 1, -1, -1):
            e = items[i]
            text = self.render(e, char_name, current_channel_id)
            use_imgs = []
            if e.images and images_left > 0:
                use_imgs = e.images[:images_left]
            cost = est_tokens(text) + image_cost * len(use_imgs)
            if cost > remaining and picked:
                cut_index = i + 1
                break
            remaining -= cost
            images_left -= len(use_imgs)
            picked.append((e, text, use_imgs))
        if cut_index:
            for _ in range(cut_index):
                dq.popleft()
            self.dirty = True
        picked.reverse()

        msgs = [{"role": "system", "content": system_prompt}]
        for e, text, imgs in picked:
            if e.char_name and e.char_name.lower() == char_name.lower() and e.kind == "msg":
                # its own drawings and GIFs are already recorded as events ("Mayo drew a picture"); a bare
                # "..." or GIF link as its own message would only teach it to write those instead
                if MEDIA_ONLY_RE.fullmatch(e.text or "") or GIF_LINK_RE.fullmatch(e.text.strip()):
                    continue
                msgs.append({"role": "assistant", "content": e.text or "..."})
                continue
            if imgs:
                parts = [{"type": "text", "text": text}]
                parts += [{"type": "image_url", "image_url": {"url": u}} for u in imgs]
                msgs.append({"role": "user", "content": parts})
            else:
                msgs.append({"role": "user", "content": text})
        if msgs[-1]["role"] == "assistant":  # the server would treat it as a prefill
            msgs.append({"role": "user", "content": "[system] Continue the conversation naturally."})
        return msgs

    def render(self, e, viewer, current_channel_id):
        where = "" if e.channel_id == current_channel_id else f" in #{e.channel_name}"
        if e.kind == "event":
            return f"[event{where}] {e.text}"
        who = e.author
        if e.char_name:
            who = f"{e.char_name} (character)"
        reply = f" (replying to #{e.reply_ref})" if e.reply_ref else ""
        n_img = len(e.images)
        img_note = f" [attached {n_img} image{'s' if n_img > 1 else ''}]" if n_img else ""
        return f"[#{e.ref}{where}] {who}{reply}: {e.text}{img_note}"
