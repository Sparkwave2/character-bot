"""Each character's personal notes: Markdown files in ./notes/<character>/."""
import os
import re
from datetime import datetime

from .config import ROOT

NOTES_DIR = os.path.join(ROOT, "notes")
MAX_NOTE_CHARS = 8000     # a single note never grows past this
MAX_NOTES = 50


def note_filename(name):
    """'Shopping list' -> 'shopping_list.md'; keeps it inside the folder."""
    base = re.sub(r"\.md$", "", str(name or "").strip(), flags=re.I)
    base = re.sub(r"[^\w\- ]+", "", base, flags=re.U).strip().replace(" ", "_").lower()[:60]
    return (base or "notes") + ".md"


class Notebook:
    def __init__(self, char_key):
        self.dir = os.path.join(NOTES_DIR, char_key)

    def _path(self, name):
        return os.path.join(self.dir, note_filename(name))

    def names(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(f for f in os.listdir(self.dir) if f.lower().endswith(".md"))

    def list(self, _args=None):
        names = self.names()
        return "Your notes: " + ", ".join(names) if names else "You have no notes yet."

    def read(self, args):
        p = self._path(args.get("name"))
        if not os.path.isfile(p):
            return f"There is no note called {os.path.basename(p)}. " + self.list()
        with open(p, "r", encoding="utf-8") as f:
            return f"{os.path.basename(p)}:\n{f.read()}"

    def write(self, args):
        """mode: append (default), replace or delete."""
        mode = str(args.get("mode") or ("replace" if args.get("append") is False else "append")).lower()
        if mode.startswith("del"):
            return self.delete(args)
        p = self._path(args.get("name"))
        content = str(args.get("text") or args.get("content") or "").strip()
        if not content:
            return "Nothing to write."
        append = not mode.startswith("rep")
        if not os.path.isfile(p) and len(self.names()) >= MAX_NOTES:
            return f"You already have {MAX_NOTES} notes; add to or delete an existing one."
        old = ""
        if append and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                old = f.read().rstrip() + "\n"
        text = old + content + "\n"
        if len(text) > MAX_NOTE_CHARS:
            return (f"That would make {os.path.basename(p)} too long ({len(text)} of {MAX_NOTE_CHARS} "
                    "characters). Rewrite it shorter with mode=replace.")
        os.makedirs(self.dir, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return f"{'Added to' if old else 'Saved'} {os.path.basename(p)}."

    def delete(self, args):
        p = self._path(args.get("name"))
        if not os.path.isfile(p):
            return f"There is no note called {os.path.basename(p)}."
        os.remove(p)
        return f"Deleted {os.path.basename(p)}."


def current_time(_args=None):
    now = datetime.now().astimezone()
    off = now.strftime("%z")
    return now.strftime("It is %A, %d %B %Y, %H:%M") + f" (UTC{off[:3]}:{off[3:]})."
