"""Characters are JSON files in ./characters:

    {
      "name": "Luna",
      "prompt": "You are Luna, ...",
      "avatar": "luna.png",         # path relative to the json, or a data:image/...;base64 URI
      "description": "A sleepy witch",  # short, shown in /characters and the summon list
      "voice": "21m00Tcm4TlvDq8ikWAM"   # ElevenLabs voice ID; without one the character can't join voice
    }
"""
import base64
import json
import os
import re
from dataclasses import dataclass, field

from .config import CHARACTERS_DIR


@dataclass
class Character:
    name: str
    prompt: str
    avatar: str = ""
    path: str = ""
    description: str = ""
    voice: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def key(self):
        return os.path.splitext(os.path.basename(self.path))[0] if self.path else slug(self.name)

    def avatar_bytes(self):
        """Raw image bytes of the profile pic, or None."""
        if not self.avatar:
            return None
        if self.avatar.startswith("data:"):
            try:
                return base64.b64decode(self.avatar.split(",", 1)[1])
            except (IndexError, ValueError):
                return None
        p = self.avatar_path()
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
        return None

    def avatar_path(self):
        if not self.avatar or self.avatar.startswith("data:"):
            return None
        if os.path.isabs(self.avatar):
            return self.avatar
        return os.path.join(os.path.dirname(self.path) or CHARACTERS_DIR, self.avatar)

    def to_json(self):
        d = dict(self.extra)
        d.update({"name": self.name, "description": self.description, "prompt": self.prompt,
                  "avatar": self.avatar, "voice": self.voice})
        return d


OWN_FIELDS = ("name", "description", "prompt", "avatar", "voice")


def slug(text):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_").lower()
    return s or "character"


class CharacterStore:
    def __init__(self, directory=CHARACTERS_DIR):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        self.characters = {}
        self.reload()

    def reload(self):
        chars = {}
        for fn in sorted(os.listdir(self.dir)):
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(self.dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                extra = {k: v for k, v in d.items() if k not in OWN_FIELDS}
                c = Character(d["name"], d.get("prompt", ""), d.get("avatar", ""), path,
                              str(d.get("description") or ""), str(d.get("voice") or "").strip(), extra)
                chars[c.name.lower()] = c
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        self.characters = chars

    def get(self, name):
        return self.characters.get((name or "").lower())

    def names(self):
        return [c.name for c in self.characters.values()]

    def save(self, char: Character, old_path=None):
        if not char.path:
            char.path = os.path.join(self.dir, slug(char.name) + ".json")
        with open(char.path, "w", encoding="utf-8") as f:
            json.dump(char.to_json(), f, indent=2, ensure_ascii=False)
        if old_path and os.path.abspath(old_path) != os.path.abspath(char.path) and os.path.exists(old_path):
            os.remove(old_path)
        self.reload()

    def delete(self, char: Character):
        if char.path and os.path.exists(char.path):
            os.remove(char.path)
        self.reload()
