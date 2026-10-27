"""Characters are JSON files in ./characters:

    {
      "name": "Luna",
      "prompt": "You are Luna, ...",
      "avatar": "luna.png",         # path relative to the json, or a data:image/...;base64 URI
      "description": "A sleepy witch",  # short, shown in /characters and the summon list
      "voice": "21m00Tcm4TlvDq8ikWAM",  # ElevenLabs voice ID; without one the character can't join voice
      "local_voice": "af_bella",        # Kokoro voice, used when the Voice page is set to the local voice
      "forms": ["Luna (cursed)"]        # characters it can transform into; linked forms share one memory
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
    local_voice: str = ""   # Kokoro voice name
    forms: list = field(default_factory=list)   # names of the characters it can transform into

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
        if self.local_voice:
            d["local_voice"] = self.local_voice
        if self.forms:
            d["forms"] = list(self.forms)
        return d


OWN_FIELDS = ("name", "description", "prompt", "avatar", "voice", "local_voice", "forms")


def slug(text):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip()).strip("_").lower()
    return s or "character"


class CharacterStore:
    def __init__(self, directory=CHARACTERS_DIR):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)
        self.characters = {}
        self.on_reload = lambda groups: None   # the memory follows changes to linked forms
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
                forms = d.get("forms") or []
                c = Character(d["name"], d.get("prompt", ""), d.get("avatar", ""), path,
                              str(d.get("description") or ""), str(d.get("voice") or "").strip(), extra,
                              str(d.get("local_voice") or "").strip(),
                              [str(n).strip() for n in forms if str(n).strip()] if isinstance(forms, list) else [])
                chars[c.name.lower()] = c
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        self.characters = chars
        self.on_reload(self.groups())

    # ---- forms (transformations) --------------------------------------------
    def forms_of(self, char):
        """The existing characters `char` can transform into (links count both ways)."""
        names = {n.lower() for n in char.forms}
        names |= {c.name.lower() for c in self.characters.values() if char.name.lower() in
                  {n.lower() for n in c.forms}}
        names.discard(char.name.lower())
        return [self.characters[n] for n in sorted(names) if n in self.characters]

    def groups(self):
        """Lists of names that share one memory: every character reachable through forms."""
        seen, out = set(), []
        for c in self.characters.values():
            if c.name.lower() in seen:
                continue
            group, todo = [], [c]
            while todo:
                x = todo.pop()
                if x.name.lower() in seen:
                    continue
                seen.add(x.name.lower())
                group.append(x.name)
                todo += self.forms_of(x)
            if len(group) > 1:
                out.append(sorted(group, key=str.lower))
        return out

    def group_of(self, char):
        """`char` and every form that shares its memory."""
        for g in self.groups():
            if char.name.lower() in {n.lower() for n in g}:
                return [self.characters[n.lower()] for n in g]
        return [char]

    def link_forms(self, char, old_name=None, old_forms=()):
        """Keeps links two-way after `char` was saved: partners list it too, dropped partners don't,
        and a renamed character is renamed in its partners' lists."""
        now = {n.lower() for n in char.forms}
        before = {n.lower() for n in old_forms}
        names = {char.name.lower(), (old_name or char.name).lower()}
        for other in list(self.characters.values()):
            if other.name.lower() in names:
                continue
            mine = [n for n in other.forms if n.lower() not in names]
            if other.name.lower() in now:
                mine.append(char.name)
            elif other.name.lower() not in before and len(mine) == len(other.forms):
                continue          # never linked: leave its file alone
            if mine != other.forms:
                other.forms = mine
                with open(other.path, "w", encoding="utf-8") as f:
                    json.dump(other.to_json(), f, indent=2, ensure_ascii=False)
        self.reload()

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
