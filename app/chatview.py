"""The local chat's message list: a read-only Text widget that looks roughly like a Discord channel."""
import datetime
import os
import tkinter as tk
import tkinter.font as tkfont
import webbrowser
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk

from .localchat import LOCAL_FILES, is_image
from .markdown import parse

TEXT, MUTED, NAME, CHAR_NAME, LINK = "#dbdee1", "#949ba4", "#f2f3f5", "#c9b8ff", "#00a8fc"
BG, CODE_BG, NOTICE = "#1e1f24", "#2b2d31", "#9aa0ad"
AVATAR, INDENT, THUMB = 40, 56, (360, 280)
COLORS = ["#5865f2", "#3ba55c", "#faa61a", "#ed4245", "#eb459e", "#7c5cff"]


def round_avatar(path, name, size=AVATAR):
    """A circular picture, or a coloured circle with the name's first letter."""
    try:
        im = Image.open(path).convert("RGBA")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size * 2, size * 2))
    except Exception:
        im = Image.new("RGBA", (size * 2, size * 2), COLORS[sum(map(ord, name or "?")) % len(COLORS)])
        d = ImageDraw.Draw(im)
        letter = (name or "?")[0].upper()
        try:
            from PIL import ImageFont
            f = ImageFont.truetype("segoeuib.ttf", size)
        except Exception:
            f = None
        d.text((size, size), letter, fill="white", anchor="mm", font=f)
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, *im.size), fill=255)
    im.putalpha(mask)
    return ImageTk.PhotoImage(im.resize((size, size), Image.LANCZOS))


class ChatView(tk.Frame):
    def __init__(self, master, avatar_for):
        """`avatar_for(name, is_char) -> path or None`."""
        super().__init__(master, bg=BG)
        self.avatar_for = avatar_for
        self.text = tk.Text(self, bg=BG, fg=TEXT, relief="flat", wrap="word", padx=16, pady=12, cursor="arrow",
                            insertwidth=0, highlightthickness=0, borderwidth=0, spacing1=1, spacing3=1)
        bar = ctk.CTkScrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.text.bind("<Key>", lambda e: None if (e.state & 4 and e.keysym.lower() in "ac") else "break")
        self.images, self.avatars, self.fonts = [], {}, {}
        self.n = 0                       # counter for one-off tags (links, spoilers, pictures)
        self.last = None                 # (author, ts) of the last message, to group like Discord
        t = self.text
        t.tag_configure("body", lmargin1=INDENT, lmargin2=INDENT, spacing3=2)
        t.tag_configure("head", font=self._font(size=11, bold=True), foreground=NAME, spacing1=10)
        t.tag_configure("charname", foreground=CHAR_NAME)
        t.tag_configure("time", font=self._font(size=8), foreground=MUTED)
        t.tag_configure("badge", font=self._font(size=8, bold=True), foreground="#8f9cff")
        t.tag_configure("notice", foreground=NOTICE, lmargin1=INDENT, lmargin2=INDENT, spacing1=8, spacing3=4,
                        font=self._font(size=10, italic=True))
        t.tag_configure("bar", foreground="#4e5058")
        t.tag_configure("u", underline=True)
        t.tag_configure("s", overstrike=True)
        t.tag_configure("code", background=CODE_BG)
        t.tag_configure("codeblock", background=CODE_BG, lmargin1=INDENT + 8, lmargin2=INDENT + 8, rmargin=24,
                        spacing1=4, spacing3=4)
        t.tag_configure("quote", lmargin1=INDENT, lmargin2=INDENT + 14, foreground="#c4c9d1")
        t.tag_configure("sub", foreground=MUTED)
        t.tag_configure("link", foreground=LINK, underline=True)
        t.tag_configure("file", background=CODE_BG, foreground=LINK, font=self._font(size=10, bold=True))
        for tag in ("code", "codeblock", "file"):
            t.tag_configure(tag, lmargincolor=BG)    # a background must not run into the left margin
        t.configure(state="disabled")

    def _font(self, size=10, bold=False, italic=False, mono=False):
        key = (size, bold, italic, mono)
        if key not in self.fonts:
            self.fonts[key] = tkfont.Font(family="Consolas" if mono else "Segoe UI", size=size,
                                          weight="bold" if bold else "normal", slant="italic" if italic else "roman")
        return self.fonts[key]

    def _font_tag(self, tags):
        size = 16 if "h1" in tags else 14 if "h2" in tags else 12 if "h3" in tags else 8 if "sub" in tags else 10
        bold = bool(tags & {"b", "h1", "h2", "h3"})
        mono = bool(tags & {"code", "codeblock"})
        name = f"font_{size}_{int(bold)}_{int('i' in tags)}_{int(mono)}"
        self.text.tag_configure(name, font=self._font(size + (-1 if mono else 0), bold, "i" in tags, mono))
        return name

    def _one_off(self):
        self.n += 1
        return f"x{self.n}"

    def _hand(self, tag):
        self.text.tag_bind(tag, "<Enter>", lambda e: self.text.configure(cursor="hand2"))
        self.text.tag_bind(tag, "<Leave>", lambda e: self.text.configure(cursor="arrow"))

    # ---- adding things ------------------------------------------------------
    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self.images, self.last = [], None

    def add(self, msg):
        at_bottom = self.text.yview()[1] > 0.98
        self.text.configure(state="normal")
        if msg["kind"] == "notice":
            self._markdown(msg["text"], ("notice",))
            self.text.insert("end", "\n", ("notice",))
            self.last = None
        else:
            self._message(msg)
        self.text.configure(state="disabled")
        if at_bottom:
            self.text.see("end")

    def _message(self, msg):
        t, author = self.text, msg["author"]
        grouped = self.last and self.last[0] == author and msg["ts"] - self.last[1] < 300
        self.last = (author, msg["ts"])
        if not grouped:
            if author not in self.avatars:
                self.avatars[author] = round_avatar(self.avatar_for(author, msg["char"]), author)
            if t.index("end-1c") != "1.0":
                t.insert("end", "\n", ("head",))
            t.image_create("end", image=self.avatars[author], padx=0, pady=0)
            t.insert("end", "  ")
            t.insert("end", author, ("head", "charname") if msg["char"] else ("head",))
            if msg["char"]:
                t.insert("end", "  AI", ("badge",))
            when = datetime.datetime.fromtimestamp(msg["ts"])
            fmt = "%H:%M" if when.date() == datetime.date.today() else "%d/%m/%Y %H:%M"
            t.insert("end", "   " + when.strftime(fmt), ("time",))
            t.insert("end", "\n")
        if msg["text"]:
            self._markdown(msg["text"], ("body",))
            if t.get("end-2c") != "\n":
                t.insert("end", "\n", ("body",))
        for fname in msg["files"]:
            path = os.path.join(LOCAL_FILES, fname)
            if is_image(fname) and os.path.isfile(path):
                self._picture(path)
            else:
                self._file(path, fname.split("_", 1)[-1])

    def _markdown(self, text, base):
        t = self.text
        for seg, tags, url in parse(text):
            names = list(base) + [self._font_tag(tags)] + [x for x in ("u", "s", "code", "codeblock", "quote",
                                                                          "bar", "sub", "link") if x in tags]
            if url:
                tag = self._one_off()
                t.tag_bind(tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
                self._hand(tag)
                names.append(tag)
            if "spoiler" in tags:
                tag = self._one_off()
                t.tag_configure(tag, background="#111214", foreground="#111214", lmargincolor=BG)
                t.tag_bind(tag, "<Button-1>", lambda e, g=tag: t.tag_configure(g, background=CODE_BG,
                                                                              foreground=TEXT))
                self._hand(tag)
                names.append(tag)
            t.insert("end", seg, tuple(names))

    def _picture(self, path):
        try:
            im = Image.open(path)
            im.thumbnail(THUMB)
            photo = ImageTk.PhotoImage(im.convert("RGBA"))
        except Exception:
            return self._file(path, os.path.basename(path))
        self.images.append(photo)
        tag = self._one_off()
        t = self.text
        start = t.index("end-1c")
        t.image_create("end", image=photo, pady=4)
        t.tag_add(tag, start, "end-1c")
        t.tag_add("body", start, "end-1c")
        t.insert("end", "\n", ("body",))
        t.tag_bind(tag, "<Button-1>", lambda e: ImagePreview(self, path))
        self._hand(tag)

    def _file(self, path, name):
        tag = self._one_off()
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        self.text.insert("end", f"  📄 {name}  ", ("body", "file", tag))
        self.text.insert("end", f"  {size / 1024:.1f} KB\n", ("body", "time"))
        self.text.tag_bind(tag, "<Button-1>", lambda e: FilePreview(self, path, name))
        self._hand(tag)


class ImagePreview(ctk.CTkToplevel):
    """A picture at full size (shrunk to fit the screen); click or Esc closes it."""

    def __init__(self, master, path):
        super().__init__(master, fg_color="#000000")
        self.title(os.path.basename(path).split("_", 1)[-1])
        im = Image.open(path)
        im.thumbnail((int(self.winfo_screenwidth() * 0.85), int(self.winfo_screenheight() * 0.85)))
        self.photo = ImageTk.PhotoImage(im.convert("RGBA"))
        lab = tk.Label(self, image=self.photo, bg="#000000", cursor="hand2")
        lab.pack()
        bar = ctk.CTkFrame(self, fg_color="#000000")
        bar.pack(fill="x")
        ctk.CTkButton(bar, text="Open original", width=120, fg_color="#2b2d31",
                      command=lambda: os.startfile(path)).pack(side="right", padx=8, pady=6)
        ctk.CTkButton(bar, text="Save as…", width=100, fg_color="#2b2d31",
                      command=lambda: _save_copy(path)).pack(side="right", pady=6)
        lab.bind("<Button-1>", lambda e: self.destroy())
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(50, self.focus_force)


class FilePreview(ctk.CTkToplevel):
    """A posted file's text, with buttons to save a copy or open it in its program."""

    def __init__(self, master, path, name):
        super().__init__(master, fg_color=BG)
        self.title(name)
        self.geometry("760x560")
        box = ctk.CTkTextbox(self, fg_color=CODE_BG, font=ctk.CTkFont(family="Consolas", size=12), wrap="none")
        box.pack(fill="both", expand=True, padx=12, pady=(12, 6))
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                box.insert("1.0", f.read(400_000))
        except OSError as e:
            box.insert("1.0", f"Couldn't open the file: {e}")
        box.configure(state="disabled")
        bar = ctk.CTkFrame(self, fg_color=BG)
        bar.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(bar, text="Open with…", width=110, fg_color="#2b2d31",
                      command=lambda: os.startfile(path)).pack(side="right")
        ctk.CTkButton(bar, text="Save as…", width=100, fg_color="#2b2d31",
                      command=lambda: _save_copy(path, name)).pack(side="right", padx=8)
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(50, self.focus_force)


def _save_copy(path, name=None):
    name = name or os.path.basename(path).split("_", 1)[-1]
    ext = os.path.splitext(name)[1]
    dest = filedialog.asksaveasfilename(initialfile=name, defaultextension=ext)
    if dest:
        with open(path, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())
