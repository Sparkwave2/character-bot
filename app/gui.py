"""Dark-mode control panel (CustomTkinter)."""
import io
import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageDraw

from .characters import Character, slug
from .config import CHARACTERS_DIR
from .cropper import CropDialog, load_image
from .draw import SDXL_MIN_BYTES, list_checkpoints
from .engine import resolve_backend
from .games import running_process_names
from .llm import VERBOSE_LOG
from .log import log, log_queue
from .tools import TOOL_LABELS, TOOL_NAMES
from .tts import KOKORO_VOICES, VOICE_KINDS, kokoro_label

ACCENT = "#7c5cff"
ACCENT_HOVER = "#6a4ae6"
BG = "#0f1117"
PANEL = "#171a23"
CARD = "#1e2230"
BORDER = "#2a2f40"
MUTED = "#8b91a7"
OK, WARN, BAD = "#3ddc97", "#f5b942", "#ff5c7a"

INVITE_PERMS = 1024 | 2048 | 64 | 65536 | 536870912 | 1048576 | 2097152  # view, send, react, history, webhooks, connect, speak


def resolve_backend_safe(choice, engine):
    return resolve_backend(choice, engine.hw) if engine.hw else None


def circle_image(src, size):
    """`src` is a path, file-like, PIL image or None (placeholder)."""
    try:
        im = src.convert("RGBA") if isinstance(src, Image.Image) else Image.open(src).convert("RGBA")
    except Exception:
        im = Image.new("RGBA", (size, size), "#2a2f40")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size * 2, size * 2))
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, *im.size), fill=255)
    im.putalpha(mask)
    return ctk.CTkImage(im, size=(size, size))


class Field:
    """A labelled config field bound to cfg[section][key]."""

    def __init__(self, parent, cfg, section, key, label, kind="str", hint="", filetypes=None, row=0):
        self.cfg, self.section, self.key, self.kind = cfg, section, key, kind
        val = cfg[section][key]
        ctk.CTkLabel(parent, text=label, anchor="w", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, sticky="w", padx=(16, 12), pady=(10, 0))
        if hint:
            ctk.CTkLabel(parent, text=hint, anchor="w", text_color=MUTED, font=ctk.CTkFont(size=11)).grid(
                row=row + 1, column=0, sticky="nw", padx=(16, 12))
        if kind == "bool":
            self.var = tk.BooleanVar(value=bool(val))
            w = ctk.CTkSwitch(parent, text="", variable=self.var, progress_color=ACCENT)
            w.grid(row=row, column=1, sticky="w", pady=(10, 0))
            return
        self.var = tk.StringVar(value=str(val))
        w = ctk.CTkEntry(parent, textvariable=self.var, height=34, border_color=BORDER,
                         fg_color=CARD, show="•" if kind == "secret" else "")
        w.grid(row=row, column=1, sticky="ew", pady=(10, 0), padx=(0, 8 if kind in ("file", "dir") else 16))
        if kind in ("file", "dir"):
            ctk.CTkButton(parent, text="Browse", width=80, height=34, fg_color=CARD, hover_color=BORDER,
                          border_width=1, border_color=BORDER,
                          command=lambda: self._browse(filetypes)).grid(row=row, column=2, pady=(10, 0), padx=(0, 16))

    def _browse(self, filetypes):
        p = filedialog.askdirectory() if self.kind == "dir" else \
            filedialog.askopenfilename(filetypes=filetypes or [("All files", "*.*")])
        if p:
            self.var.set(os.path.normpath(p))

    def commit(self):
        raw = self.var.get()
        try:
            if self.kind == "int":
                v = int(float(raw))
            elif self.kind == "float":
                v = float(raw)
            elif self.kind == "bool":
                v = bool(raw)
            else:
                v = raw.strip()
        except ValueError:
            raise ValueError(f"'{raw}' is not a valid value for {self.key}")
        self.cfg[self.section][self.key] = v


class App(ctk.CTk):
    def __init__(self, core):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        super().__init__(fg_color=BG)
        self.core = core
        self.cfg = core.cfg
        self.title("Character Bot")
        self.geometry("1180x760")
        self.minsize(980, 620)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_debug = tk.BooleanVar(value=False)
        self.channel_vars = {}
        self.selected_char = None
        self.avatar_choice = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self.content = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.pages = {}
        for name, builder in [("Dashboard", self.page_dashboard), ("Model", self.page_model),
                              ("Discord", self.page_discord), ("Characters", self.page_characters),
                              ("Voice", self.page_voice), ("Drawing", self.page_drawing),
                              ("Games", self.page_games),
                              ("Behavior", self.page_behavior),
                              ("Logs", self.page_logs)]:
            frame = ctk.CTkFrame(self.content, fg_color=BG, corner_radius=0)
            frame.grid(row=0, column=0, sticky="nsew")
            frame.grid_columnconfigure(0, weight=1)
            builder(frame)
            self.pages[name] = frame
        self.show("Dashboard")
        self.after(200, self._tick)
        self.after(600, self._autostart)

    # ---- layout helpers -----------------------------------------------------
    def _build_sidebar(self):
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, width=210)
        bar.grid(row=0, column=0, sticky="nsw")
        bar.grid_propagate(False)
        ctk.CTkLabel(bar, text="◆ Character Bot", font=ctk.CTkFont(size=18, weight="bold")).pack(
            anchor="w", padx=20, pady=(24, 28))
        self.nav = {}
        icons = {"Dashboard": "◉", "Model": "⚙", "Discord": "✉", "Characters": "☺",
                 "Voice": "♪", "Drawing": "✎", "Games": "▶", "Behavior": "≡", "Logs": "☰"}
        for name, icon in icons.items():
            b = ctk.CTkButton(bar, text=f"  {icon}   {name}", anchor="w", height=40, corner_radius=10,
                              fg_color="transparent", hover_color=CARD, text_color="#d6d9e4",
                              font=ctk.CTkFont(size=14), command=lambda n=name: self.show(n))
            b.pack(fill="x", padx=12, pady=3)
            self.nav[name] = b
        self.side_status = ctk.CTkLabel(bar, text="", justify="left", text_color=MUTED,
                                        font=ctk.CTkFont(size=12))
        self.side_status.pack(side="bottom", anchor="w", padx=20, pady=20)

    def show(self, name):
        self.pages[name].tkraise()
        for n, b in self.nav.items():
            b.configure(fg_color=ACCENT if n == name else "transparent",
                        hover_color=ACCENT_HOVER if n == name else CARD)
        if name == "Characters":
            self.refresh_char_list()

    def header(self, parent, title, subtitle=""):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=24, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=32, pady=(28, 0))
        if subtitle:
            ctk.CTkLabel(parent, text=subtitle, text_color=MUTED).grid(row=1, column=0, sticky="w", padx=32)

    def card(self, parent, row, title=None, **grid):
        c = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        c.grid(row=row, column=0, sticky="nsew", padx=32, pady=(18, 0), **grid)
        c.grid_columnconfigure(1, weight=1)
        if title:
            ctk.CTkLabel(c, text=title, font=ctk.CTkFont(size=15, weight="bold")).grid(
                row=0, column=0, columnspan=3, sticky="w", padx=16, pady=(14, 0))
        return c

    def button(self, parent, text, command, primary=True, **kw):
        return ctk.CTkButton(parent, text=text, command=command, height=36, corner_radius=10,
                             fg_color=ACCENT if primary else CARD,
                             hover_color=ACCENT_HOVER if primary else BORDER,
                             border_width=0 if primary else 1, border_color=BORDER, **kw)

    def scroll_page(self, frame):
        s = ctk.CTkScrollableFrame(frame, fg_color=BG, corner_radius=0)
        s.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        s.grid_columnconfigure(0, weight=1)
        return s

    def save_fields(self, fields):
        try:
            for f in fields:
                f.commit()
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e))
            return False
        self.cfg.save()
        self.core.memory.history_limit = int(self.cfg["behavior"]["history_limit"])
        log("[gui] settings saved")
        return True

    # ---- dashboard ----------------------------------------------------------
    def page_dashboard(self, f):
        self.header(f, "Dashboard", "Status at a glance")
        grid = ctk.CTkFrame(f, fg_color=BG)
        grid.grid(row=2, column=0, sticky="ew", padx=20, pady=(16, 0))
        for i in range(3):
            grid.grid_columnconfigure(i, weight=1, uniform="s")
        self.stat = {}
        for i, (key, title) in enumerate([("model", "Model"), ("discord", "Discord"), ("game", "Game watch")]):
            c = ctk.CTkFrame(grid, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
            c.grid(row=0, column=i, sticky="nsew", padx=12)
            ctk.CTkLabel(c, text=title, text_color=MUTED).pack(anchor="w", padx=18, pady=(16, 0))
            lab = ctk.CTkLabel(c, text="…", font=ctk.CTkFont(size=20, weight="bold"))
            lab.pack(anchor="w", padx=18)
            sub = ctk.CTkLabel(c, text="", text_color=MUTED, wraplength=280, justify="left")
            sub.pack(anchor="w", padx=18, pady=(0, 10))
            row = ctk.CTkFrame(c, fg_color="transparent")
            row.pack(anchor="w", padx=14, pady=(0, 16))
            self.stat[key] = (lab, sub, row)
        _, _, r = self.stat["model"]
        self.button(r, "Load", lambda: self.core.submit(self.core.load_model()), width=90).pack(side="left", padx=4)
        self.button(r, "Unload", lambda: self.core.submit(self.core.unload_model()), primary=False, width=90).pack(side="left", padx=4)
        _, _, r = self.stat["discord"]
        self.button(r, "Connect", lambda: self.core.submit(self.core.connect_discord()), width=90).pack(side="left", padx=4)
        self.button(r, "Disconnect", lambda: self.core.submit(self.core.disconnect_discord()), primary=False, width=90).pack(side="left", padx=4)
        _, _, r = self.stat["game"]
        self.button(r, "Manage list", lambda: self.show("Games"), primary=False, width=110).pack(side="left", padx=4)

        c = self.card(f, 3, "Summoned characters")
        self.summoned_box = ctk.CTkLabel(c, text="", justify="left", anchor="w", text_color="#d6d9e4")
        self.summoned_box.grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(6, 16))
        c = self.card(f, 4, "Context")
        self.ctx_box = ctk.CTkFrame(c, fg_color="transparent")
        self.ctx_box.grid(row=1, column=0, columnspan=3, sticky="ew", padx=16, pady=(6, 14))
        self.ctx_box.grid_columnconfigure(1, weight=1)
        self.ctx_rows, self.ctx_data, self.ctx_next = {}, None, 0
        self.ctx_empty = ctk.CTkLabel(self.ctx_box, text="", text_color=MUTED, anchor="w")
        self.ctx_empty.grid(row=0, column=0, columnspan=3, sticky="w")
        c = self.card(f, 5, "Quick start")
        ctk.CTkLabel(c, justify="left", anchor="w", text_color=MUTED, text=(
            "1. Model → pick your Qwen-VL GGUF and its mmproj file, then Load (llama.cpp installs itself).\n"
            "2. Discord → paste your bot token, Connect, tick the channels the bot may speak in.\n"
            "3. Characters → create or edit characters (name, prompt, profile picture).\n"
            "4. In Discord use /summon <character> and /dismiss <character>.")).grid(
            row=1, column=0, sticky="w", padx=16, pady=(6, 16))

    # ---- model --------------------------------------------------------------
    def page_model(self, f):
        s = self.scroll_page(f)
        self.header(s, "Model", "llama.cpp server running a Qwen vision GGUF")
        gguf = [("GGUF", "*.gguf"), ("All files", "*.*")]
        exe = [("llama-server", "llama-server*"), ("All files", "*.*")]
        self._build_engine_card(s, 2)
        c = self.card(s, 3, "Model files")
        self.model_fields = [
            Field(c, self.cfg, "llm", "model_path", "Model GGUF", "file", "e.g. Qwen2.5-VL-7B-Instruct-Q4_K_M", gguf, row=1),
            Field(c, self.cfg, "llm", "mmproj_path", "Vision projector", "file", "mmproj-*.gguf (needed for images)", gguf, row=3),
        ]
        c = self.card(s, 4, "Runtime")
        specs = [("ctx_size", "Context size", "int", "Tokens; history scrolls to fit"),
                 ("kv_cache", "Context memory type", "str", "q8_0 = half the VRAM of f16; q4_0 = a quarter"),
                 ("gpu_layers", "GPU layers", "int", "99 = everything on GPU"),
                 ("threads", "CPU threads", "int", "0 = auto"),
                 ("port", "Port", "int", ""),
                 ("flash_attn", "Flash attention", "bool", ""),
                 ("autoload", "Load on startup", "bool", ""),
                 ("extra_args", "Extra arguments", "str", "Passed to llama-server as-is")]
        self.model_fields += [Field(c, self.cfg, "llm", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        c = self.card(s, 5, "Sampling")
        specs = [("temperature", "Temperature", "float"), ("top_p", "Top P", "float"), ("min_p", "Min P", "float"),
                 ("repeat_penalty", "Repeat penalty", "float"), ("max_tokens", "Max reply tokens", "int"),
                 ("use_native_tools", "Native tool calling", "bool")]
        self.model_fields += [Field(c, self.cfg, "llm", k, l, t, row=1 + i * 2) for i, (k, l, t) in enumerate(specs)]
        c = self.card(s, 6, "Advanced")
        self.model_fields.append(Field(c, self.cfg, "llm", "server_path", "Custom llama-server", "file",
                                       "Optional. Leave empty to use the built-in engine", exe, row=1))
        self.model_fields.append(Field(c, self.cfg, "llm", "verbose_log", "Verbose model log", "bool",
                                       "Writes every prompt, word for word, and every reply to "
                                       "logs/llm-verbose.log", row=3))
        self.button(c, "Open log", self._open_verbose_log, primary=False, width=100).grid(
            row=3, column=2, padx=(0, 16), pady=(10, 0))
        ctk.CTkLabel(c, text="").grid(row=5, column=0, pady=2)
        bar = ctk.CTkFrame(s, fg_color=BG)
        bar.grid(row=7, column=0, sticky="w", padx=32, pady=20)
        self.button(bar, "Save", lambda: self.save_fields(self.model_fields), width=110).pack(side="left", padx=(0, 8))
        self.button(bar, "Save & reload model", self._reload_model, primary=False, width=170).pack(side="left")

    def _open_verbose_log(self):
        if not os.path.isfile(VERBOSE_LOG):
            messagebox.showinfo("Verbose model log", "Nothing is logged yet. Turn on Verbose model log, Save, "
                                                     "and let a character reply.")
            return
        os.startfile(VERBOSE_LOG)

    BACKEND_LABELS = {"auto": "Auto (recommended)", "cuda": "CUDA (NVIDIA)", "vulkan": "Vulkan (any GPU)",
                      "cpu": "CPU only", "rocm": "ROCm (AMD)"}

    def _build_engine_card(self, parent, row):
        c = self.card(parent, row, "Engine · llama.cpp")
        self.engine_status = ctk.CTkLabel(c, text="…", font=ctk.CTkFont(size=18, weight="bold"), anchor="w")
        self.engine_status.grid(row=1, column=0, columnspan=3, sticky="w", padx=16, pady=(8, 0))
        self.engine_hw = ctk.CTkLabel(c, text="Detecting hardware…", text_color=MUTED, anchor="w")
        self.engine_hw.grid(row=2, column=0, columnspan=3, sticky="w", padx=16)

        ctk.CTkLabel(c, text="Build", font=ctk.CTkFont(size=13)).grid(row=3, column=0, sticky="w", padx=16, pady=(12, 0))
        labels = list(self.BACKEND_LABELS.values())
        self.backend_var = tk.StringVar(value=self.BACKEND_LABELS.get(self.cfg["llm"]["backend"], labels[0]))
        ctk.CTkOptionMenu(c, values=labels, variable=self.backend_var, command=self._set_backend, width=220,
                          fg_color=CARD, button_color=BORDER, button_hover_color=ACCENT,
                          dropdown_fg_color=CARD).grid(row=3, column=1, sticky="w", pady=(12, 0))
        ctk.CTkLabel(c, text="Update on startup", font=ctk.CTkFont(size=13)).grid(row=4, column=0, sticky="w", padx=16, pady=(10, 0))
        self.autoupdate_var = tk.BooleanVar(value=self.cfg["llm"]["auto_update"])
        ctk.CTkSwitch(c, text="", variable=self.autoupdate_var, progress_color=ACCENT,
                      command=self._set_autoupdate).grid(row=4, column=1, sticky="w", pady=(10, 0))

        bar = ctk.CTkFrame(c, fg_color="transparent")
        bar.grid(row=5, column=0, columnspan=3, sticky="w", padx=12, pady=(14, 0))
        self.engine_install_btn = self.button(bar, "Install", lambda: self._engine_install(False), width=130)
        self.engine_install_btn.pack(side="left", padx=4)
        self.button(bar, "Check for updates", lambda: self.core.submit(self.core.check_engine_update()),
                    primary=False, width=150).pack(side="left", padx=4)
        self.button(bar, "Reinstall", lambda: self._engine_install(True), primary=False, width=100).pack(side="left", padx=4)
        self.engine_bar = ctk.CTkProgressBar(c, progress_color=ACCENT, fg_color=CARD, height=8)
        self.engine_bar.set(0)
        self.engine_bar.grid(row=6, column=0, columnspan=3, sticky="ew", padx=16, pady=(14, 0))
        self.engine_msg = ctk.CTkLabel(c, text="", text_color=MUTED, anchor="w")
        self.engine_msg.grid(row=7, column=0, columnspan=3, sticky="w", padx=16, pady=(2, 14))

    def _set_backend(self, label):
        key = next(k for k, v in self.BACKEND_LABELS.items() if v == label)
        self.cfg["llm"]["backend"] = key
        self.cfg.save()
        if self.core.engine.installed and key != "auto":
            log("[gui] build changed; click Install to switch")

    def _set_autoupdate(self):
        self.cfg["llm"]["auto_update"] = bool(self.autoupdate_var.get())
        self.cfg.save()

    def _engine_install(self, force):
        if self.core.engine.busy:
            return
        self.core.engine.progress = (0.0, "Starting…")
        self.core.submit(self.core.install_engine(force=force))

    def _refresh_engine(self):
        eng = self.core.engine
        self.engine_status.configure(text=eng.describe(),
                                     text_color=OK if eng.installed or self.cfg["llm"]["server_path"] else WARN)
        if eng.hw:
            hw = eng.hw
            gpu = f"{hw['gpu_name']} · driver CUDA {hw['cuda_version']}" if hw["nvidia"] else (
                "Vulkan GPU available" if hw["vulkan"] else "No GPU acceleration found")
            auto = resolve_backend("auto", hw)
            self.engine_hw.configure(text=f"Detected: {gpu}  ·  Auto picks {auto.upper()}  ·  {hw['os']}-{hw['arch']}")
        frac, msg = eng.progress
        if eng.update_available and not eng.busy and frac in (0.0, 1.0):
            msg = f"Update available: {eng.latest_tag}"
        self.engine_bar.set(frac)
        self.engine_msg.configure(text=msg)
        want = resolve_backend_safe(self.cfg["llm"]["backend"], eng)
        if eng.busy:
            text = "Installing…"
        elif not eng.installed:
            text = "Install"
        elif eng.update_available or (want and want != eng.manifest.get("backend")):
            text = "Update" if eng.update_available else "Switch build"
        else:
            text = "Up to date"
        self.engine_install_btn.configure(text=text, state="disabled" if eng.busy else "normal")

    def _reload_model(self):
        if self.save_fields(self.model_fields):
            async def go():
                await self.core.llm.stop()
                await self.core.load_model()
            self.core.submit(go())

    # ---- discord ------------------------------------------------------------
    def page_discord(self, f):
        s = self.scroll_page(f)
        self.header(s, "Discord", "Bot connection and the channels it may speak in")
        c = self.card(s, 2, "Bot")
        self.discord_fields = [
            Field(c, self.cfg, "discord", "token", "Bot token", "secret", "Developer Portal → Bot → Token", row=1),
            Field(c, self.cfg, "discord", "autoconnect", "Connect on startup", "bool", row=3),
        ]
        bar = ctk.CTkFrame(c, fg_color="transparent")
        bar.grid(row=5, column=0, columnspan=3, sticky="w", padx=12, pady=14)
        self.button(bar, "Save & connect", self._connect, width=140).pack(side="left", padx=4)
        self.button(bar, "Disconnect", lambda: self.core.submit(self.core.disconnect_discord()),
                    primary=False, width=110).pack(side="left", padx=4)
        self.button(bar, "Copy invite link", self._copy_invite, primary=False, width=140).pack(side="left", padx=4)
        ctk.CTkLabel(c, text="Enable the MESSAGE CONTENT intent for the bot. Direct messages are ignored.",
                     text_color=MUTED).grid(row=6, column=0, columnspan=3, sticky="w", padx=16, pady=(0, 14))

        c = self.card(s, 3, "Allowed channels")
        top = ctk.CTkFrame(c, fg_color="transparent")
        top.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 0))
        self.button(top, "Refresh", self.refresh_channels, primary=False, width=90).pack(side="left", padx=4)
        self.button(top, "Save selection", self.save_channels, width=130).pack(side="left", padx=4)
        self.channel_frame = ctk.CTkFrame(c, fg_color="transparent")
        self.channel_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=16, pady=(8, 16))
        ctk.CTkLabel(self.channel_frame, text="Connect the bot to list channels.", text_color=MUTED).pack(anchor="w")

    def _connect(self):
        if self.save_fields(self.discord_fields):
            fut = self.core.submit(self.core.connect_discord())
            fut.add_done_callback(lambda _: None)

    def _copy_invite(self):
        bot = self.core.bot
        if not bot or not bot.user:
            messagebox.showinfo("Invite link", "Connect the bot first.")
            return
        url = (f"https://discord.com/oauth2/authorize?client_id={bot.user.id}"
               f"&scope=bot%20applications.commands&permissions={INVITE_PERMS}")
        self.clipboard_clear()
        self.clipboard_append(url)
        log(f"[gui] invite link copied: {url}")

    def refresh_channels(self):
        bot = self.core.bot
        for w in self.channel_frame.winfo_children():
            w.destroy()
        if not bot or not bot.is_ready():
            ctk.CTkLabel(self.channel_frame, text="Connect the bot to list channels.", text_color=MUTED).pack(anchor="w")
            return
        allowed = {str(c) for c in self.cfg["discord"]["allowed_channels"]}
        self.channel_vars = {}
        groups = bot.list_channels()
        if not groups:
            ctk.CTkLabel(self.channel_frame, text="The bot isn't in any server yet — use the invite link.",
                         text_color=MUTED).pack(anchor="w")
        for guild, chans in groups:
            ctk.CTkLabel(self.channel_frame, text=guild, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 2))
            box = ctk.CTkFrame(self.channel_frame, fg_color="transparent")
            box.pack(anchor="w", fill="x")
            for i, (cid, name) in enumerate(chans):
                v = tk.BooleanVar(value=cid in allowed)
                self.channel_vars[cid] = v
                ctk.CTkCheckBox(box, text=f"# {name}", variable=v, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                border_color=BORDER).grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 28), pady=4)
        self._channels_loaded = True

    def save_channels(self):
        if not self.channel_vars:
            return
        known = set(self.channel_vars)
        keep = [c for c in self.cfg["discord"]["allowed_channels"] if str(c) not in known]
        self.cfg["discord"]["allowed_channels"] = keep + [c for c, v in self.channel_vars.items() if v.get()]
        self.cfg.save()
        log(f"[gui] {len(self.cfg['discord']['allowed_channels'])} channel(s) allowed")

    # ---- characters ---------------------------------------------------------
    def page_characters(self, f):
        self.header(f, "Characters", f"JSON files in {CHARACTERS_DIR}")
        f.grid_rowconfigure(2, weight=1)
        body = ctk.CTkFrame(f, fg_color=BG)
        body.grid(row=2, column=0, sticky="nsew", padx=32, pady=18)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER, width=250)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        left.grid_propagate(False)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.button(left, "+ New character", self.new_character).grid(row=0, column=0, sticky="ew", padx=12, pady=12)
        self.char_list = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.char_list.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 8))
        self.button(left, "Reload from disk", self._reload_chars, primary=False).grid(row=2, column=0, sticky="ew", padx=12, pady=12)

        ed = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        ed.grid(row=0, column=1, sticky="nsew")
        ed.grid_columnconfigure(1, weight=1)
        ed.grid_rowconfigure(4, weight=1)
        self.avatar_label = ctk.CTkLabel(ed, text="", image=circle_image(None, 96))
        self.avatar_label.grid(row=0, column=0, rowspan=2, padx=20, pady=20)
        self.char_name = tk.StringVar()
        ctk.CTkEntry(ed, textvariable=self.char_name, placeholder_text="Name", height=40, fg_color=CARD,
                     border_color=BORDER, font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=1, sticky="sew", padx=(0, 20), pady=(20, 4))
        ab = ctk.CTkFrame(ed, fg_color="transparent")
        ab.grid(row=1, column=1, sticky="nw", pady=(4, 0))
        self.button(ab, "Choose picture…", self.choose_avatar, primary=False, width=140).pack(side="left")
        self.button(ab, "Adjust crop", self.adjust_crop, primary=False, width=110).pack(side="left", padx=(8, 0))
        self.char_path_label = ctk.CTkLabel(ab, text="", text_color=MUTED)
        self.char_path_label.pack(side="left", padx=12)
        self.char_desc = tk.StringVar()
        ctk.CTkEntry(ed, textvariable=self.char_desc, height=34, fg_color=CARD, border_color=BORDER,
                     placeholder_text="Short description (shown in /characters)").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 10))
        vrow = ctk.CTkFrame(ed, fg_color="transparent")
        vrow.grid(row=3, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 10))
        vrow.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(vrow, text="Voice", text_color=MUTED).grid(row=0, column=0, padx=(0, 10))
        self.char_voice = tk.StringVar()
        ctk.CTkEntry(vrow, textvariable=self.char_voice, height=34, fg_color=CARD, border_color=BORDER,
                     placeholder_text="ElevenLabs voice ID or name").grid(
            row=0, column=1, sticky="ew")
        self.button(vrow, "Pick…", self.pick_voice, primary=False, width=80).grid(row=0, column=2, padx=(8, 0))
        ctk.CTkLabel(vrow, text="Local", text_color=MUTED).grid(row=0, column=3, padx=(16, 10))
        self.char_local_voice = tk.StringVar()
        ctk.CTkEntry(vrow, textvariable=self.char_local_voice, height=34, width=150, fg_color=CARD,
                     border_color=BORDER, placeholder_text="Kokoro voice").grid(row=0, column=4)
        self.button(vrow, "Pick…", self.pick_local_voice, primary=False, width=80).grid(row=0, column=5, padx=(8, 0))
        self.char_prompt = ctk.CTkTextbox(ed, fg_color=CARD, border_color=BORDER, border_width=1,
                                          font=ctk.CTkFont(size=13), wrap="word")
        self.char_prompt.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 12))
        bar = ctk.CTkFrame(ed, fg_color="transparent")
        bar.grid(row=5, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))
        self.button(bar, "Save", self.save_character, width=100).pack(side="left", padx=4)
        self.button(bar, "Clear memory", self.clear_memory, primary=False, width=120).pack(side="left", padx=4)
        ctk.CTkButton(bar, text="Delete", command=self.delete_character, width=90, height=36, corner_radius=10,
                      fg_color="#3a1f29", hover_color="#552433", text_color=BAD).pack(side="right", padx=4)
        self.new_character()

    def _reload_chars(self):
        self.core.characters.reload()
        self.refresh_char_list()

    def refresh_char_list(self):
        for w in self.char_list.winfo_children():
            w.destroy()
        for ch in self.core.characters.characters.values():
            img = circle_image(ch.avatar_path() or None, 28) if ch.avatar_path() else circle_image(None, 28)
            sel = self.selected_char and self.selected_char.path == ch.path
            ctk.CTkButton(self.char_list, text="  " + ch.name, image=img, anchor="w", height=42, corner_radius=10,
                          fg_color=CARD if sel else "transparent", hover_color=CARD, text_color="#e6e8ef",
                          command=lambda c=ch: self.select_character(c)).pack(fill="x", padx=6, pady=2)

    def select_character(self, ch):
        self.selected_char = ch
        self.avatar_choice = None
        self.char_name.set(ch.name)
        self.char_desc.set(ch.description)
        self.char_voice.set(ch.voice)
        self.char_local_voice.set(ch.local_voice)
        self.char_prompt.delete("1.0", "end")
        self.char_prompt.insert("1.0", ch.prompt)
        self.char_path_label.configure(text=os.path.basename(ch.path))
        self.avatar_label.configure(image=circle_image(ch.avatar_path(), 96) if ch.avatar_path() else circle_image(None, 96))
        self.refresh_char_list()

    def new_character(self):
        self.selected_char = None
        self.avatar_choice = None
        self.char_name.set("")
        self.char_desc.set("")
        self.char_voice.set("")
        self.char_local_voice.set("")
        self.char_prompt.delete("1.0", "end")
        self.char_prompt.insert("1.0", "You are ... Describe personality, speaking style, backstory and boundaries here.")
        self.char_path_label.configure(text="(new)")
        self.avatar_label.configure(image=circle_image(None, 96))
        if hasattr(self, "char_list"):
            self.refresh_char_list()

    def choose_avatar(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.gif *.bmp"), ("All files", "*.*")])
        if not p:
            return
        try:
            im = load_image(p)
        except Exception as e:
            messagebox.showerror("Profile picture", f"Couldn't open that image:\n{e}")
            return
        self._open_cropper(im)

    def adjust_crop(self):
        """Re-crop the current (unsaved or saved) picture."""
        im = self.avatar_choice
        if im is None and self.selected_char:
            data = self.selected_char.avatar_bytes()
            if data:
                im = load_image(io.BytesIO(data))
        if im is None:
            messagebox.showinfo("Profile picture", "Choose a picture first.")
            return
        self._open_cropper(im)

    def _open_cropper(self, im):
        colors = {"bg": BG, "panel": PANEL, "card": CARD, "border": BORDER, "muted": MUTED,
                  "accent": ACCENT, "accent_hover": ACCENT_HOVER}
        CropDialog(self, im, self._set_avatar, colors)

    def _set_avatar(self, im):
        self.avatar_choice = im  # cropped PIL image, written on Save
        self.avatar_label.configure(image=circle_image(im, 96))
        self.char_path_label.configure(text="picture changed, not saved yet")

    def save_character(self):
        name = self.char_name.get().strip()
        if not name:
            messagebox.showerror("Character", "Name is required.")
            return
        old = self.selected_char
        existing = self.core.characters.get(name)
        if existing and (not old or existing.path != old.path):
            messagebox.showerror("Character", f"A character named {name} already exists.")
            return
        ch = Character(name, self.char_prompt.get("1.0", "end").strip(), old.avatar if old else "", "",
                       self.char_desc.get().strip(), self.char_voice.get().strip(),
                       dict(old.extra) if old else {}, self.char_local_voice.get().strip())
        ch.path = os.path.join(CHARACTERS_DIR, slug(name) + ".json")
        if self.avatar_choice is not None:
            dest = os.path.join(CHARACTERS_DIR, slug(name) + ".png")
            self.avatar_choice.save(dest, "PNG")
            ch.avatar = os.path.basename(dest)
        elif old and old.avatar and old.avatar_path() and not old.avatar.startswith("data:"):
            ch.avatar = old.avatar if os.path.isabs(old.avatar) else old.avatar
        self.core.characters.save(ch, old_path=old.path if old else None)
        self.select_character(self.core.characters.get(name))
        log(f"[gui] saved character {name}")

    def pick_voice(self):
        if not self.core.tts.key:
            messagebox.showinfo("Voice", "Set your ElevenLabs API key on the Voice page first.")
            return

        async def fetch():
            return await self.core.tts.list_voices(await self.core.llm.session())
        try:
            voices = self.core.submit(fetch()).result(timeout=25)
        except Exception as e:
            messagebox.showerror("Voice", f"Couldn't load your ElevenLabs voices:\n{e}")
            return
        if not voices:
            messagebox.showinfo("Voice", "Your ElevenLabs account has no voices.")
            return
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Pick a voice")
        win.geometry("360x480")
        win.transient(self)
        win.after(100, win.grab_set)
        lst = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        lst.pack(fill="both", expand=True, padx=12, pady=12)

        def choose(vid):
            self.char_voice.set(vid)
            win.destroy()
        order = list(VOICE_KINDS)
        for name, vid, kind in sorted(voices, key=lambda v: (order.index(v[2]), v[0].lower())):
            label, free = VOICE_KINDS[kind]
            ctk.CTkButton(lst, text=f"{name}   ·   {label}", anchor="w", height=34, fg_color="transparent",
                          hover_color=CARD, text_color="#e6e8ef" if free else MUTED,
                          command=lambda v=vid: choose(v)).pack(fill="x", padx=4, pady=1)

    def pick_local_voice(self):
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Pick a local voice")
        win.geometry("360x480")
        win.transient(self)
        win.after(100, win.grab_set)
        lst = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        lst.pack(fill="both", expand=True, padx=12, pady=12)

        def choose(v):
            self.char_local_voice.set(v)
            win.destroy()
        for v in KOKORO_VOICES:
            ctk.CTkButton(lst, text=kokoro_label(v), anchor="w", height=34, fg_color="transparent",
                          hover_color=CARD, text_color="#e6e8ef", command=lambda v=v: choose(v)).pack(
                fill="x", padx=4, pady=1)

    def delete_character(self):
        ch = self.selected_char
        if ch and messagebox.askyesno("Delete", f"Delete {ch.name}? This removes its JSON file."):
            self.core.characters.delete(ch)
            self.new_character()

    def clear_memory(self):
        ch = self.selected_char
        if ch and messagebox.askyesno("Clear memory", f"Forget everything {ch.name} has seen?"):
            self.core.loop.call_soon_threadsafe(self.core.memory.forget, ch.name)
            log(f"[gui] cleared memory of {ch.name}")

    # ---- games --------------------------------------------------------------
    def page_games(self, f):
        s = self.scroll_page(f)
        self.header(s, "Games", "Text generation is unloaded while any of these are running")
        c = self.card(s, 2, "Watcher")
        self.game_fields = [Field(c, self.cfg, "games", "enabled", "Enabled", "bool", row=1),
                            Field(c, self.cfg, "games", "poll_seconds", "Check every (s)", "float", row=3)]
        self.button(c, "Save", lambda: self.save_fields(self.game_fields), width=90).grid(
            row=5, column=0, sticky="w", padx=16, pady=14)
        c = self.card(s, 3, "Game processes")
        add = ctk.CTkFrame(c, fg_color="transparent")
        add.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(10, 0))
        self.game_entry = ctk.CTkEntry(add, placeholder_text="process name, e.g. eldenring.exe", width=320,
                                       height=36, fg_color=CARD, border_color=BORDER)
        self.game_entry.pack(side="left", padx=4)
        self.button(add, "Add", lambda: self.add_game(self.game_entry.get()), width=70).pack(side="left", padx=4)
        self.button(add, "Pick from running…", self.pick_process, primary=False, width=160).pack(side="left", padx=4)
        self.button(add, "Browse .exe…", self.browse_exe, primary=False, width=120).pack(side="left", padx=4)
        self.game_list = ctk.CTkFrame(c, fg_color="transparent")
        self.game_list.grid(row=2, column=0, columnspan=3, sticky="ew", padx=16, pady=(10, 16))
        self.refresh_games()

    def refresh_games(self):
        for w in self.game_list.winfo_children():
            w.destroy()
        procs = self.cfg["games"]["processes"]
        if not procs:
            ctk.CTkLabel(self.game_list, text="No games added yet.", text_color=MUTED).pack(anchor="w")
        for p in procs:
            row = ctk.CTkFrame(self.game_list, fg_color=CARD, corner_radius=10)
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=p, font=ctk.CTkFont(size=13)).pack(side="left", padx=14, pady=8)
            ctk.CTkButton(row, text="Remove", width=80, height=28, fg_color="transparent", hover_color="#3a1f29",
                          text_color=BAD, command=lambda x=p: self.remove_game(x)).pack(side="right", padx=8)

    def add_game(self, name):
        name = os.path.basename(name.strip())
        if name and name.lower() not in {p.lower() for p in self.cfg["games"]["processes"]}:
            self.cfg["games"]["processes"].append(name)
            self.cfg.save()
            self.refresh_games()
        self.game_entry.delete(0, "end")

    def remove_game(self, name):
        self.cfg["games"]["processes"] = [p for p in self.cfg["games"]["processes"] if p != name]
        self.cfg.save()
        self.refresh_games()

    def browse_exe(self):
        p = filedialog.askopenfilename(filetypes=[("Executables", "*.exe"), ("All files", "*.*")])
        if p:
            self.add_game(p)

    def pick_process(self):
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Running processes")
        win.geometry("420x560")
        win.transient(self)
        win.after(100, win.grab_set)
        q = tk.StringVar()
        ctk.CTkEntry(win, textvariable=q, placeholder_text="Filter…", height=36, fg_color=CARD,
                     border_color=BORDER).pack(fill="x", padx=14, pady=14)
        lst = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        lst.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        names = sorted(running_process_names(), key=str.lower)

        def render(*_):
            for w in lst.winfo_children():
                w.destroy()
            for n in [n for n in names if q.get().lower() in n.lower()][:200]:
                ctk.CTkButton(lst, text=n, anchor="w", fg_color="transparent", hover_color=CARD, height=30,
                              command=lambda x=n: (self.add_game(x), win.destroy())).pack(fill="x")
        q.trace_add("write", render)
        render()

    # ---- behavior -----------------------------------------------------------
    def page_behavior(self, f):
        s = self.scroll_page(f)
        self.header(s, "Behavior", "Turn-taking, context and vision")
        c = self.card(s, 2, "Conversation")
        specs = [("response_delay", "Reply delay (s)", "float", "Waits for follow-up messages before replying"),
                 ("max_bot_chain", "Max character chain", "int", "Consecutive character-to-character replies"),
                 ("bot_reply_chance", "Character reply chance", "float", "0–1, when one character speaks unprompted"),
                 ("backfill_messages", "History on summon", "int", "Recent messages read when summoned"),
                 ("history_limit", "Stored messages per character", "int", "Hard cap; context trimming happens anyway"),
                 ("typing_indicator", "Show typing indicator", "bool", ""),
                 ("decide_commands", "Ask about commands before each reply", "bool",
                  "One extra model call that makes characters actually use their commands")]
        self.behavior_fields = [Field(c, self.cfg, "behavior", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        c = self.card(s, 3, "Images, audio and links")
        specs = [("max_images_in_context", "Images kept in context", "int", "Older images become [image]"),
                 ("image_token_cost", "Tokens per image (estimate)", "int", "Used for context trimming"),
                 ("max_image_side", "Max image size (px)", "int", "Images are downscaled"),
                 ("fetch_image_links", "Read pasted image links", "bool", ""),
                 ("voice_messages", "Characters hear voice messages", "bool", "Transcribed locally with Whisper"),
                 ("link_previews", "Characters read posted links", "bool", "The model describes linked pages"),
                 ("check_links", "Check links before characters post them", "bool",
                  "Dead links are refused; image links are shown to the character first")]
        self.behavior_fields += [Field(c, self.cfg, "behavior", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        c = self.card(s, 4, "Commands characters can use")
        for i, name in enumerate(TOOL_NAMES):
            self.behavior_fields.append(Field(c, self.cfg, "tools", name, TOOL_LABELS[name][0], "bool",
                                              f"{name}", row=1 + i * 2))
        self.behavior_fields.append(Field(c, self.cfg, "gifs", "klipy_key", "KLIPY API key", "secret",
                                          "Free at partner.klipy.com; needed for post_gif",
                                          row=1 + len(TOOL_NAMES) * 2))
        ctk.CTkLabel(c, text="").grid(row=3 + len(TOOL_NAMES) * 2, column=0, pady=4)
        self.button(s, "Save", lambda: self.save_fields(self.behavior_fields), width=110).grid(
            row=5, column=0, sticky="w", padx=32, pady=20)

    # ---- voice --------------------------------------------------------------
    def page_voice(self, f):
        s = self.scroll_page(f)
        self.header(s, "Voice", "Characters with a voice can join voice chat with /join")
        c = self.card(s, 2, "Speaking (ElevenLabs)")
        specs = [("elevenlabs_key", "ElevenLabs API key", "secret", "elevenlabs.io → Developers → API keys"),
                 ("tts_model", "TTS model", "str", "eleven_flash_v2_5 starts in ~0.1 s; eleven_v3/v4 sound best "
                                                   "but take ~0.5 s or more")]
        self.voice_fields = [Field(c, self.cfg, "voice", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        c = self.card(s, 3, "Local voice (Kokoro, runs on this PC)")
        specs = [("local_tts", "Use the local voice instead of ElevenLabs", "bool",
                  "Free and fast (~0.2 s to start on the GPU). Downloads ~350 MB on first use"),
                 ("local_voice", "Default voice", "str", "For characters without their own local voice, e.g. af_heart"),
                 ("local_speed", "Speed", "float", "1.0 = normal"),
                 ("local_tts_device", "Device", "str", "auto (GPU if it works) or cpu"),
                 ("local_tts_dir", "Model folder", "dir", "Empty = models/kokoro in the app folder")]
        self.voice_fields += [Field(c, self.cfg, "voice", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        c = self.card(s, 4, "Listening (Whisper, runs locally)")
        specs = [("stt_model", "Whisper model", "str", "tiny, base, small, medium, large-v3 or turbo"),
                 ("stt_device", "Device", "str", "auto, cuda or cpu"),
                 ("stt_language", "Language", "str", "e.g. en; empty = detect automatically"),
                 ("stt_model_dir", "Model folder", "str", "Empty = models/whisper in the app folder"),
                 ("silence_seconds", "Pause that ends speech (s)", "float",
                  "How long someone is quiet before the character answers"),
                 ("interruptible", "Characters stop when talked over", "bool",
                  "They drop the rest of what they were saying and answer the new message instead")]
        self.voice_fields += [Field(c, self.cfg, "voice", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        self.button(s, "Save", lambda: self.save_fields(self.voice_fields), width=110).grid(
            row=5, column=0, sticky="w", padx=32, pady=20)

    # ---- drawing ------------------------------------------------------------
    def page_drawing(self, f):
        s = self.scroll_page(f)
        self.header(s, "Drawing", "Characters draw with Stable Diffusion (SDXL) through the A1111 or Forge WebUI")
        c = self.card(s, 2, "WebUI")
        specs = [("webui_dir", "WebUI folder", "dir", "The folder with webui.bat and launch.py"),
                 ("url", "API address", "str", "The app starts the WebUI here (API only, no window)"),
                 ("auto_start", "Start it when needed", "bool", "Off = only use a WebUI you started with --api"),
                 ("launch_args", "Extra launch options", "str", "--medvram-sdxl keeps SDXL small on A1111"),
                 ("unload_minutes", "Free GPU memory after (min)", "float", "Unloads the model when idle; 0 = never")]
        self.draw_fields = [Field(c, self.cfg, "draw", k, l, t, h, row=1 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        c = self.card(s, 3, "Picture")
        self.draw_model = tk.StringVar(value=self.cfg["draw"]["model"])
        ctk.CTkLabel(c, text="Model", anchor="w", font=ctk.CTkFont(size=13)).grid(
            row=1, column=0, sticky="w", padx=(16, 12), pady=(10, 0))
        ctk.CTkLabel(c, text="A checkpoint in models/Stable-diffusion", anchor="w", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky="nw", padx=(16, 12))
        ctk.CTkEntry(c, textvariable=self.draw_model, height=34, border_color=BORDER, fg_color=CARD).grid(
            row=1, column=1, sticky="ew", pady=(10, 0), padx=(0, 8))
        ctk.CTkButton(c, text="Pick…", width=80, height=34, fg_color=CARD, hover_color=BORDER, border_width=1,
                      border_color=BORDER, command=self.pick_checkpoint).grid(row=1, column=2, pady=(10, 0), padx=(0, 16))
        specs = [("prompt_prefix", "Prompt prefix", "str", "Added before every prompt, e.g. score_9, score_8_up, for Pony"),
                 ("negative_prompt", "Negative prompt", "str", ""),
                 ("width", "Width", "int", "SDXL works best around 1024×1024"),
                 ("height", "Height", "int", ""),
                 ("steps", "Steps", "int", ""),
                 ("cfg_scale", "CFG scale", "float", ""),
                 ("sampler", "Sampler", "str", "e.g. DPM++ 2M, Euler a"),
                 ("scheduler", "Scheduler", "str", "e.g. Karras, Automatic")]
        self.draw_fields += [Field(c, self.cfg, "draw", k, l, t, h, row=3 + i * 2) for i, (k, l, t, h) in enumerate(specs)]
        ctk.CTkLabel(c, text="Turn on “Draw pictures” on the Behavior page to give characters the command.",
                     text_color=MUTED).grid(row=3 + len(specs) * 2, column=0, columnspan=3, sticky="w", padx=16, pady=12)
        row = ctk.CTkFrame(s, fg_color="transparent")
        row.grid(row=4, column=0, sticky="w", padx=32, pady=20)
        self.button(row, "Save", self.save_drawing, width=110).pack(side="left")
        self.button(row, "Free GPU memory", lambda: self.core.submit(self._draw_unload()), primary=False,
                    width=150).pack(side="left", padx=8)
        self.button(row, "Stop WebUI", lambda: self.core.submit(self.core.draw.stop()), primary=False,
                    width=120).pack(side="left")

    def _draw_unload(self):
        async def go():
            await self.core.draw.unload(await self.core.llm.session())
        return go()

    def save_drawing(self):
        self.cfg["draw"]["model"] = self.draw_model.get().strip()
        self.save_fields(self.draw_fields)

    def pick_checkpoint(self):
        folder = self.draw_fields[0].var.get().strip()
        models = list_checkpoints(folder)
        if not models:
            messagebox.showinfo("Drawing", "No checkpoints found. Set the WebUI folder first; models are read "
                                           "from its models/Stable-diffusion folder.")
            return
        win = ctk.CTkToplevel(self, fg_color=BG)
        win.title("Pick a model")
        win.geometry("520x520")
        win.transient(self)
        win.after(100, win.grab_set)
        lst = ctk.CTkScrollableFrame(win, fg_color=PANEL)
        lst.pack(fill="both", expand=True, padx=12, pady=12)

        def choose(name):
            self.draw_model.set(name)
            win.destroy()
        for name, size in models:
            xl = size >= SDXL_MIN_BYTES
            ctk.CTkButton(lst, text=f"{name}   ·   {size / 1e9:.1f} GB" + ("   ·   SDXL" if xl else ""), anchor="w",
                          height=34, fg_color="transparent", hover_color=CARD,
                          text_color="#e6e8ef" if xl else MUTED, command=lambda n=name: choose(n)).pack(
                fill="x", padx=4, pady=1)

    # ---- logs ---------------------------------------------------------------
    def page_logs(self, f):
        self.header(f, "Logs")
        f.grid_rowconfigure(2, weight=1)
        self.log_box = ctk.CTkTextbox(f, fg_color=PANEL, border_color=BORDER, border_width=1,
                                      font=ctk.CTkFont(family="Consolas", size=12), wrap="word")
        self.log_box.grid(row=2, column=0, sticky="nsew", padx=32, pady=(14, 8))
        self.log_box.configure(state="disabled")
        bar = ctk.CTkFrame(f, fg_color=BG)
        bar.grid(row=3, column=0, sticky="w", padx=32, pady=(0, 18))
        ctk.CTkSwitch(bar, text="Show llama-server output", variable=self.show_debug,
                      progress_color=ACCENT).pack(side="left")
        self.button(bar, "Clear", self._clear_logs, primary=False, width=80).pack(side="left", padx=12)

    def _clear_logs(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ---- periodic refresh ---------------------------------------------------
    def _tick(self):
        lines = []
        try:
            while True:
                level, line = log_queue.get_nowait()
                if level != "debug" or self.show_debug.get():
                    lines.append(line)
        except queue.Empty:
            pass
        if lines:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", "\n".join(lines) + "\n")
            if float(self.log_box.index("end")) > 3000:
                self.log_box.delete("1.0", "500.0")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self._refresh_status()
        self._refresh_engine()
        self.after(400, self._tick)

    def _refresh_status(self):
        llm, game = self.core.llm, self.core.games.active_game
        lab, sub, _ = self.stat["model"]
        if llm.suspended_by:
            lab.configure(text="Suspended", text_color=WARN)
            sub.configure(text=f"Unloaded while {llm.suspended_by} runs")
        else:
            color = {"ready": OK, "starting": WARN, "stopping": WARN}.get(llm.state, MUTED)
            lab.configure(text=llm.state.capitalize(), text_color=color)
            sub.configure(text=os.path.basename(self.cfg["llm"]["model_path"]) or "No model selected")
        lab, sub, _ = self.stat["discord"]
        ds = self.core.discord_status
        lab.configure(text=ds.capitalize(), text_color={"connected": OK, "connecting": WARN}.get(ds, MUTED))
        bot = self.core.bot
        sub.configure(text=f"{bot.user} · {len(bot.guilds)} server(s)" if bot and bot.user and ds == "connected"
                      else f"{len(self.cfg['discord']['allowed_channels'])} channel(s) allowed")
        lab, sub, _ = self.stat["game"]
        lab.configure(text=game or ("Watching" if self.cfg["games"]["enabled"] else "Off"),
                      text_color=WARN if game else (OK if self.cfg["games"]["enabled"] else MUTED))
        sub.configure(text=f"{len(self.cfg['games']['processes'])} game(s) in list")
        if ds == "connected" and bot:
            ov = bot.summoned_overview()
            self.summoned_box.configure(text="\n".join(f"{ch}:  {', '.join(n)}" for ch, n in ov) or "Nobody summoned yet.")
            if not getattr(self, "_channels_loaded", False):
                self.refresh_channels()
        else:
            self.summoned_box.configure(text="Connect to Discord to see summoned characters.")
            self._channels_loaded = False
        self.side_status.configure(text=f"Model: {'suspended' if llm.suspended_by else llm.state}\nDiscord: {ds}")
        self._refresh_context(bot if ds == "connected" else None)

    def _refresh_context(self, bot):
        """Context meter: one bar per summoned character (computed on the core loop every 3 s)."""
        now = time.monotonic()
        if bot and now >= self.ctx_next:
            self.ctx_next = now + 3

            async def report():
                return bot.context_report()
            fut = self.core.submit(report())
            fut.add_done_callback(lambda f: setattr(self, "ctx_data", f.result() if not f.exception() else []))
        rows = self.ctx_data if bot else None
        names = [(r[0], r[1]) for r in rows or []]
        if names != list(self.ctx_rows):
            for widgets in self.ctx_rows.values():
                for w in widgets:
                    w.destroy()
            self.ctx_rows = {}
            for i, key in enumerate(names):
                lab = ctk.CTkLabel(self.ctx_box, text=f"{key[0]}  ·  #{key[1]}", anchor="w", width=200)
                lab.grid(row=i, column=0, sticky="w", pady=3)
                bar = ctk.CTkProgressBar(self.ctx_box, height=10, progress_color=ACCENT)
                bar.grid(row=i, column=1, sticky="ew", padx=12, pady=3)
                num = ctk.CTkLabel(self.ctx_box, text="", text_color=MUTED, anchor="e", width=260)
                num.grid(row=i, column=2, sticky="e", pady=3)
                self.ctx_rows[key] = (lab, bar, num)
        if not names:
            self.ctx_empty.configure(text="Connect to Discord and summon a character to see its context."
                                     if not bot else "Nobody summoned yet." if rows is not None else "…")
            self.ctx_empty.grid()
            return
        self.ctx_empty.grid_remove()
        for name, ch, est, last, size in rows:
            _lab, bar, num = self.ctx_rows[(name, ch)]
            frac = min(1.0, est / max(size, 1))
            bar.set(frac)
            bar.configure(progress_color=BAD if frac > 0.9 else WARN if frac > 0.75 else ACCENT)
            num.configure(text=f"≈{est:,} / {size:,} tokens" + (f"  ·  last reply {last:,}" if last else ""))

    def _autostart(self):
        self.core.submit(self.core.startup_engine())
        if self.cfg["llm"]["autoload"]:
            self.core.submit(self.core.load_model())
        if self.cfg["discord"]["autoconnect"] and self.cfg["discord"]["token"]:
            self.core.submit(self.core.connect_discord())

    def on_close(self):
        try:
            self.core.submit(self.core.shutdown()).result(timeout=15)
        except Exception:
            pass
        self.destroy()
