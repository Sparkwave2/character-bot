"""Persistent settings stored in config.json next to the program."""
import json
import os
import threading
from copy import deepcopy

from .tools import TOOL_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.json")
STATE_PATH = os.path.join(ROOT, "state.json")
CHARACTERS_DIR = os.path.join(ROOT, "characters")

DEFAULTS = {
    "llm": {
        "backend": "auto",          # managed llama.cpp build: auto | cuda | vulkan | cpu | rocm
        "auto_update": True,        # check GitHub for a newer llama.cpp build on startup
        "server_path": "",          # optional custom llama-server; empty = managed install
        "model_path": "",           # Qwen-VL GGUF
        "mmproj_path": "",          # matching mmproj GGUF (vision projector)
        "host": "127.0.0.1",
        "port": 8089,
        "ctx_size": 32768,
        "kv_cache": "q8_0",         # f16 | q8_0 | q4_0: q8_0 halves the context's VRAM at no visible quality cost
        "gpu_layers": 99,
        "threads": 0,               # 0 = llama.cpp default
        "flash_attn": True,         # needed for a quantized KV cache
        "extra_args": "",
        "autoload": False,
        "temperature": 0.8,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "max_tokens": 400,
        "use_native_tools": True,   # send OpenAI-style tools (needs --jinja)
        "verbose_log": False,       # write every prompt (word for word) and reply to logs/llm-verbose.log
        "models_dir": "",           # where the model picker downloads to; empty = ./models
    },
    "discord": {
        "token": "",
        "autoconnect": False,
        "allowed_channels": [],     # list of channel ids (strings)
    },
    "behavior": {
        "max_bot_chain": 3,         # max consecutive character-to-character replies
        "bot_reply_chance": 0.6,    # chance another character answers a character message
        "response_delay": 1.5,      # seconds to wait for more messages before replying
        "backfill_messages": 25,    # channel history loaded on /summon
        "image_token_cost": 800,    # estimated tokens per image in context
        "max_images_in_context": 4, # older images become "[image]" placeholders
        "max_image_side": 896,      # images are downscaled to this
        "fetch_image_links": True,
        "voice_messages": True,     # Discord voice messages are transcribed with Whisper
        "decide_commands": True,    # one extra model call per reply where it must pick a command or "just reply"
        "link_previews": True,     # linked pages are opened and described by the model
        "check_links": True,        # characters' links are opened before posting; dead ones are refused
        "history_limit": 400,       # hard cap on stored messages per channel
        "typing_indicator": True,
    },
    "voice": {
        "local_tts": False,         # speak with Kokoro on this PC instead of ElevenLabs (free, fast)
        "elevenlabs_key": "",
        "tts_model": "eleven_flash_v2_5",
        "local_voice": "af_heart",  # Kokoro voice for characters without their own local voice
        "local_speed": 1.0,
        "local_tts_device": "auto", # auto (GPU if it works) | cpu
        "local_tts_dir": "",        # empty = ./models/kokoro
        "stt_model": "small",       # Whisper size: tiny | base | small | medium | large-v3 | turbo
        "stt_device": "auto",       # auto (GPU if it works) | cuda | cpu
        "stt_language": "",         # e.g. "en"; empty = detect
        "stt_model_dir": "",        # empty = ./models/whisper
        "silence_seconds": 0.8,     # a pause this long ends what someone is saying
        "interruptible": True,      # talking over a character stops it, and it answers the new message instead
    },
    "tools": {name: on for name, (_label, on) in TOOL_LABELS.items()},   # each command on/off
    "files": {                      # the create_file command
        "extensions": "txt, md, py, js, ts, html, css, json, csv, xml, yaml, toml, ini, sh, bat, ps1, c, cpp, "
                      "h, cs, java, go, rs, lua, sql, svg, tex, srt",
        "max_tokens": 4000,         # how long a file may get
    },
    "localchat": {
        "user_name": "",            # your name in the local chat; empty = your Windows user name
    },
    "gifs": {
        "klipy_key": "",            # free key from partner.klipy.com
    },
    "draw": {
        "webui_dir": "",            # A1111 or Forge folder (the one with webui.bat); empty = only connect
        "url": "http://127.0.0.1:7861",
        "auto_start": True,         # start the WebUI (API only, no window) the first time someone draws
        "launch_args": "--medvram-sdxl",
        "model": "",                # checkpoint file name, e.g. ponyDiffusionV6XL.safetensors
        "prompt_prefix": "",        # added before every prompt, e.g. "score_9, score_8_up, " for Pony models
        "negative_prompt": "lowres, bad anatomy, bad hands, worst quality, low quality, watermark, text",
        "width": 1024,
        "height": 1024,
        "steps": 25,
        "cfg_scale": 6.0,
        "sampler": "DPM++ 2M",
        "scheduler": "Karras",
        "unload_minutes": 5,        # free the GPU memory after this long without drawing (0 = never)
    },
    "games": {
        "enabled": True,
        "poll_seconds": 5,
        "processes": [],            # e.g. ["eldenring.exe"]
    },
}


def _merge(base, override):
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.data = deepcopy(DEFAULTS)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.data = _merge(DEFAULTS, saved)
                if "tools" not in saved:   # older configs had two switches for these
                    b = self.data["behavior"]
                    self.data["tools"]["view_profile_picture"] = bool(b.get("allow_profile_pictures", True))
                    self.data["tools"]["web_search"] = self.data["tools"]["read_webpage"] = \
                        bool(b.get("web_search", True))
            except (OSError, json.JSONDecodeError):
                self.data = deepcopy(DEFAULTS)

    def save(self):
        with self._lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)

    def __getitem__(self, section):
        return self.data[section]


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)
