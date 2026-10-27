"""Runs llama.cpp's llama-server as a child process and talks to its
OpenAI-compatible API. Unloading = stopping the process (frees all VRAM)."""
import asyncio
import os
import shlex
import subprocess
import sys

import aiohttp

from . import winjob
from .log import log


class LlamaServer:
    STOPPED, STARTING, READY, STOPPING = "stopped", "starting", "ready", "stopping"

    def __init__(self, cfg, engine):
        self.cfg = cfg
        self.engine = engine
        self.proc = None
        self.state = self.STOPPED
        self.suspended_by = None   # game process name while suspended
        self.peak_tokens = 0       # biggest prompt+reply (tokens) since the caller last reset it
        self.work = {}             # token counts from llama-server's timings since the caller last reset it
        self._lock = asyncio.Lock()
        self._session = None
        self.on_state = lambda s: None

    # ---- process management -------------------------------------------------
    @property
    def base_url(self):
        c = self.cfg["llm"]
        return f"http://{c['host']}:{c['port']}"

    @property
    def ready(self):
        return self.state == self.READY

    def _set(self, state):
        self.state = state
        try:
            self.on_state(state)
        except Exception:
            pass

    def build_args(self):
        c = self.cfg["llm"]
        server = self.engine.server_path()
        if not server or not os.path.isfile(server):
            raise RuntimeError("llama-server is not installed (Model → Engine → Install)")
        if not c["model_path"] or not os.path.isfile(c["model_path"]):
            raise RuntimeError("Model GGUF path is not set or does not exist")
        args = [server, "-m", c["model_path"],
                "--host", c["host"], "--port", str(c["port"]),
                "-c", str(c["ctx_size"]), "-ngl", str(c["gpu_layers"]),
                "--jinja", "-np", "1"]
        if c.get("mmproj_path"):
            if not os.path.isfile(c["mmproj_path"]):
                raise RuntimeError("mmproj path does not exist")
            args += ["--mmproj", c["mmproj_path"]]
        if c.get("threads"):
            args += ["-t", str(c["threads"])]
        kv = (c.get("kv_cache") or "f16").strip().lower()
        if c.get("flash_attn") or kv not in ("f16", ""):
            args += ["-fa", "on"]          # a quantized V cache only works with flash attention
        if kv in ("q8_0", "q4_0", "q5_0", "q5_1", "q4_1", "bf16"):
            args += ["-ctk", kv, "-ctv", kv]
        if c.get("extra_args"):
            args += shlex.split(c["extra_args"], posix=(os.name != "nt"))
        return args

    async def start(self):
        async with self._lock:
            if self.proc and self.proc.returncode is None:
                return
            args = self.build_args()
            if await self._port_in_use():
                raise RuntimeError(f"port {self.cfg['llm']['port']} is already in use (another llama-server "
                                   "still running?). Close it or change the port.")
            log(f"[llm] starting: {' '.join(args)}")
            self._set(self.STARTING)
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                self.proc = await asyncio.create_subprocess_exec(
                    *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **kwargs)
            except OSError as e:
                self._set(self.STOPPED)
                raise RuntimeError(f"failed to launch llama-server: {e}")
            winjob.attach(self.proc.pid)  # dies with us, even if we crash
            asyncio.create_task(self._pump_output(self.proc))
            ok = await self._wait_healthy(self.proc)
            if ok:
                self._set(self.READY)
                log("[llm] model loaded and ready")
            else:
                log("[llm] server failed to become ready (see output above)")
                await self._kill()
                self._set(self.STOPPED)

    async def stop(self):
        async with self._lock:
            if not self.proc:
                self._set(self.STOPPED)
                return
            self._set(self.STOPPING)
            await self._kill()
            self._set(self.STOPPED)
            log("[llm] model unloaded")

    async def _kill(self):
        p, self.proc = self.proc, None
        if not p or p.returncode is not None:
            return
        try:
            p.terminate()
            await asyncio.wait_for(p.wait(), 10)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                p.kill()
            except ProcessLookupError:
                pass

    async def _pump_output(self, proc):
        noisy = ("slot ", "update_slots", "srv  log_server_r", "print_timing", "prompt eval", "eval time", "total time")
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text and not any(n in text for n in noisy):
                log(f"[llama] {text}", level="debug")
        if proc is self.proc and self.state == self.READY:
            log(f"[llm] llama-server exited unexpectedly (code {proc.returncode})")
            self.proc = None
            self._set(self.STOPPED)

    async def _port_in_use(self):
        c = self.cfg["llm"]
        try:
            _, w = await asyncio.wait_for(asyncio.open_connection(c["host"], int(c["port"])), 1)
        except (OSError, asyncio.TimeoutError):
            return False
        w.close()
        return True

    async def _wait_healthy(self, proc, timeout=900):
        session = await self.session()
        for _ in range(int(timeout * 2)):
            if proc.returncode is not None:
                return False
            try:
                async with session.get(self.base_url + "/health", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        await self.stop()
        if self._session and not self._session.closed:
            await self._session.close()

    # ---- inference ----------------------------------------------------------
    async def chat(self, messages, tools=None, tool_choice="auto"):
        """Returns the raw `message` dict of the first choice. tool_choice="required" makes the
        model call at least one tool (llama-server enforces it with a grammar)."""
        if not self.ready:
            raise RuntimeError("model is not loaded")
        c = self.cfg["llm"]
        body = {
            "messages": messages,
            "temperature": c["temperature"],
            "top_p": c["top_p"],
            "min_p": c["min_p"],
            "repeat_penalty": c["repeat_penalty"],
            "max_tokens": c["max_tokens"],
            "stream": False,
        }
        if tools and c.get("use_native_tools", True):
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        session = await self.session()
        async with session.post(self.base_url + "/v1/chat/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=600)) as r:
            data = await r.json(content_type=None)
            if r.status != 200:
                err = data.get("error", data) if isinstance(data, dict) else data
                # Some templates reject tools; retry once without them.
                if "tools" in body and tool_choice != "auto":
                    raise RuntimeError(f"tool_choice={tool_choice} rejected: {err}")
                if "tools" in body and "tool" in str(err).lower():
                    log(f"[llm] request with tools failed ({err}); retrying without native tools")
                    c["use_native_tools"] = False
                    return await self.chat(messages, tools)
                raise RuntimeError(f"llama-server error {r.status}: {err}")
        u = data.get("usage") or {}
        used = int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
        self.peak_tokens = max(self.peak_tokens, used)
        t = data.get("timings") or {}
        for k in ("prompt_n", "cache_n", "predicted_n"):   # tokens read anew, reused from cache, written
            self.work[k] = self.work.get(k, 0) + int(t.get(k) or 0)
        self.work["calls"] = self.work.get("calls", 0) + 1
        return data["choices"][0]["message"]
