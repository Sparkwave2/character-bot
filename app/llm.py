"""Runs llama.cpp's llama-server as a child process and talks to its
OpenAI-compatible API. Unloading = stopping the process (frees all VRAM)."""
import asyncio
import datetime
import json
import os
import shlex
import sys
import threading

import aiohttp

from . import winjob
from .config import ROOT
from .log import log

VERBOSE_LOG = os.path.join(ROOT, "logs", "llm-verbose.log")
VERBOSE_MAX_BYTES = 50_000_000     # then it moves to llm-verbose.old.log and starts over
_verbose_lock = threading.Lock()
_verbose_n = 0


def _without_image_data(obj):
    """Base64 images would bury the text; keep a placeholder with their size."""
    if isinstance(obj, dict):
        return {k: (f"[image, {len(v)} characters of base64]" if k == "url" and str(v).startswith("data:")
                    else _without_image_data(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_without_image_data(v) for v in obj]
    return obj


def write_verbose(prompt, data, status):
    """Appends one request: the full prompt, then exactly what the model answered."""
    global _verbose_n
    msg = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
    usage = data.get("usage") or {} if isinstance(data, dict) else {}
    with _verbose_lock:
        _verbose_n += 1
        parts = [f"{'=' * 30} {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  call {_verbose_n} {'=' * 30}",
                 "----- prompt (word for word, as the model reads it) -----", prompt,
                 f"----- reply (HTTP {status}, {usage.get('prompt_tokens', '?')} prompt tokens, "
                 f"{usage.get('completion_tokens', '?')} written) -----"]
        if status != 200:
            parts.append(json.dumps(data, ensure_ascii=False, indent=1))
        if msg.get("reasoning_content"):
            parts += ["[reasoning]", msg["reasoning_content"]]
        if msg.get("content"):
            parts.append(msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"]))
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            parts.append(f"[tool call] {fn.get('name')}({fn.get('arguments')})")
        try:
            os.makedirs(os.path.dirname(VERBOSE_LOG), exist_ok=True)
            if os.path.isfile(VERBOSE_LOG) and os.path.getsize(VERBOSE_LOG) > VERBOSE_MAX_BYTES:
                os.replace(VERBOSE_LOG, VERBOSE_LOG.replace(".log", ".old.log"))
            with open(VERBOSE_LOG, "a", encoding="utf-8") as f:
                f.write("\n".join(parts) + "\n\n")
        except OSError as e:
            log(f"[llm] couldn't write the verbose log: {e}", level="warn")


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
            kwargs = winjob.child_options()
            if sys.platform != "win32":   # the build's own libraries (and the CUDA runtime) are next to it
                lib_dir = os.path.dirname(args[0])
                kwargs["env"] = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(
                    p for p in (lib_dir, os.environ.get("LD_LIBRARY_PATH")) if p))
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
    async def _rendered(self, session, body):
        """The prompt exactly as the model reads it: llama-server applies the chat template for us.
        Falls back to the JSON that was sent if the server can't do that."""
        req = {k: body[k] for k in ("messages", "tools", "tool_choice") if k in body}
        try:
            async with session.post(self.base_url + "/apply-template", json=req,
                                    timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    prompt = (await r.json(content_type=None)).get("prompt")
                    if isinstance(prompt, str):
                        return prompt
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            pass
        return "(llama-server couldn't render the chat template; this is the JSON that was sent)\n" + \
            json.dumps(_without_image_data(req), ensure_ascii=False, indent=1)

    async def chat(self, messages, tools=None, tool_choice="auto", max_tokens=None, stop=None):
        """Returns the raw `message` dict of the first choice. tool_choice="required" makes the
        model call at least one tool (llama-server enforces it with a grammar). `max_tokens`
        overrides the reply limit (files are longer than chat messages); `stop` ends the reply at
        any of these strings. A last message from the assistant is continued (a prefill)."""
        if not self.ready:
            raise RuntimeError("model is not loaded")
        c = self.cfg["llm"]
        body = {
            "messages": messages,
            "temperature": c["temperature"],
            "top_p": c["top_p"],
            "min_p": c["min_p"],
            "repeat_penalty": c["repeat_penalty"],
            "max_tokens": max_tokens or c["max_tokens"],
            "stream": False,
        }
        if stop:
            body["stop"] = stop
        if tools and c.get("use_native_tools", True):
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        session = await self.session()
        verbose = c.get("verbose_log") and await self._rendered(session, body)
        async with session.post(self.base_url + "/v1/chat/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=600)) as r:
            data = await r.json(content_type=None)
            if verbose:
                await asyncio.to_thread(write_verbose, verbose, data, r.status)
            if r.status != 200:
                err = data.get("error", data) if isinstance(data, dict) else data
                # Some templates reject tools; retry once without them.
                if "tools" in body and tool_choice != "auto":
                    raise RuntimeError(f"tool_choice={tool_choice} rejected: {err}")
                if "tools" in body and "tool" in str(err).lower():
                    log(f"[llm] request with tools failed ({err}); retrying without native tools")
                    c["use_native_tools"] = False
                    return await self.chat(messages, tools, max_tokens=max_tokens)
                raise RuntimeError(f"llama-server error {r.status}: {err}")
        u = data.get("usage") or {}
        used = int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
        self.peak_tokens = max(self.peak_tokens, used)
        t = data.get("timings") or {}
        for k in ("prompt_n", "cache_n", "predicted_n"):   # tokens read anew, reused from cache, written
            self.work[k] = self.work.get(k, 0) + int(t.get(k) or 0)
        self.work["calls"] = self.work.get("calls", 0) + 1
        return data["choices"][0]["message"]
