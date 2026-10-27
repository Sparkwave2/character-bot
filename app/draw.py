"""Drawing with Stable Diffusion (SDXL) through the AUTOMATIC1111 / Forge WebUI API.

The WebUI is started on demand as an API-only background process (no browser window),
or an already running one with --api is used. Forge manages GPU memory on its own and
shares the card with llama-server; with A1111, --medvram-sdxl keeps SDXL small and the
checkpoint is unloaded after a few idle minutes so the VRAM goes back to other programs."""
import asyncio
import base64
import glob
import os
import subprocess
import sys
from urllib.parse import urlparse

import aiohttp
import psutil

from . import winjob
from .log import log

MODEL_EXTS = (".safetensors", ".ckpt")
SDXL_MIN_BYTES = 5.5e9    # SD 1.5 checkpoints are 2-4 GB, SDXL ones about 6.5 GB


class DrawError(Exception):
    pass


def list_checkpoints(webui_dir):
    """[(file name relative to models/Stable-diffusion, size in bytes)], biggest (SDXL) first."""
    root = os.path.join(webui_dir or "", "models", "Stable-diffusion")
    out = []
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(MODEL_EXTS) and os.path.isfile(p):
            out.append((os.path.relpath(p, root).replace("\\", "/"), os.path.getsize(p)))
    return sorted(out, key=lambda x: (x[1] < SDXL_MIN_BYTES, x[0].lower()))


class ImageGen:
    def __init__(self, cfg):
        self.cfg = cfg
        self.proc = None            # the WebUI if we started it
        self.lock = asyncio.Lock()  # one drawing at a time
        self.unload_task = None
        self.paused_by = None       # game name while games hold the GPU

    @property
    def s(self):
        return self.cfg["draw"]

    @property
    def url(self):
        return (self.s.get("url") or "http://127.0.0.1:7861").rstrip("/")

    async def _alive(self, session):
        try:
            async with session.get(self.url + "/sdapi/v1/samplers", timeout=aiohttp.ClientTimeout(total=3)) as r:
                return r.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def ensure_running(self, session):
        if await self._alive(session):
            return
        folder = self.s.get("webui_dir") or ""
        if not (self.s.get("auto_start", True) and os.path.isfile(os.path.join(folder, "launch.py"))):
            raise DrawError(f"the image generator isn't running at {self.url} (start the WebUI with --api, "
                            "or set its folder on the Drawing page)")
        if not (self.proc and self.proc.returncode is None):
            await self._launch(folder)
        log("[draw] waiting for the WebUI to start (the first start can take a few minutes)")
        for _ in range(600):
            if self.proc.returncode is not None:
                self.proc = None
                raise DrawError("the WebUI exited while starting (see the debug log)")
            if await self._alive(session):
                log("[draw] WebUI is ready")
                return
            await asyncio.sleep(1)
        raise DrawError("the WebUI didn't start within 10 minutes")

    async def _launch(self, folder):
        port = urlparse(self.url).port or 7861
        args = ["--api", "--nowebui", "--port", str(port)] + (self.s.get("launch_args") or "").split()
        if sys.platform == "win32":
            py = os.path.join(folder, "venv", "Scripts", "python.exe")
            cmd = [py, "launch.py", *args] if os.path.isfile(py) else ["cmd", "/c", "webui.bat", *args]
        else:
            py = os.path.join(folder, "venv", "bin", "python")
            cmd = [py, "launch.py", *args] if os.path.isfile(py) else ["bash", "webui.sh", *args]
        env = dict(os.environ, COMMANDLINE_ARGS="", PYTHONUNBUFFERED="1")
        for k in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):   # don't leak our own venv into theirs
            env.pop(k, None)
        log(f"[draw] starting the WebUI: {' '.join(cmd)}")
        kw = winjob.child_options()
        self.proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=folder, env=env, stdin=subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **kw)
        winjob.attach(self.proc.pid)   # closes with the app
        asyncio.create_task(self._pump(self.proc))

    @staticmethod
    async def _pump(proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text and "it/s]" not in text and "s/it]" not in text:   # skip progress bars
                log(f"[webui] {text}", level="debug")

    async def draw(self, session, prompt):
        """Returns PNG bytes."""
        s = self.s
        if self.paused_by:
            raise DrawError(f"drawing is paused while {self.paused_by} is running")
        if not s.get("model"):
            raise DrawError("no Stable Diffusion model is chosen (Drawing page)")
        async with self.lock:
            await self.ensure_running(session)
            body = {
                "prompt": (s.get("prompt_prefix") or "") + prompt,
                "negative_prompt": s.get("negative_prompt") or "",
                "width": int(s["width"]), "height": int(s["height"]),
                "steps": int(s["steps"]), "cfg_scale": float(s["cfg_scale"]),
                "sampler_name": s.get("sampler") or "DPM++ 2M",
                "scheduler": s.get("scheduler") or "Automatic",
                "override_settings": {"sd_model_checkpoint": s["model"]},
                "override_settings_restore_afterwards": False,
                "send_images": True, "save_images": False,
            }
            log(f"[draw] drawing: {body['prompt'][:120]}")
            try:
                async with session.post(self.url + "/sdapi/v1/txt2img", json=body,
                                        timeout=aiohttp.ClientTimeout(total=900)) as r:
                    data = await r.json(content_type=None)
                    if r.status != 200:
                        err = data.get("detail") or data.get("error") or data if isinstance(data, dict) else data
                        raise DrawError(f"the WebUI answered HTTP {r.status}: {str(err)[:300]}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                raise DrawError(f"couldn't reach the WebUI: {e!r}")
        images = data.get("images") or []
        if not images:
            raise DrawError("the WebUI returned no image")
        self._schedule_unload(session)
        return base64.b64decode(images[0].split(",", 1)[-1])

    def _schedule_unload(self, session):
        minutes = float(self.s.get("unload_minutes", 5) or 0)
        if self.unload_task:
            self.unload_task.cancel()
        if minutes > 0:
            self.unload_task = asyncio.create_task(self._unload_later(session, minutes * 60))

    async def _unload_later(self, session, delay):
        await asyncio.sleep(delay)
        await self.unload(session)

    async def unload(self, session):
        """Free the WebUI's GPU memory but keep it running (the next drawing reloads the model)."""
        async with self.lock:
            try:
                async with session.post(self.url + "/sdapi/v1/unload-checkpoint",
                                        timeout=aiohttp.ClientTimeout(total=60)) as r:
                    if r.status == 200:
                        log("[draw] Stable Diffusion model unloaded (GPU memory freed)")
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

    async def stop(self):
        """Stop the WebUI if this app started it."""
        p, self.proc = self.proc, None
        if not p or p.returncode is not None:
            return
        try:   # launch.py starts the real server as a child process
            for c in psutil.Process(p.pid).children(recursive=True):
                c.kill()
        except psutil.Error:
            pass
        try:
            p.kill()
            await asyncio.wait_for(p.wait(), 10)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass
        log("[draw] WebUI stopped")
