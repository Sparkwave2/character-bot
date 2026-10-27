"""Managed llama.cpp: picks the right prebuilt llama-server for this machine,
downloads it from the official ggml-org GitHub releases and keeps it updated.

Installs live in ./llama.cpp/<tag>-<backend>/ with a manifest in
./llama.cpp/installed.json."""
import asyncio
import ctypes.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile

import aiohttp

from .config import ROOT
from .log import log

REPO_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
ENGINE_DIR = os.path.join(ROOT, "llama.cpp")
MANIFEST = os.path.join(ENGINE_DIR, "installed.json")
BACKENDS = ["auto", "cuda", "vulkan", "cpu", "rocm"]
EXE = "llama-server.exe" if sys.platform == "win32" else "llama-server"


def _run(cmd):
    try:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, **kw).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_hardware():
    """Return dict(os, arch, nvidia, cuda_version, vulkan, gpu_name, vram_gb)."""
    osname = {"win32": "win", "darwin": "macos"}.get(sys.platform, "ubuntu")
    m = platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x64"
    hw = {"os": osname, "arch": arch, "nvidia": False, "cuda_version": None, "vulkan": False, "gpu_name": "", "vram_gb": None}
    if osname != "macos":
        out = _run(["nvidia-smi"])
        v = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", out)
        if v:
            hw["nvidia"] = True
            hw["cuda_version"] = v.group(1)
            hw["gpu_name"] = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip().splitlines()[0:1]
            hw["gpu_name"] = hw["gpu_name"][0] if hw["gpu_name"] else "NVIDIA GPU"
            mem = re.findall(r"\d+", _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]))
            hw["vram_gb"] = int(mem[0]) / 1024 if mem else None
        if osname == "win":
            hw["vulkan"] = os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "vulkan-1.dll"))
        else:
            hw["vulkan"] = bool(ctypes.util.find_library("vulkan"))
    return hw


def resolve_backend(choice, hw):
    if hw["os"] == "macos":
        return "metal"
    if choice != "auto":
        return choice
    if hw["nvidia"]:
        return "cuda"
    if hw["vulkan"]:
        return "vulkan"
    return "cpu"


def _ver(s):
    return tuple(int(x) for x in re.findall(r"\d+", s or "0"))


def pick_assets(assets, backend, hw):
    """Choose the asset(s) to download from one release. Returns list of asset dicts or []."""
    names = {a["name"]: a for a in assets}
    o, arch = hw["os"], hw["arch"]

    def find(pattern):
        rx = re.compile(pattern)
        return [a for n, a in names.items() if rx.fullmatch(n)]

    ext = r"\.zip" if o == "win" else r"\.tar\.gz"
    if backend == "metal":
        return find(rf"llama-b\d+-bin-macos-{arch}{ext}")[:1]
    if backend == "cpu":
        tag = "cpu-" if o == "win" else ""
        return find(rf"llama-b\d+-bin-{o}-{tag}{arch}{ext}")[:1]
    if backend == "vulkan":
        return find(rf"llama-b\d+-bin-{o}-vulkan-{arch}{ext}")[:1]
    if backend == "rocm":
        return find(rf"llama-b\d+-bin-{o}-rocm-[\d.]+-{arch}{ext}")[:1]
    if backend == "cuda":
        builds = find(rf"llama-b\d+-bin-{o}-cuda-([\d.]+)-{arch}{ext}")
        if not builds:
            return []
        driver = _ver(hw.get("cuda_version")) or (99,)

        def cver(a):
            return _ver(re.search(r"cuda-([\d.]+)-", a["name"]).group(1))
        # same-or-older major version than the driver supports (minor versions are compatible)
        ok = [a for a in builds if cver(a)[0] <= driver[0]] or builds
        main = max(ok, key=cver)
        v = re.search(r"cuda-([\d.]+)-", main["name"]).group(1)
        runtime = find(rf"cudart-llama-(b\d+-)?bin-{o}-cuda-{re.escape(v)}-{arch}{ext}")
        return [main] + runtime[:1]
    return []


def _safe_extract(archive, dest):
    dest_abs = os.path.abspath(dest)

    def check(name):
        p = os.path.abspath(os.path.join(dest, name))
        if not (p == dest_abs or p.startswith(dest_abs + os.sep)):
            raise RuntimeError(f"unsafe path in archive: {name}")

    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            for n in names:
                check(n)
            z.extractall(dest)
        return [n for n in names if not n.endswith("/")]
    else:
        with tarfile.open(archive, "r:gz") as t:
            members = []
            for m in t.getmembers():
                check(m.name)
                if m.issym() or m.islnk():
                    check(os.path.join(os.path.dirname(m.name), m.linkname))
                if m.isdev():
                    continue
                members.append(m)
            t.extractall(dest, members=members)
        return [m.name for m in members if m.isfile()]


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hw = None
        self.busy = False
        self.progress = (0.0, "")        # (fraction, text) polled by the GUI
        self.latest_tag = None
        self.manifest = self._read_manifest()
        self.cleanup()

    # ---- state --------------------------------------------------------------
    def _read_manifest(self):
        try:
            with open(MANIFEST, "r", encoding="utf-8") as f:
                m = json.load(f)
            if os.path.isfile(m.get("server_path", "")):
                return m
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def cleanup(self):
        """Remove old builds and leftovers. Folders still locked by a running
        server are skipped and retried next time."""
        if not os.path.isdir(ENGINE_DIR):
            return
        keep = os.path.abspath(self.manifest.get("dir", ""))
        for fn in os.listdir(ENGINE_DIR):
            p = os.path.join(ENGINE_DIR, fn)
            if os.path.isdir(p) and os.path.abspath(p) != keep:
                shutil.rmtree(p, ignore_errors=True)

    def hardware(self):
        if self.hw is None:
            self.hw = detect_hardware()
        return self.hw

    @property
    def installed(self):
        return bool(self.manifest)

    def server_path(self):
        """Custom path from settings wins; otherwise the managed install."""
        custom = self.cfg["llm"].get("server_path", "").strip()
        if custom:
            return custom
        return self.manifest.get("server_path", "")

    def describe(self):
        if self.cfg["llm"].get("server_path", "").strip():
            return "Using custom llama-server"
        if not self.manifest:
            return "Not installed"
        return f"llama.cpp {self.manifest['tag']} · {self.manifest['label']}"

    @property
    def update_available(self):
        return bool(self.latest_tag and self.manifest and _ver(self.latest_tag) > _ver(self.manifest.get("tag")))

    # ---- network ------------------------------------------------------------
    async def _releases(self, session):
        async with session.get(REPO_API, headers={"Accept": "application/vnd.github+json"},
                               timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                raise RuntimeError(f"GitHub API returned {r.status}")
            return await r.json()

    async def _find_release(self, session, backend, hw):
        for rel in await self._releases(session):
            if not re.fullmatch(r"b\d+", rel.get("tag_name", "")):
                continue
            assets = pick_assets(rel.get("assets", []), backend, hw)
            if assets:
                return rel, assets
        raise RuntimeError(f"no recent llama.cpp build for {hw['os']}-{hw['arch']} / {backend}")

    async def check_update(self, session):
        hw = self.hardware()
        backend = resolve_backend(self.cfg["llm"].get("backend", "auto"), hw)
        rel, _ = await self._find_release(session, backend, hw)
        self.latest_tag = rel["tag_name"]
        return self.latest_tag

    async def install(self, session, force=False):
        """Download + extract the best build. Returns the llama-server path."""
        if self.busy:
            raise RuntimeError("an install is already running")
        self.busy = True
        try:
            return await self._install(session, force)
        finally:
            self.busy = False

    async def _install(self, session, force):
        hw = self.hardware()
        backend = resolve_backend(self.cfg["llm"].get("backend", "auto"), hw)
        self.progress = (0.0, "Looking up the latest llama.cpp build…")
        rel, assets = await self._find_release(session, backend, hw)
        tag = rel["tag_name"]
        self.latest_tag = tag
        if (not force and self.manifest.get("tag") == tag and self.manifest.get("backend") == backend):
            self.progress = (1.0, "Already up to date")
            return self.manifest["server_path"]

        os.makedirs(ENGINE_DIR, exist_ok=True)
        dl_dir = os.path.join(ENGINE_DIR, "_downloads")
        os.makedirs(dl_dir, exist_ok=True)
        target = os.path.join(ENGINE_DIR, f"{tag}-{backend}")
        n = 2
        while os.path.exists(target):  # never touch a folder a running server may be using
            target = os.path.join(ENGINE_DIR, f"{tag}-{backend}-{n}")
            n += 1
        staging = target + ".partial"
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)

        # The CUDA runtime zip rarely changes between builds: reuse it from the old install.
        runtime = self.manifest.get("runtime", {})
        reused = None
        for a in [a for a in assets if a["name"].startswith("cudart-")]:
            old_dir = self.manifest.get("dir", "")
            if not force and runtime.get("asset") == a["name"] and all(
                    os.path.isfile(os.path.join(old_dir, p)) for p in runtime.get("files", [])):
                for p in runtime["files"]:
                    dest = os.path.join(staging, p)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(os.path.join(old_dir, p), dest)
                reused = runtime
                assets = [x for x in assets if x is not a]
                log(f"[engine] reusing CUDA runtime from the previous install")
        all_assets = assets

        total = sum(a.get("size", 0) for a in assets) or 1
        done = 0
        for a in assets:
            path = os.path.join(dl_dir, a["name"])
            log(f"[engine] downloading {a['name']} ({a.get('size', 0) // 1048576} MB)")
            async with session.get(a["browser_download_url"], timeout=aiohttp.ClientTimeout(total=None, sock_read=120)) as r:
                if r.status != 200:
                    raise RuntimeError(f"download failed ({r.status}) for {a['name']}")
                with open(path, "wb") as f:
                    async for chunk in r.content.iter_chunked(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        self.progress = (min(done / total, 0.99) * 0.9,
                                         f"Downloading {a['name']}  {done / 1048576:.0f} / {total / 1048576:.0f} MB")
            if a.get("size") and os.path.getsize(path) != a["size"]:
                raise RuntimeError(f"incomplete download: {a['name']}")
            self.progress = (0.92, f"Extracting {a['name']}…")
            names = await asyncio.to_thread(_safe_extract, path, staging)
            if a["name"].startswith("cudart-"):
                reused = {"asset": a["name"], "names": [os.path.basename(x) for x in names if os.path.basename(x)]}
            os.remove(path)

        server = None
        for dirpath, _, files in os.walk(staging):
            if EXE in files:
                server = os.path.join(dirpath, EXE)
                break
        if not server:
            raise RuntimeError(f"{EXE} not found in the downloaded build")
        # the CUDA runtime (DLLs on Windows, .so files on Linux) must sit next to the executable
        if backend == "cuda":
            server_dir = os.path.dirname(server)
            for dirpath, _, files in os.walk(staging):
                if dirpath == server_dir:
                    continue
                for fn in files:
                    lib = fn.lower().endswith(".dll") if hw["os"] == "win" else ".so" in fn
                    if lib and not os.path.exists(os.path.join(server_dir, fn)):
                        shutil.move(os.path.join(dirpath, fn), server_dir)
        if os.name != "nt":
            for fn in os.listdir(os.path.dirname(server)):
                p = os.path.join(os.path.dirname(server), fn)
                if os.path.isfile(p) and (fn.startswith("llama") or fn.endswith((".so", ".dylib"))):
                    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        runtime_info = {}
        if reused:
            names = set(reused.get("names") or [os.path.basename(p) for p in reused.get("files", [])])
            files = []
            for dirpath, _, fns in os.walk(staging):
                files += [os.path.relpath(os.path.join(dirpath, fn), staging) for fn in fns if fn in names]
            runtime_info = {"asset": reused["asset"], "files": files}

        os.replace(staging, target)
        server = os.path.join(target, os.path.relpath(server, staging))
        label = {"cuda": "CUDA", "vulkan": "Vulkan", "cpu": "CPU", "rocm": "ROCm", "metal": "Metal"}[backend]
        main = next(a["name"] for a in all_assets if a["name"].startswith("llama-"))
        if backend == "cuda":
            label += " " + re.search(r"cuda-([\d.]+)-", main).group(1)
        self.manifest = {"tag": tag, "backend": backend, "label": label, "dir": target,
                         "server_path": server, "asset": main, "runtime": runtime_info}
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        shutil.rmtree(dl_dir, ignore_errors=True)
        self.progress = (1.0, f"Installed llama.cpp {tag} ({label})")
        log(f"[engine] installed llama.cpp {tag} ({label})")
        return server
