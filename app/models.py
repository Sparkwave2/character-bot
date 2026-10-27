"""Built-in list of vision models that work with the app, how much VRAM each needs, and downloading them
from Hugging Face (with resume)."""
import asyncio
import os
import time
from dataclasses import dataclass, field

import aiohttp

from .config import ROOT
from .log import log

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
KV_FACTOR = {"f16": 1.0, "bf16": 1.0, "q8_0": 0.53, "q5_1": 0.38, "q5_0": 0.35, "q4_1": 0.31, "q4_0": 0.28}
OVERHEAD_GB = 0.8          # compute buffers, CUDA context
TAGS = {                   # tag -> what it means (shown as a tooltip line in the picker)
    "uncensored": "abliterated: refusals removed, plays any character",
    "fast": "replies quickly even on small GPUs",
    "smart": "follows long character prompts and commands well",
    "MoE": "mixture of experts: big-model knowledge at small-model speed",
    "official": "from the original authors",
}


@dataclass
class ModelInfo:
    name: str
    repo: str
    file: str
    size: int                 # bytes of the GGUF
    mmproj: str
    mmproj_size: int
    kv_mb: float              # context memory per 1,000 tokens at f16
    quality: int              # 1-10, used to pick the recommendation
    tags: list = field(default_factory=list)
    note: str = ""

    def vram_gb(self, ctx, kv="q8_0"):
        kv_gb = self.kv_mb * ctx / 1000 * KV_FACTOR.get(kv, 1.0) / 1024
        return (self.size + self.mmproj_size) / 1024 ** 3 + kv_gb + OVERHEAD_GB

    @property
    def download_gb(self):
        return (self.size + self.mmproj_size) / 1e9


# Sizes are the real file sizes on Hugging Face. Every entry is a vision model (it can see images).
CATALOG = [
    ModelInfo("Qwen3-VL 2B", "Qwen/Qwen3-VL-2B-Instruct-GGUF", "Qwen3VL-2B-Instruct-Q4_K_M.gguf", 1107409952,
              "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", 445053216, 112, 2, ["fast", "official"],
              "Tiny; for 4 GB cards. Simple characters only."),
    ModelInfo("Qwen3-VL 4B abliterated", "noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
              "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf", 2497282240, "mmproj-F16.gguf", 836180576,
              144, 4, ["uncensored", "fast"], "Small and quick; decent at roleplay."),
    ModelInfo("Gemma 3 4B", "ggml-org/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf", 2489757856,
              "mmproj-model-f16.gguf", 851251104, 60, 3, ["fast", "official"], "Google's small model."),
    ModelInfo("Qwen3-VL 8B abliterated", "noctrex/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
              "Huihui-Qwen3-VL-8B-Instruct-abliterated-Q4_K_M.gguf", 5027785248, "mmproj-F16.gguf", 1159030272,
              144, 6, ["uncensored", "fast"], "Good balance for 8-12 GB cards."),
    ModelInfo("Qwen3-VL 8B", "Qwen/Qwen3-VL-8B-Instruct-GGUF", "Qwen3VL-8B-Instruct-Q4_K_M.gguf", 5027784800,
              "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf", 752289728, 144, 6, ["official"],
              "The original; refuses some topics."),
    ModelInfo("Gemma 3 12B abliterated", "bartowski/mlabonne_gemma-3-12b-it-abliterated-GGUF",
              "mlabonne_gemma-3-12b-it-abliterated-Q4_K_M.gguf", 7300778656,
              "mmproj-mlabonne_gemma-3-12b-it-abliterated-f16.gguf", 854200224, 80, 6, ["uncensored", "smart"],
              "Natural, lively writing."),
    ModelInfo("Mistral Small 3.2 24B", "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
              "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf", 14333922848, "mmproj-F16.gguf", 878053568,
              160, 8, ["smart"], "Strong writer, reliable commands; for 16-24 GB cards."),
    ModelInfo("Gemma 3 27B", "ggml-org/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf", 16546404736,
              "mmproj-model-f16.gguf", 857739168, 100, 8, ["smart", "official"], "Google's big model; slower."),
    ModelInfo("Qwen3-VL 30B-A3B abliterated", "noctrex/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-GGUF",
              "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-Q4_K_M.gguf", 18556687872, "mmproj-Q8_0.gguf",
              712149376, 96, 9, ["uncensored", "smart", "MoE"], "Best for 24 GB cards: smart and fast."),
    ModelInfo("Qwen3-VL 30B-A3B abliterated (smaller)", "noctrex/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-GGUF",
              "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-IQ3_XXS.gguf", 11849328128, "mmproj-Q8_0.gguf",
              712149376, 96, 7, ["uncensored", "MoE"], "Squeezed to fit 16 GB cards; a little less sharp."),
    ModelInfo("Qwen3-VL 30B-A3B", "Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF", "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf",
              18556687168, "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf", 712148928, 96, 9,
              ["smart", "MoE", "official"], "The original; refuses some topics."),
    ModelInfo("Qwen3-VL 32B", "Qwen/Qwen3-VL-32B-Instruct-GGUF", "Qwen3VL-32B-Instruct-Q4_K_M.gguf", 19762150432,
              "mmproj-Qwen3VL-32B-Instruct-Q8_0.gguf", 772360224, 256, 10, ["smart", "official"],
              "Smartest here but slow, and only a small context fits in 24 GB."),
]


def fit(model, vram_gb, ctx, kv):
    """'fits' | 'tight' | 'too big' (None if the VRAM is unknown)."""
    if not vram_gb:
        return None
    need = model.vram_gb(ctx, kv)
    return "fits" if need <= vram_gb - 0.5 else "tight" if need <= vram_gb + 0.3 else "too big"


def recommended(vram_gb, ctx, kv):
    """The best model that fits; uncensored ones win ties (characters shouldn't refuse their role)."""
    ok = [m for m in CATALOG if fit(m, vram_gb, ctx, kv) == "fits"]
    return max(ok, key=lambda m: (m.quality, "uncensored" in m.tags), default=None)


def models_dir(cfg):
    return cfg["llm"].get("models_dir") or os.path.join(ROOT, "models")


def local_paths(cfg, model):
    """Where the model and its projector go. Each repo gets its own folder because projector names repeat."""
    folder = os.path.join(models_dir(cfg), model.repo.split("/")[-1])
    return os.path.join(folder, model.file), os.path.join(folder, model.mmproj)


def downloaded(cfg, model):
    return all(os.path.isfile(p) for p in local_paths(cfg, model))


class Downloader:
    """One download at a time; the GUI reads `progress` and `busy`."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.busy = None          # ModelInfo being downloaded
        self.progress = (0.0, "")
        self._cancel = False

    def cancel(self):
        self._cancel = True

    async def download(self, session, model):
        if self.busy:
            raise RuntimeError(f"already downloading {self.busy.name}")
        self.busy, self._cancel = model, False
        try:
            model_path, mmproj_path = local_paths(self.cfg, model)
            total = model.size + model.mmproj_size
            done = 0
            for url, dest, size in [(HF.format(repo=model.repo, file=model.mmproj), mmproj_path, model.mmproj_size),
                                    (HF.format(repo=model.repo, file=model.file), model_path, model.size)]:
                await self._fetch(session, url, dest, size, done, total, model.name)
                done += size
            self.progress = (1.0, f"{model.name} is ready")
            log(f"[models] downloaded {model.name} to {os.path.dirname(model_path)}")
            return model_path, mmproj_path
        except asyncio.CancelledError:
            self.progress = (0.0, "Download cancelled (it resumes where it stopped next time)")
            raise
        except Exception as e:
            self.progress = (0.0, f"Download failed: {e}")
            log(f"[models] download of {model.name} failed: {e}", level="warn")
            raise
        finally:
            self.busy = None

    async def _fetch(self, session, url, dest, size, done_before, total, name):
        if os.path.isfile(dest) and os.path.getsize(dest) == size:
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=None, sock_read=60)) as r:
            if r.status == 200:
                have = 0              # the server ignored the range; start over
            elif r.status != 206:
                raise RuntimeError(f"Hugging Face returned HTTP {r.status} for {os.path.basename(dest)}")
            t0, start = time.monotonic(), have
            with open(part, "ab" if have else "wb") as f:
                async for chunk in r.content.iter_chunked(1 << 20):
                    if self._cancel:
                        raise asyncio.CancelledError()
                    f.write(chunk)
                    have += len(chunk)
                    speed = (have - start) / max(time.monotonic() - t0, 0.1)
                    got = done_before + have
                    left = (total - got) / speed if speed else 0
                    self.progress = (got / total, f"Downloading {name}: {got / 1e9:.1f} of {total / 1e9:.1f} GB · "
                                                  f"{speed / 1e6:.0f} MB/s · {int(left // 60)} min left")
        if have != size:
            raise RuntimeError(f"{os.path.basename(dest)} is incomplete ({have} of {size} bytes); try again")
        os.replace(part, dest)
