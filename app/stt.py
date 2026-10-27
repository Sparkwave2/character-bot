"""Speech to text with faster-whisper. The model loads on first use and stays loaded."""
import glob
import os
import sys
import threading

import numpy as np

from .config import ROOT
from .log import log

_dll_dirs = []


def _add_cuda_dlls():
    """pip's nvidia-cublas-cu12 / nvidia-cudnn-cu12 put their DLLs in site-packages/nvidia/*/bin."""
    if sys.platform != "win32" or _dll_dirs:
        return
    for sp in sys.path:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                _dll_dirs.append(os.add_dll_directory(d))
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            except OSError:
                pass


class Transcriber:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.loaded_as = None
        self._lock = threading.Lock()

    def _settings(self):
        v = self.cfg["voice"]
        return (v.get("stt_model") or "small", v.get("stt_device") or "auto",
                v.get("stt_model_dir") or os.path.join(ROOT, "models", "whisper"))

    def load(self):
        """Blocking. Reloads if the model or device setting changed."""
        with self._lock:
            name, device, folder = self._settings()
            if self.model is not None and self.loaded_as[:2] == (name, device):
                return
            from faster_whisper import WhisperModel
            _add_cuda_dlls()
            os.makedirs(folder, exist_ok=True)
            tries = [("cuda", "float16"), ("cpu", "int8")] if device == "auto" else \
                [(device, "float16" if device == "cuda" else "int8")]
            err = None
            for dev, ctype in tries:
                try:
                    log(f"[voice] loading Whisper {name} on {dev}")
                    m = WhisperModel(name, device=dev, compute_type=ctype, download_root=folder)
                    if dev == "cuda":  # missing cuBLAS/cuDNN only shows up on first use
                        list(m.transcribe(np.zeros(16000, np.float32))[0])
                    self.model, self.loaded_as = m, (name, device, dev)
                    log(f"[voice] Whisper {name} ready on {dev}")
                    return
                except Exception as e:
                    err = e
                    log(f"[voice] Whisper on {dev} failed: {e}", level="warn")
            raise RuntimeError(f"couldn't load Whisper: {err}")

    def transcribe(self, pcm48_stereo, hint=None):
        """48 kHz 16-bit stereo PCM (Discord's format) -> text. Blocking.
        `hint` primes the spelling of names, e.g. "Talking with Mayo." """
        self.load()
        a = np.frombuffer(pcm48_stereo, dtype=np.int16).astype(np.float32) / 32768.0
        a = a[: len(a) // 6 * 6].reshape(-1, 6)   # 3 stereo frames -> one 16 kHz mono sample
        audio = a.mean(axis=1)
        lang = (self.cfg["voice"].get("stt_language") or "").strip() or None
        with self._lock:
            segments, _info = self.model.transcribe(audio, language=lang, beam_size=1, vad_filter=True,
                                                    condition_on_previous_text=False, initial_prompt=hint)
            parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.6]
        return " ".join(p for p in parts if p).strip()
