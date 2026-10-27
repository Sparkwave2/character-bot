"""Pictures for the window are opened and resized on worker threads, so a big image (or a long chat full of
them) never freezes the window. Only turning the result into a Tk image happens on the Tk thread."""
import os
import queue
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageSequence


def file_key(path, *extra):
    """Cache key that changes when the file does (a new profile picture under the same name)."""
    try:
        stamp = os.path.getmtime(path) if path else None
    except OSError:
        stamp = None
    return (path, stamp) + extra


def circle(src, size, fill="#2a2f40"):
    """A round crop at twice `size` (for sharp scaling); `src` is a path, PIL image or None (plain circle)."""
    try:
        im = src.convert("RGBA") if isinstance(src, Image.Image) else Image.open(src).convert("RGBA")
    except Exception:
        im = Image.new("RGBA", (size, size), fill)
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size * 2, size * 2),
                                                                                    Image.LANCZOS)
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, *im.size), fill=255)
    im.putalpha(mask)
    return im


def thumbnail(path, box):
    im = Image.open(path)
    im.thumbnail(box)
    return im.convert("RGBA")


def gif_frames(path, box, limit=120):
    """[(frame, milliseconds)] of an animated picture, each frame shrunk to fit `box`."""
    im = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(im):
        f = frame.convert("RGBA")
        f.thumbnail(box)
        frames.append((f, max(20, int(frame.info.get("duration") or 100))))
        if len(frames) >= limit:
            break
    return frames


class ImageLoader:
    def __init__(self, root, workers=3):
        self.root = root
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="images")
        self.done = queue.Queue()
        self.cache = {}       # key -> prepared PIL image (None if it couldn't be opened)
        self.waiting = {}     # key -> [deliver, ...]
        self.root.after(30, self._pump)

    def get(self, key, prepare, deliver, cache=True):
        """Runs `prepare() -> PIL image` (or anything else, e.g. GIF frames) on a worker, once per key,
        and then `deliver(result)` on the Tk thread; right away if it is cached. `result` is None when
        the file couldn't be opened."""
        if key in self.cache:
            return deliver(self.cache[key])
        if key in self.waiting:
            self.waiting[key].append(deliver)
            return
        self.waiting[key] = [deliver]

        def work():
            try:
                im = prepare()
                if isinstance(im, Image.Image):
                    im.load()
            except Exception:
                im = None
            self.done.put((key, im, cache))
        self.pool.submit(work)

    def _pump(self):
        try:
            while True:
                key, im, keep = self.done.get_nowait()
                if len(self.cache) > 400:
                    self.cache.clear()
                if keep:
                    self.cache[key] = im
                for deliver in self.waiting.pop(key, []):
                    try:
                        deliver(im)
                    except tk.TclError:   # the widget it was for is gone
                        pass
        except queue.Empty:
            pass
        self.root.after(30, self._pump)
