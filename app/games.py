"""Unloads the model while any listed game process is running."""
import asyncio

import psutil

from .log import log


def running_process_names():
    names = set()
    for p in psutil.process_iter(["name"]):
        n = p.info.get("name")
        if n:
            names.add(n)
    return names


class GameWatcher:
    def __init__(self, core):
        self.core = core
        self.active_game = None
        self.on_change = lambda g: None

    async def run(self):
        while True:
            g = self.core.cfg["games"]
            try:
                await self.tick(g)
            except Exception as e:
                log(f"[games] watcher error: {e!r}")
            await asyncio.sleep(max(1, float(g.get("poll_seconds", 5))))

    async def tick(self, g):
        llm = self.core.llm
        watch = {p.lower() for p in g.get("processes", []) if p.strip()}
        found = None
        if g.get("enabled", True) and watch:
            names = await asyncio.to_thread(running_process_names)
            found = next((n for n in names if n.lower() in watch), None)

        if found and not self.active_game:
            self.active_game = found
            self.on_change(found)
            if llm.state != llm.STOPPED:
                log(f"[games] {found} detected; unloading model")
                llm.suspended_by = found
                await llm.stop()
            elif self.core.want_loaded:
                llm.suspended_by = found
        elif not found and self.active_game:
            log(f"[games] {self.active_game} closed")
            self.active_game = None
            self.on_change(None)
            if llm.suspended_by:
                llm.suspended_by = None
                if self.core.want_loaded:
                    log("[games] reloading model")
                    await self.core.load_model()
