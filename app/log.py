"""Tiny thread-safe log bus: the GUI drains `log_queue`."""
import datetime
import queue

log_queue = queue.Queue()


def log(msg, level="info"):
    line = f"{datetime.datetime.now():%H:%M:%S}  {msg}"
    log_queue.put((level, line))
    if level != "debug":
        try:  # console may be missing (pythonw) or unable to encode emoji
            print(line, flush=True)
        except (UnicodeEncodeError, OSError, AttributeError):
            pass
