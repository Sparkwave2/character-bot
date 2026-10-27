"""The few things the window does differently on Windows, macOS and Linux."""
import os
import subprocess
import sys

if sys.platform == "win32":
    UI_FONT, MONO_FONT, BOLD_TTF = "Segoe UI", "Consolas", "segoeuib.ttf"
elif sys.platform == "darwin":
    UI_FONT, MONO_FONT, BOLD_TTF = "Helvetica Neue", "Menlo", "Arial Bold.ttf"
else:
    UI_FONT, MONO_FONT, BOLD_TTF = "DejaVu Sans", "DejaVu Sans Mono", "DejaVuSans-Bold.ttf"


def open_path(path):
    """Open a file in its usual program (a folder in the file manager)."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
