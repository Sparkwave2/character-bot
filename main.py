import sys

from app.core import Core
from app.gui import App


def main():
    if sys.platform == "win32":
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    core = Core()
    App(core).mainloop()


if __name__ == "__main__":
    main()
