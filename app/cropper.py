"""Square crop dialog for profile pictures.

Drag inside the box to move it, drag a corner to resize, or use the mouse
wheel. The circle shows what Discord will display."""
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageEnhance, ImageTk

VIEW_W, VIEW_H = 560, 460
HANDLE = 9            # handle half-size in screen px
MIN_SIDE = 16         # smallest crop in image px
OUTPUT_SIDE = 512     # saved avatar size


def load_image(src):
    """Open a path or file-like as RGBA (first frame for GIFs)."""
    im = Image.open(src)
    im.seek(0)
    return im.convert("RGBA")


def circle_preview(im, size):
    im = im.resize((size * 2, size * 2), Image.LANCZOS)
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, *im.size), fill=255)
    out = im.copy()
    out.putalpha(mask)
    return ctk.CTkImage(out, size=(size, size))


class CropDialog(ctk.CTkToplevel):
    def __init__(self, master, image, on_done, colors):
        super().__init__(master, fg_color=colors["bg"])
        self.title("Crop profile picture")
        self.resizable(False, False)
        self.transient(master)
        self.on_done = on_done
        self.c = colors
        self.src = image
        w, h = image.size

        # fit the image into the view
        self.scale = min(VIEW_W / w, VIEW_H / h)
        dw, dh = max(1, round(w * self.scale)), max(1, round(h * self.scale))
        self.ox, self.oy = (VIEW_W - dw) // 2, (VIEW_H - dh) // 2
        disp = image.resize((dw, dh), Image.LANCZOS)
        bg = Image.new("RGBA", disp.size, (40, 44, 58, 255))  # shows through transparent pixels
        self.bright = Image.alpha_composite(bg, disp).convert("RGB")
        self.dim = ImageEnhance.Brightness(self.bright).enhance(0.35)
        self.reset()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(padx=18, pady=18)
        self.canvas = tk.Canvas(body, width=VIEW_W, height=VIEW_H, bg=colors["panel"],
                                highlightthickness=0, cursor="fleur")
        self.canvas.grid(row=0, column=0, rowspan=2)
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw")

        side = ctk.CTkFrame(body, fg_color="transparent")
        side.grid(row=0, column=1, sticky="n", padx=(18, 0))
        ctk.CTkLabel(side, text="Preview", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        self.big = ctk.CTkLabel(side, text="")
        self.big.pack(pady=(10, 6))
        row = ctk.CTkFrame(side, fg_color="transparent")
        row.pack()
        self.small = ctk.CTkLabel(row, text="")
        self.small.pack(side="left", padx=4)
        ctk.CTkLabel(row, text="in chat", text_color=colors["muted"]).pack(side="left")
        self.size_label = ctk.CTkLabel(side, text="", text_color=colors["muted"])
        self.size_label.pack(pady=(14, 0))
        ctk.CTkLabel(side, justify="left", text_color=colors["muted"], font=ctk.CTkFont(size=11), text=(
            "Drag to move\nDrag a corner to resize\nScroll to zoom")).pack(anchor="w", pady=(14, 0))

        bar = ctk.CTkFrame(body, fg_color="transparent")
        bar.grid(row=1, column=1, sticky="s", padx=(18, 0))
        mk = dict(height=36, corner_radius=10, width=150)
        ctk.CTkButton(bar, text="Use crop", command=self.accept, fg_color=colors["accent"],
                      hover_color=colors["accent_hover"], **mk).pack(pady=3)
        ctk.CTkButton(bar, text="Reset", command=lambda: (self.reset(), self.redraw()), fg_color=colors["card"],
                      hover_color=colors["border"], **mk).pack(pady=3)
        ctk.CTkButton(bar, text="Cancel", command=self.destroy, fg_color=colors["card"],
                      hover_color=colors["border"], **mk).pack(pady=3)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.zoom(0.95))  # X11
        self.canvas.bind("<Button-5>", lambda e: self.zoom(1.05))
        self.bind("<Return>", lambda e: self.accept())
        self.bind("<Escape>", lambda e: self.destroy())
        self._drag = None
        self.redraw()
        self.after(100, self._grab)

    def _grab(self):
        try:
            self.grab_set()
            self.focus_force()
        except tk.TclError:
            pass

    # ---- geometry (box is x, y, side in source-image pixels) ---------------
    def reset(self):
        w, h = self.src.size
        s = min(w, h)
        self.box = [(w - s) / 2, (h - s) / 2, s]

    def clamp(self):
        w, h = self.src.size
        x, y, s = self.box
        s = max(min(MIN_SIDE, w, h), min(s, w, h))
        x = min(max(0, x), w - s)
        y = min(max(0, y), h - s)
        self.box = [x, y, s]

    def to_screen(self, x, y):
        return self.ox + x * self.scale, self.oy + y * self.scale

    def to_image(self, sx, sy):
        return (sx - self.ox) / self.scale, (sy - self.oy) / self.scale

    def corners(self):
        x, y, s = self.box
        return {"nw": (x, y), "ne": (x + s, y), "sw": (x, y + s), "se": (x + s, y + s)}

    def hit(self, sx, sy):
        for name, (cx, cy) in self.corners().items():
            px, py = self.to_screen(cx, cy)
            if abs(px - sx) <= HANDLE + 3 and abs(py - sy) <= HANDLE + 3:
                return name
        x, y, s = self.box
        ix, iy = self.to_image(sx, sy)
        if x <= ix <= x + s and y <= iy <= y + s:
            return "move"
        return None

    # ---- mouse ---------------------------------------------------------------
    def on_hover(self, e):
        h = self.hit(e.x, e.y)
        cur = {"nw": "size_nw_se", "se": "size_nw_se", "ne": "size_ne_sw", "sw": "size_ne_sw",
               "move": "fleur"}.get(h, "arrow")
        try:
            self.canvas.configure(cursor=cur)
        except tk.TclError:
            self.canvas.configure(cursor="crosshair" if h in ("nw", "ne", "sw", "se") else "fleur")

    def on_press(self, e):
        mode = self.hit(e.x, e.y)
        if mode is None:  # click outside: centre the box there
            x, y, s = self.box
            ix, iy = self.to_image(e.x, e.y)
            self.box = [ix - s / 2, iy - s / 2, s]
            self.clamp()
            self.redraw()
            mode = "move"
        x, y, s = self.box
        opposite = {"nw": (x + s, y + s), "ne": (x, y + s), "sw": (x + s, y), "se": (x, y)}
        self._drag = (mode, self.to_image(e.x, e.y), list(self.box), opposite.get(mode))

    def on_drag(self, e):
        if not self._drag:
            return
        mode, (sx, sy), start, anchor = self._drag
        ix, iy = self.to_image(e.x, e.y)
        w, h = self.src.size
        if mode == "move":
            self.box = [start[0] + ix - sx, start[1] + iy - sy, start[2]]
        else:
            ax, ay = anchor
            dx = 1 if mode in ("ne", "se") else -1
            dy = 1 if mode in ("sw", "se") else -1
            side = max((ix - ax) * dx, (iy - ay) * dy, MIN_SIDE)
            # keep the box inside the image on the side it grows toward
            side = min(side, (w - ax) if dx > 0 else ax, (h - ay) if dy > 0 else ay)
            self.box = [ax if dx > 0 else ax - side, ay if dy > 0 else ay - side, side]
        self.clamp()
        self.redraw()

    def on_wheel(self, e):
        self.zoom(0.95 if e.delta > 0 else 1.05)

    def zoom(self, f):
        x, y, s = self.box
        w, h = self.src.size
        cx, cy = x + s / 2, y + s / 2
        ns = max(min(MIN_SIDE, w, h), min(s * f, w, h))  # clamp size first so the box doesn't drift
        self.box = [cx - ns / 2, cy - ns / 2, ns]
        self.clamp()
        self.redraw()

    # ---- drawing ---------------------------------------------------------------
    def crop(self):
        x, y, s = self.box
        x, y, s = round(x), round(y), max(1, round(s))
        im = self.src.crop((x, y, x + s, y + s))
        if s > OUTPUT_SIDE:
            im = im.resize((OUTPUT_SIDE, OUTPUT_SIDE), Image.LANCZOS)
        return im

    def redraw(self):
        x, y, s = self.box
        frame = Image.new("RGB", (VIEW_W, VIEW_H), self.c["panel"])
        frame.paste(self.dim, (self.ox, self.oy))
        l, t = self.to_screen(x, y)
        r, b = self.to_screen(x + s, y + s)
        li, ti = round(l - self.ox), round(t - self.oy)
        ri, bi = round(r - self.ox), round(b - self.oy)
        frame.paste(self.bright.crop((li, ti, ri, bi)), (round(l), round(t)))
        d = ImageDraw.Draw(frame)
        d.ellipse((l, t, r, b), outline=(255, 255, 255), width=1)
        d.rectangle((l, t, r, b), outline=self.c["accent"], width=2)
        for cx, cy in ((l, t), (r, t), (l, b), (r, b)):
            d.rectangle((cx - HANDLE, cy - HANDLE, cx + HANDLE, cy + HANDLE), fill=(255, 255, 255),
                        outline=self.c["accent"], width=2)
        self._photo = ImageTk.PhotoImage(frame)
        self.canvas.itemconfigure(self.canvas_img, image=self._photo)
        cropped = self.crop()
        self.big.configure(image=circle_preview(cropped, 128))
        self.small.configure(image=circle_preview(cropped, 40))
        self.size_label.configure(text=f"{round(s)} × {round(s)} px")

    def accept(self):
        result = self.crop()
        self.destroy()
        self.on_done(result)
