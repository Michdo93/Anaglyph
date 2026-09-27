#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_anaglyph.py — erzeugt aus Stereo-Material ein Rot-Cyan-Anaglyph-Video.

Trennung der beiden Augenbilder erfolgt spaeter in der Rot-Cyan-Brille, nicht
im Display -> funktioniert daher auch auf einem POV-Hologramm-Luefter, der nur
ein einziges (unpolarisiertes) Farbbild darstellt.

Eingabe-Varianten (eine waehlen):
  --sbs   video.mp4              Side-by-Side-Stereo (linke Haelfte = L, rechte = R)
  --tab   video.mp4             Top-and-Bottom-Stereo (obere Haelfte = L, untere = R)
  --left  L.mp4 --right R.mp4   zwei getrennte Videos gleicher Laenge/Framerate
  --mono  video.mp4 [--shift N] EIN 2D-Video; erzeugt durch horizontalen Versatz
                                einen Pseudo-3D-Effekt (KEIN echtes Stereo!)

Anaglyph-Methoden (--method):
  dubois     (Standard) Dubois-Matrizen, bestes Verhaeltnis Farbe/Ghosting
  color      volle Farbe, mehr Ghosting  (R = L.R, G/B = R.G/R.B)
  half-color L wird entsaettigt -> weniger Netzhaut-Rivalitaet
  gray       beide Augen in Graustufen -> minimales Ghosting, keine Farbe

Fan-Format (optional):
  --size N   Ausgabe quadratisch NxN (der Luefter ist rund/quadratisch)
  --circle   alles ausserhalb des eingeschriebenen Kreises schwarz (Rand
             ist auf dem Luefter ohnehin unsichtbar)
  --depth D  horizontale Bildverschiebung in Pixeln (Tiefe/Parallaxe justieren;
             positiv = mehr Tiefe, 0 = in Schirmebene). Default 0.

Beispiele:
  python make_anaglyph.py --sbs stereo.mp4 -o out.mp4
  python make_anaglyph.py --left l.mp4 --right r.mp4 --size 512 --circle -o fan.mp4
  python make_anaglyph.py --mono clip.mp4 --shift 8 --circle -o pseudo.mp4
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import numpy as np
import cv2

# Dubois Rot/Cyan-Matrizen (arbeiten auf RGB, 0..1)
_ML = np.array([[ 0.437,  0.449,  0.164],
                [-0.062, -0.062, -0.024],
                [-0.048, -0.050, -0.017]], dtype=np.float32)
_MR = np.array([[-0.011, -0.032, -0.007],
                [ 0.377,  0.761,  0.009],
                [-0.026, -0.093,  1.234]], dtype=np.float32)


def _to_rgb(f):        # OpenCV liefert BGR
    return f[:, :, ::-1].astype(np.float32) / 255.0

def _to_bgr(rgb):
    out = np.clip(rgb, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)[:, :, ::-1]


def anaglyph(left_bgr, right_bgr, method="dubois", depth=0):
    """Kombiniert linkes+rechtes Frame (BGR uint8) zu einem Anaglyph-BGR-Frame."""
    L = _to_rgb(left_bgr)
    R = _to_rgb(right_bgr)
    if depth:                                   # rechtes Bild horizontal schieben
        R = np.roll(R, depth, axis=1)
        if depth > 0:
            R[:, :depth] = 0
        else:
            R[:, depth:] = 0

    if method == "dubois":
        out = L @ _ML.T + R @ _MR.T
    elif method == "color":
        out = np.stack([L[:, :, 0], R[:, :, 1], R[:, :, 2]], axis=-1)
    elif method == "half-color":
        lum = L @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out = np.stack([lum, R[:, :, 1], R[:, :, 2]], axis=-1)
    elif method == "gray":
        ll = L @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        rl = R @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        out = np.stack([ll, rl, rl], axis=-1)
    else:
        raise ValueError(f"unbekannte Methode: {method}")
    return _to_bgr(out)


def _square(frame, size, circle):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = frame[y0:y0 + s, x0:x0 + s]
    if size and size != s:
        sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    if circle:
        n = sq.shape[0]
        yy, xx = np.ogrid[:n, :n]
        r = n / 2.0
        mask = (xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2 <= r ** 2
        sq = sq * mask[:, :, None]
    return sq


class _Reader:
    """Liest Frames; splittet SBS/TAB oder liest zwei Quellen parallel."""
    def __init__(self, args):
        self.mode = args.mode
        if self.mode in ("sbs", "tab", "mono"):
            self.cap = cv2.VideoCapture(args.src)
            if not self.cap.isOpened():
                sys.exit(f"Kann Video nicht oeffnen: {args.src}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        else:  # left/right
            self.capL = cv2.VideoCapture(args.left)
            self.capR = cv2.VideoCapture(args.right)
            if not (self.capL.isOpened() and self.capR.isOpened()):
                sys.exit("Kann left/right-Video nicht oeffnen.")
            self.fps = self.capL.get(cv2.CAP_PROP_FPS) or 25.0
        self.shift = args.shift

    def __iter__(self):
        if self.mode in ("sbs", "tab", "mono"):
            while True:
                ok, f = self.cap.read()
                if not ok:
                    break
                if self.mode == "sbs":
                    w = f.shape[1] // 2
                    yield f[:, :w], f[:, w:2 * w]
                elif self.mode == "tab":
                    h = f.shape[0] // 2
                    yield f[:h], f[h:2 * h]
                else:  # mono -> Pseudo-Stereo per Versatz
                    r = np.roll(f, -abs(self.shift), axis=1)
                    r[:, -abs(self.shift):] = 0
                    yield f, r
        else:
            while True:
                okL, fL = self.capL.read()
                okR, fR = self.capR.read()
                if not (okL and okR):
                    break
                yield fL, fR

    def release(self):
        for c in ("cap", "capL", "capR"):
            if hasattr(self, c):
                getattr(self, c).release()


def main(argv=None):
    p = argparse.ArgumentParser(description="Stereo-Video -> Rot-Cyan-Anaglyph")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sbs", metavar="VIDEO", help="Side-by-Side-Stereo")
    g.add_argument("--tab", metavar="VIDEO", help="Top-and-Bottom-Stereo")
    g.add_argument("--mono", metavar="VIDEO", help="2D-Video (Pseudo-3D)")
    p.add_argument("--left", metavar="VIDEO")
    p.add_argument("--right", metavar="VIDEO")
    p.add_argument("-o", "--out", default="anaglyph.mp4")
    p.add_argument("--method", default="dubois",
                   choices=["dubois", "color", "half-color", "gray"])
    p.add_argument("--shift", type=int, default=6,
                   help="Pseudo-3D-Versatz fuer --mono (px)")
    p.add_argument("--depth", type=int, default=0,
                   help="Parallaxe-Feinjustage (px), fuer alle Modi")
    p.add_argument("--size", type=int, default=0, help="quadratische Ausgabe NxN")
    p.add_argument("--circle", action="store_true", help="Kreismaske (Rand schwarz)")
    p.add_argument("--fps", type=float, default=0.0, help="Ausgabe-FPS ueberschreiben")
    args = p.parse_args(argv)

    if args.left or args.right:
        if not (args.left and args.right):
            sys.exit("--left UND --right zusammen angeben.")
        args.mode, args.src = "lr", None
    elif args.sbs:
        args.mode, args.src = "sbs", args.sbs
    elif args.tab:
        args.mode, args.src = "tab", args.tab
    else:
        args.mode, args.src = "mono", args.mono

    reader = _Reader(args)
    fps = args.fps or reader.fps
    writer = None
    n = 0
    for L, R in reader:
        frame = anaglyph(L, R, args.method, args.depth)
        if args.size or args.circle:
            frame = _square(frame, args.size, args.circle)
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        writer.write(frame)
        n += 1
    reader.release()
    if writer:
        writer.release()
    print(f"{n} Frames -> {args.out}  ({args.method}, {fps:.2f} fps"
          + (f", {args.size}x{args.size}" if args.size else "")
          + (", Kreis" if args.circle else "") + ")")
    print("Rot-Cyan-Brille aufsetzen (Rot = linkes Auge). In der Original-App "
          "als Video importieren -> BIN erzeugen -> auf den Luefter laden.")


if __name__ == "__main__":
    main()
