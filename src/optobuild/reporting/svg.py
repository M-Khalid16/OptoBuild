"""Minimal, dependency-free SVG rendering of ``reporting.figures.Curve`` objects.

Lines are drawn as polylines split at NaN separators; scatter curves as small
circles. Axis ranges come from the finite data; ticks use 1-2-5 steps. All
text is XML-escaped. Large scatter plots are decimated to at most
``MAX_SCATTER_POINTS`` (every k-th point, stated in the title) to keep reports
small; line data are drawn in full.
"""

from __future__ import annotations

import math
from html import escape

import numpy as np

from optobuild.reporting.figures import Curve

MAX_SCATTER_POINTS = 4000


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick values (1, 2, 5 x 10^k steps) covering [lo, hi]."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + 1e-9 * step:
        ticks.append(0.0 if abs(v) < 1e-12 * step else v)
        v += step
    return ticks


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def render_curve(curve: Curve, width: int = 640, height: int = 360) -> str:
    """SVG document (string) for one curve."""
    x = np.asarray(curve.x, dtype=float)
    y = np.asarray(curve.y, dtype=float)
    title = curve.title
    if curve.scatter and x.size > MAX_SCATTER_POINTS:
        k = math.ceil(x.size / MAX_SCATTER_POINTS)
        x, y = x[::k], y[::k]
        title += f" (1 in {k} points shown)"
    ok = np.isfinite(x) & np.isfinite(y)
    left, right, top, bottom = 70, 15, 30, 45
    pw, ph = width - left - right, height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fff" stroke="#444"/>',
        f'<text x="{width / 2}" y="18" text-anchor="middle" font-size="13">{escape(title)}</text>',
        f'<text x="{left + pw / 2}" y="{height - 8}" text-anchor="middle" font-size="12">'
        f"{escape(curve.x_label)}</text>",
        f'<text x="14" y="{top + ph / 2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 14 {top + ph / 2})">{escape(curve.y_label)}</text>',
    ]
    if not ok.any():
        parts.append(
            f'<text x="{left + pw / 2}" y="{top + ph / 2}" text-anchor="middle">no finite data'
            "</text></svg>"
        )
        return "".join(parts)
    x0, x1 = float(x[ok].min()), float(x[ok].max())
    y0, y1 = float(y[ok].min()), float(y[ok].max())
    if x1 == x0:
        x0, x1 = x0 - 1, x1 + 1
    if y1 == y0:
        y0, y1 = y0 - 1, y1 + 1
    pad = 0.04 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad

    def px(v: np.ndarray | float) -> np.ndarray:
        return left + (np.asarray(v) - x0) / (x1 - x0) * pw

    def py(v: np.ndarray | float) -> np.ndarray:
        return top + ph - (np.asarray(v) - y0) / (y1 - y0) * ph

    for t in nice_ticks(x0, x1):
        xp = float(px(t))
        parts.append(f'<line x1="{xp:.1f}" y1="{top + ph}" x2="{xp:.1f}" y2="{top + ph + 4}" '
                     'stroke="#444"/>')  # fmt: skip
        parts.append(f'<text x="{xp:.1f}" y="{top + ph + 16}" text-anchor="middle" '
                     f'font-size="10">{_fmt(t)}</text>')  # fmt: skip
    for t in nice_ticks(y0, y1):
        yp = float(py(t))
        parts.append(f'<line x1="{left - 4}" y1="{yp:.1f}" x2="{left}" y2="{yp:.1f}" '
                     'stroke="#444"/>')  # fmt: skip
        parts.append(f'<text x="{left - 6}" y="{yp + 3:.1f}" text-anchor="end" '
                     f'font-size="10">{_fmt(t)}</text>')  # fmt: skip
    xs, ys = px(x), py(y)
    if curve.scatter:
        pts = "".join(
            f'<circle cx="{a:.1f}" cy="{b:.1f}" r="1.2"/>'
            for a, b in zip(xs[ok], ys[ok], strict=True)
        )
        parts.append(f'<g fill="#1c7ed6" fill-opacity="0.5">{pts}</g>')
    else:
        segments: list[list[str]] = [[]]
        for a, b, good in zip(xs, ys, ok, strict=True):
            if good:
                segments[-1].append(f"{a:.1f},{b:.1f}")
            elif segments[-1]:
                segments.append([])
        for seg in segments:
            if len(seg) > 1:
                parts.append(f'<polyline fill="none" stroke="#1c7ed6" stroke-width="1" '
                             f'points="{" ".join(seg)}"/>')  # fmt: skip
    parts.append("</svg>")
    return "".join(parts)


__all__ = ["MAX_SCATTER_POINTS", "nice_ticks", "render_curve"]
