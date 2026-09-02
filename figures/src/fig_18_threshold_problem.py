"""fig 18 — the threshold problem. [SCHEMATIC] full-width.

Betti numbers under each panel are computed by ripser at that threshold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import vr

META = dict(id=18, name="threshold_problem", slide=27, type="schematic",
            data_source="12-point ring; Betti numbers computed at each t",
            caption="One threshold is never enough: too small and the cloud "
                    "is dust, too large and everything fills in, and nothing "
                    "tells you where the middle is.")


def build():
    P = vr.ring(n=12, r=1.0, jitter=0.055, seed=5)
    STY = {"ICE": ICE, "NAVY": NAVY, "INK": INK}
    ts = [0.34, 0.72, 1.80]      # 1.80 is past the k/n = 1/3 chord (1.732)
    heads = ["t too small", "t about right", "t too large"]

    fig, axes = new_fig("full", ncols=3, figsize=(11.5, 4.3))
    for ax, t, head in zip(axes, ts, heads):
        blank_axes(ax)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-2.35, 2.05)
        vr.draw(ax, P, t, STY, pt_ms=6)
        b = vr.betti(P, t)
        b0, b1 = b[0], (b[1] if len(b) > 1 else 0)
        good = b1 == 1
        note(ax, (0, 1.78), head, fs=11.5,
             color=AMBER if good else MUTED,
             weight="bold" if good else "normal")
        note(ax, (0, -1.82), f"t = {t:.2f}", fs=10.5, color=MUTED)
        note(ax, (0, -2.2), f"β₀ = {b0},  β₁ = {b1}", fs=11,
             color=AMBER if good else MUTED, weight="bold" if good else "normal")
    note(axes[1], (0, -2.62), "only this one shows the real structure — "
         "and nothing in the data tells you where it is",
         fs=10.5, color=INK, style="italic")
    axes[1].set_ylim(-2.9, 2.05)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
