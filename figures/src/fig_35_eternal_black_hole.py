"""fig 35 — the eternal black hole. [SCHEMATIC] full-width.

Reused in Part IX (fig 75), so the two-boundary structure is drawn to be
unmistakable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=35, name="eternal_black_hole", slide=44, type="schematic",
            data_source="none (schematic)",
            caption="The eternal black hole has two boundaries joined by a "
                    "bridge; the single parameter β fixes both the "
                    "entanglement and the throat.")


def build():
    from matplotlib.patches import Ellipse
    fig, ax = new_fig("full")
    blank_axes(ax)
    ax.set_xlim(0, 20); ax.set_ylim(-0.9, 7.4)

    x0, x1, mouth, waist = 4.6, 15.4, 1.95, 0.72
    ts = np.linspace(0, 1, 400)
    xs = x0 + ts * (x1 - x0)
    prof = mouth - (mouth - waist) * np.sin(np.pi * ts)
    ax.fill_between(xs, 4.0 - prof, 4.0 + prof, color=ICE, alpha=0.6, lw=0,
                    zorder=2)
    for sgn in (+1, -1):
        ax.plot(xs, 4.0 + sgn * prof, color=LIFT, lw=2.6, zorder=3)

    for cx, lab in ((x0, "left boundary  L"), (x1, "right boundary  R")):
        ax.add_patch(Ellipse((cx, 4.0), 0.85, 2 * mouth, fc="none", ec=NAVY,
                             lw=3.0, zorder=5))
        note(ax, (cx, 6.55), lab, fs=12, color=INK, weight="bold")

    ax.plot([x0, x1], [4.0, 4.0], color=AMBER, lw=2.0, ls=(0, (5, 4)),
            zorder=6)
    ax.text(10.0, 4.42, "the bridge", ha="center", va="center",
            fontsize=11.5, color=AMBER, weight="bold", bbox=halo(), zorder=7)
    note(ax, (10.0, 2.35), "|TFD(β)⟩", fs=15, color=INK)
    note(ax, (10.0, 1.35),
         "one β controls both the entanglement and the throat",
         fs=11.5, color=MUTED, style="italic")
    note(ax, (x0 - 2.15, 4.0), "SYK_L", fs=11.5, color=MUTED)
    note(ax, (x1 + 2.15, 4.0), "SYK_R", fs=11.5, color=MUTED)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
