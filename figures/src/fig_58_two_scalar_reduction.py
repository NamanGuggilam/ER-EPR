"""fig 58 — the collapse to two scalars. [SCHEMATIC] full-width.

The funnel: 576 matrix entries → 12 numbers → 2 numbers → the whole barcode.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=58, name="two_scalar_reduction", slide=81, type="schematic",
            data_source="none (schematic); structure proved in fig 57",
            caption="Everything the persistent homology can see reduces to "
                    "Ĝ(1) and Ĝ(8), weighted 23 : 1 toward the shortest "
                    "separation.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 5.0))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(-0.6, 8.6)

    stages = [
        (2.1, 3.0, "24 × 24\ndistance matrix", "576 entries", NAVY),
        (6.3, 2.6, "12-vector\nĜ(1) … Ĝ(12)", "circulant", NAVY),
        (10.3, 2.2, "two scalars\nĜ(1),  Ĝ(8)", "monotone + A–A", AMBER),
    ]
    for (x, w, lab, sub, col) in stages:
        h = w * 1.15
        box(ax, (x - w / 2, 4.6 - h / 2), w, h, lab, fc=CARD, ec=col, fs=11.5,
            lw=2.0 if col is AMBER else 1.4)
        ax.text(x, 4.6 - h / 2 - 0.42, sub, ha="center", fontsize=9.5,
                color=MUTED)
    arrow(ax, (3.7, 4.6), (4.9, 4.6), color=MUTED, lw=1.8)
    arrow(ax, (7.7, 4.6), (9.1, 4.6), color=AMBER, lw=1.8)

    box(ax, (13.0, 5.75), 6.4, 1.5,
        "H₀ :  23 copies of  [0, 1/Ĝ(1)]", fc="white", ec=NAVY, fs=11.5)
    box(ax, (13.0, 3.55), 6.4, 1.5,
        "H₁ :  one bar  [1/Ĝ(1), 1/Ĝ(8)]", fc="white", ec=AMBER, fs=11.5,
        lw=2.0)
    arrow(ax, (11.5, 5.0), (12.8, 6.5), color=NAVY, lw=1.6)
    arrow(ax, (11.5, 4.2), (12.8, 4.3), color=AMBER, lw=1.6)

    note(ax, (16.2, 2.75), "23 : 1  weighting toward Ĝ(1)", fs=11.5,
         color=AMBER, weight="bold")
    note(ax, (10.0, 1.35),
         "the persistent homology carries no information beyond two "
         "correlator values", fs=11.5, color=INK)
    note(ax, (10.0, 0.62),
         "\"H₁ > 0 on both sides\" is guaranteed by the construction — "
         "it cannot fail, so it cannot be evidence",
         fs=10.5, color=MUTED, style="italic")

    for i, x in enumerate(np.linspace(1.0, 3.2, 7)):
        for y in np.linspace(3.3, 5.9, 7):
            ax.plot([x], [y], "s", color=ICE, ms=3.2, zorder=1)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
