"""fig 38 — the rung ladder. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=38, name="rung_ladder", slide=47, type="schematic",
            data_source="none (schematic); statuses as recorded in the repo",
            caption="The ladder: each rung makes the comparison harder to "
                    "pass, and rung 4 is the first that could fail.")


def build():
    rows = [
        (1, "Bell pair vs RT throat", "pipeline only", CLAY),
        (2, "two-level TFD across β", "pipeline only", CLAY),
        (3, "SYK vs a JT time-slice", "diagnosed", AMBER),
        (4, "SYK vs the exact Schwarzian", "closed — negative", TEAL),
        (5, "to be specified", "in progress", NAVY),
    ]
    fig, ax = new_fig("full", figsize=(11.5, 5.0))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(-0.4, 5.9)

    for i, (n, what, status, col) in enumerate(rows):
        y = 4.7 - i * 1.02
        ax.plot([2.3, 17.7], [y, y], color=GRID, lw=1.2, zorder=1)
        circle(ax, (2.3, y), 0.30, fc=col, ec=col, zorder=4)
        ax.text(2.3, y, str(n), ha="center", va="center", fontsize=11,
                color="white", weight="bold", zorder=5)
        ax.text(3.1, y, what, ha="left", va="center", fontsize=11.5,
                color=INK, zorder=4)
        box(ax, (13.9, y - 0.28), 3.8, 0.56, status, fc=CARD, ec=col, tc=col,
            fs=10.5, lw=1.6)

    ax.plot([2.3, 2.3], [4.7, 0.62], color=GRID, lw=2.0, zorder=0)
    note(ax, (10.0, -0.10),
         "each rung adds structure the previous one could not test",
         fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
