"""fig 12 — scalar test vs structured test. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=12, name="what_counts_as_evidence", slide=18, type="schematic",
            data_source="none (schematic)",
            caption="A scalar comparison can only agree or disagree one way; "
                    "a barcode comparison has many independent ways to fail.")


def build():
    fig, ax = new_fig("full")
    blank_axes(ax)
    ax.set_xlim(0, 20); ax.set_ylim(0, 8.4)

    # --- top row: scalar test
    note(ax, (1.4, 6.6), "scalar test", fs=12, color=MUTED, weight="bold",
         ha="left")
    box(ax, (4.6, 5.9), 3.0, 1.3, "S(A) = 0.693", fc=CARD, ec=NAVY, fs=12)
    box(ax, (12.4, 5.9), 3.0, 1.3, "ℓ = 0.693", fc=CARD, ec=NAVY, fs=12)
    arrow(ax, (7.9, 6.55), (12.1, 6.55), color=MUTED, lw=1.8)
    note(ax, (10.0, 7.05), "one comparison", fs=10.5, color=MUTED)
    note(ax, (10.0, 5.35), "agrees, or doesn't", fs=10.5, color=MUTED,
         style="italic")

    ax.plot([0.8, 19.2], [4.75, 4.75], color=GRID, lw=1.2, zorder=1)

    # --- bottom row: structured test
    note(ax, (1.4, 3.7), "structured test", fs=12, color=AMBER, weight="bold",
         ha="left")
    rng = np.random.default_rng(11)
    left = [(0.0, 1.15), (0.0, 1.55), (0.0, 1.95), (0.62, 2.45), (0.85, 3.05)]
    right = [(0.0, 1.05), (0.0, 1.70), (0.0, 2.05), (0.55, 2.30), (0.95, 2.85)]
    for k, ((b0, d0), (b1, d1)) in enumerate(zip(left, right)):
        y = 3.5 - 0.62 * k
        col = AMBER if k >= 3 else NAVY
        bar_h(ax, y, 4.6 + b0, 4.6 + d0, color=col, lw=4.2)
        bar_h(ax, y, 12.4 + b1, 12.4 + d1, color=col, lw=4.2)
        arrow(ax, (4.6 + d0 + 0.15, y), (12.4 + b1 - 0.15, y),
              color=MUTED, lw=0.9, style="-", ls=(0, (3, 3)))
    note(ax, (6.1, 0.55), "barcode", fs=11, color=INK)
    note(ax, (13.9, 0.55), "barcode", fs=11, color=INK)
    note(ax, (10.0, 0.55), "many comparisons", fs=10.5, color=AMBER,
         weight="bold")
    note(ax, (10.0, -0.02), "can disagree in more ways", fs=10.5, color=MUTED,
         style="italic")
    ax.set_ylim(-0.35, 8.0)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
