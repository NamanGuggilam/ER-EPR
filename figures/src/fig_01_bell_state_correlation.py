"""fig 01 — Bell-state correlation. [SCHEMATIC] half-slide. matplotlib patches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=1, name="bell_state_correlation", slide=5, type="schematic",
            data_source="none (schematic)",
            caption="The Bell state |Φ⁺⟩: each qubit is individually "
                    "indefinite, yet the two outcomes always agree.")


def build():
    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.5)

    circle(ax, (2.0, 5.4), 0.85, fc=ICE, ec=NAVY, lw=1.8)
    circle(ax, (8.0, 5.4), 0.85, fc=ICE, ec=NAVY, lw=1.8)
    note(ax, (2.0, 5.4), "A", fs=15, color=INK, weight="bold")
    note(ax, (8.0, 5.4), "B", fs=15, color=INK, weight="bold")

    wavy(ax, (2.9, 5.4), (7.1, 5.4), amp=0.34, cycles=5, color=AMBER, lw=2.4)
    note(ax, (5.0, 6.35), "entangled", color=AMBER, fs=11, weight="bold")

    note(ax, (5.0, 4.15), "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2", color=INK, fs=12)

    # outcome table
    rows = [("A = 0", "⇒", "B = 0", "½"), ("A = 1", "⇒", "B = 1", "½")]
    xs = [2.75, 4.35, 5.6, 7.6]
    note(ax, (xs[0], 3.05), "outcome", color=MUTED, fs=9.5)
    note(ax, (xs[2], 3.05), "outcome", color=MUTED, fs=9.5)
    note(ax, (xs[3], 3.05), "probability", color=MUTED, fs=9.5)
    for i, (a, imp, b, p) in enumerate(rows):
        y = 2.35 - 0.78 * i
        box(ax, (xs[0] - 0.72, y - 0.29), 1.44, 0.58, a, fc=CARD, ec=NAVY, fs=11)
        note(ax, (xs[1], y), imp, color=MUTED, fs=13)
        box(ax, (xs[2] - 0.72, y - 0.29), 1.44, 0.58, b, fc=CARD, ec=NAVY, fs=11)
        note(ax, (xs[3], y), p, color=INK, fs=11)

    note(ax, (5.0, 0.5),
         "neither has a definite value before measurement",
         color=MUTED, fs=10.5, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
