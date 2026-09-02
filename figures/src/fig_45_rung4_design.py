"""fig 45 — the rung 4 design. [SCHEMATIC] full-width.

The symmetry is the point: identical treatment on both sides, which is
exactly what rung 3 lacked.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=45, name="rung4_design", slide=66, type="schematic",
            data_source="none (schematic); mirrors rung4.stage_compare",
            caption="Rung 4 puts both sides on the same thermal circle, "
                    "through the same distance rule and the same homology "
                    "computation — the symmetry rung 3 lacked.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 5.2))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(-0.5, 9.2)

    xL, xR = 4.6, 15.4
    box(ax, (xL - 3.3, 7.6), 6.6, 1.15, "SYK  —  exact diagonalisation",
        fc=CARD, ec=NAVY, fs=11.5)
    box(ax, (xR - 3.3, 7.6), 6.6, 1.15, "Schwarzian  —  exact MTV integral",
        fc=CARD, ec=LIFT, fs=11.5)
    note(ax, (xL, 8.95), "EPR side", fs=11, color=MUTED)
    note(ax, (xR, 8.95), "ER side", fs=11, color=MUTED)

    for x, col in ((xL, NAVY), (xR, LIFT)):
        box(ax, (x - 3.3, 5.75), 6.6, 1.05, "G(τ) on the same 24-point circle",
            fc="white", ec=col, fs=11)
        arrow(ax, (x, 7.55), (x, 6.85), color=col, lw=1.6)
        box(ax, (x - 3.3, 4.05), 6.6, 1.05, "d = 1/Ĝ,   Ĝ = G/G(β/2)",
            fc="white", ec=col, fs=11)
        arrow(ax, (x, 5.70), (x, 5.15), color=col, lw=1.6)
        box(ax, (x - 3.3, 2.35), 6.6, 1.05, "ripser  (maxdim = 1)",
            fc="white", ec=col, fs=11)
        arrow(ax, (x, 4.00), (x, 3.45), color=col, lw=1.6)
        arrow(ax, (x, 2.30), (x, 1.60), color=col, lw=1.6)

    box(ax, (6.9, 0.35), 6.2, 1.2,
        "Wasserstein  (H₀ and H₁)", fc=ICE, ec=AMBER, fs=12, lw=2.0)

    for x0, x1 in ((xL + 3.35, 6.85), (xR - 3.35, 13.15)):
        pass
    note(ax, (10.0, 6.28), "identical", fs=10.5, color=AMBER, weight="bold")
    note(ax, (10.0, 4.58), "identical", fs=10.5, color=AMBER, weight="bold")
    note(ax, (10.0, 2.88), "identical", fs=10.5, color=AMBER, weight="bold")
    for y in (6.28, 4.58, 2.88):
        ax.plot([8.0, 12.0], [y - 0.30, y - 0.30], color=AMBER, lw=1.0,
                ls=(0, (3, 3)), zorder=1)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
