"""fig 17 — cycle vs boundary. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=17, name="cycle_vs_boundary", slide=26, type="schematic",
            data_source="none (schematic)",
            caption="Homology counts cycles that are not boundaries: the "
                    "hollow triangle is a hole, the filled one is not.")


def build():
    fig, axes = new_fig("half", ncols=2, figsize=(5.6, 3.9))
    from matplotlib.patches import Polygon
    V = np.array([[-0.85, -0.55], [0.85, -0.55], [0.0, 0.95]])

    for ax, filled in zip(axes, (False, True)):
        blank_axes(ax)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-2.4, 1.6)
        col = AMBER if not filled else NAVY
        if filled:
            ax.add_patch(Polygon(V, closed=True, fc=ICE, ec="none", zorder=2))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ax.plot(V[[a, b], 0], V[[a, b], 1], color=col, lw=3.0, zorder=3)
        ax.plot(V[:, 0], V[:, 1], "o", color=INK, ms=7, zorder=4)

    note(axes[0], (0, -1.25), "a cycle, not a boundary", fs=11, color=INK)
    note(axes[0], (0, -1.75), "counts as a hole", fs=11, color=AMBER,
         weight="bold")
    note(axes[0], (0, -2.22), "∂c = 0,  c ∉ im ∂", fs=10, color=MUTED)

    note(axes[1], (0, -1.25), "a cycle that IS a boundary", fs=11, color=INK)
    note(axes[1], (0, -1.75), "does not count", fs=11, color=MUTED)
    note(axes[1], (0, -2.22), "c = ∂(triangle)", fs=10, color=MUTED)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
