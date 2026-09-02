"""fig 29 — SYK all-to-all 4-body coupling. [SCHEMATIC] square."""
import sys
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=29, name="syk_all_to_all", slide=37, type="schematic",
            data_source="none (schematic); 4 of the C(8,4)=70 quartets drawn",
            caption="SYK: every quartet of Majoranas interacts, with Gaussian "
                    "random strength of variance 6J²/N³.")


def build():
    from matplotlib.patches import Polygon
    N = 8
    th = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    P = np.c_[np.cos(th), np.sin(th)]

    fig, ax = new_fig("square")
    blank_axes(ax)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.85, 1.6)

    for i, j in combinations(range(N), 2):
        ax.plot(P[[i, j], 0], P[[i, j], 1], color=GRID, lw=0.8, zorder=1)

    quartets = [(0, 1, 3, 5), (2, 4, 6, 7), (0, 2, 5, 6), (1, 3, 4, 7)]
    shades = [NAVY, AMBER, TEAL, LIFT]
    for q, col in zip(quartets, shades):
        ax.add_patch(Polygon(P[list(q)], closed=True, fc=col, ec=col, lw=1.4,
                             alpha=0.22, zorder=2))

    ax.plot(P[:, 0], P[:, 1], "o", color=INK, ms=10, zorder=4)
    for i, (x, y) in enumerate(P):
        ax.text(x * 1.20, y * 1.20, f"χ{i + 1}", ha="center", va="center",
                fontsize=10, color=MUTED)

    note(ax, (0, -1.55), "⟨J²ᵢⱼₖₗ⟩ = 6J²/N³", fs=12.5, color=INK)
    note(ax, (0, -1.80), "every one of the C(8,4) = 70 quartets couples",
         fs=10, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
