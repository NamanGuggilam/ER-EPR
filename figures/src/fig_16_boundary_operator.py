"""fig 16 — the boundary operator and ∂∂ = 0. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=16, name="boundary_operator", slide=25, type="schematic",
            data_source="none (schematic)",
            caption="The boundary of a triangle is its three oriented edges; "
                    "applying ∂ twice cancels everything.")


def build():
    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.2)

    V = np.array([[1.35, 4.55], [4.05, 4.55], [2.70, 6.90]])
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(V, closed=True, fc=NAVY, ec="none", alpha=0.16,
                         zorder=2))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        arrow(ax, V[a], V[b], color=NAVY, lw=2.0, mutation=15)
    ax.plot(V[:, 0], V[:, 1], "o", color=INK, ms=7, zorder=5)
    for i, (x, y) in enumerate(V):
        note(ax, (x - 0.30, y - 0.02) if i == 0 else
                 (x + 0.30, y - 0.02) if i == 1 else (x, y + 0.38),
             f"v{i}", fs=10.5, color=MUTED)

    note(ax, (7.0, 5.75), "∂[v₀v₁v₂] =", fs=12, color=INK)
    note(ax, (7.0, 5.05), "[v₁v₂] − [v₀v₂] + [v₀v₁]", fs=12, color=NAVY)

    ax.plot([0.7, 9.3], [3.55, 3.55], color=GRID, lw=1.2)

    note(ax, (5.0, 2.85), "∂∂[v₀v₁v₂] = ∂[v₁v₂] − ∂[v₀v₂] + ∂[v₀v₁]",
         fs=11.5, color=INK)
    terms = "([v₂]−[v₁]) − ([v₂]−[v₀]) + ([v₁]−[v₀])"
    note(ax, (5.0, 1.95), terms, fs=11.5, color=MUTED)
    note(ax, (5.0, 1.05), "= 0", fs=15, color=AMBER, weight="bold")
    note(ax, (5.0, 0.38), "every term cancels — a boundary has no boundary",
         fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
