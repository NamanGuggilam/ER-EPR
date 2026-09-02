"""fig 20 — the Vietoris-Rips clique rule. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=20, name="vr_clique_rule", slide=29, type="schematic",
            data_source="none (schematic); edge lengths compared to a stated t",
            caption="The Vietoris–Rips rule: fill a simplex exactly when every "
                    "pair inside it is within t.")


def build():
    from matplotlib.patches import Polygon
    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.6)

    A = np.array([[1.05, 3.6], [3.35, 3.6], [2.2, 5.6]])
    ax.add_patch(Polygon(A, closed=True, fc=NAVY, ec="none", alpha=0.22,
                         zorder=2))
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ax.plot(A[[a, b], 0], A[[a, b], 1], color=NAVY, lw=2.2, zorder=3)
    ax.plot(A[:, 0], A[:, 1], "o", color=INK, ms=7, zorder=4)
    note(ax, (2.2, 2.75), "all three pairs ≤ t", fs=10.5, color=TEAL)
    note(ax, (2.2, 2.15), "⇒ filled", fs=11.5, color=TEAL, weight="bold")

    B = np.array([[6.0, 3.6], [8.3, 3.6], [8.9, 6.35]])
    for a, b in ((0, 1), (1, 2), (2, 0)):
        far = (a, b) == (2, 0)
        ax.plot(B[[a, b], 0], B[[a, b], 1], color=CLAY if far else NAVY,
                lw=2.2, ls="--" if far else "-", zorder=3)
    ax.plot(B[:, 0], B[:, 1], "o", color=INK, ms=7, zorder=4)
    note(ax, (6.15, 5.15), "> t", fs=10.5, color=CLAY, ha="center")
    note(ax, (7.45, 2.75), "one pair too far", fs=10.5, color=CLAY)
    note(ax, (7.45, 2.15), "⇒ stays hollow", fs=11.5, color=CLAY, weight="bold")

    box(ax, (0.7, 0.35), 8.6, 1.05,
        "every pair within t  ⇔  include the simplex",
        fc=CARD, ec=NAVY, fs=11.5)
    note(ax, (5.0, 6.95), "one threshold t, applied to every pair",
         fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
