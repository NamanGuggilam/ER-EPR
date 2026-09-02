"""fig 15 — the simplex ladder. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=15, name="simplex_ladder", slide=23, type="schematic",
            data_source="none (schematic)",
            caption="The building blocks: a k-simplex is the filled hull of "
                    "k+1 points.")


def build():
    fig, axes = new_fig("full", ncols=4, figsize=(11.5, 3.8))
    from matplotlib.patches import Polygon
    names = ["0-simplex\npoint", "1-simplex\nedge", "2-simplex\ntriangle",
             "3-simplex\ntetrahedron"]
    counts = ["1 vertex", "2 vertices", "3 vertices", "4 vertices"]

    pts = [np.array([[0.0, 0.0]]),
           np.array([[-0.7, 0.0], [0.7, 0.0]]),
           np.array([[-0.75, -0.45], [0.75, -0.45], [0.0, 0.85]]),
           np.array([[-0.8, -0.5], [0.8, -0.5], [0.15, 0.35], [-0.05, 1.0]])]
    faces = [[], [(0, 1)], [(0, 1, 2)],
             [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]]

    for ax, P, F, nm, cnt in zip(axes, pts, faces, names, counts):
        blank_axes(ax)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.5, 1.7)
        for f in F:
            if len(f) == 2:
                ax.plot(P[list(f), 0], P[list(f), 1], color=NAVY, lw=2.2,
                        zorder=3)
            else:
                ax.add_patch(Polygon(P[list(f)], closed=True, fc=NAVY,
                                     ec=NAVY, lw=1.8, alpha=0.24, zorder=2))
        if len(P) > 1:
            from itertools import combinations
            for i, j in combinations(range(len(P)), 2):
                ax.plot(P[[i, j], 0], P[[i, j], 1], color=NAVY, lw=1.8,
                        zorder=3)
        ax.plot(P[:, 0], P[:, 1], "o", color=INK, ms=7, zorder=4)
        for v, p in enumerate(P):
            note(ax, (p[0] + 0.16, p[1] + 0.17), f"v{v}", fs=9.5, color=MUTED)
        note(ax, (0, -1.05), nm, fs=11, color=INK)
        note(ax, (0, -1.42), cnt, fs=10, color=MUTED)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
