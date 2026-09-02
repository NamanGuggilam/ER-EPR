"""fig 25 — the Wasserstein matching. [SCHEMATIC] half-slide.

Costs shown are the real L2 costs of the drawn points, and the matching is
the one persim's Hungarian solver actually returns for them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=25, name="wasserstein_matching", slide=34, type="schematic",
            data_source="illustrative diagrams; matching + costs from "
                        "persim.wasserstein (p=1, L2 ground metric)",
            caption="The Wasserstein distance is the cheapest pairing of two "
                    "diagrams, with unmatched points sent to the diagonal.")


def build():
    from persim import wasserstein
    A = np.array([[0.20, 1.35], [0.55, 1.95], [0.95, 2.60]])
    B = np.array([[0.32, 1.20], [0.62, 2.10], [1.55, 1.75]])
    W, match = wasserstein(A, B, matching=True)

    fig, ax = new_fig("half")
    style_axes(ax)
    lim = 3.0
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1.0, zorder=2)
    ax.plot(A[:, 0], A[:, 1], "o", color=NAVY, ms=8, zorder=5, label="diagram X")
    ax.plot(B[:, 0], B[:, 1], "s", color=AMBER, ms=8, zorder=5, label="diagram Y")

    for i, j, c in match:
        i, j = int(i), int(j)
        if i >= 0 and j >= 0:
            ax.plot([A[i, 0], B[j, 0]], [A[i, 1], B[j, 1]], color=MUTED,
                    ls=(0, (3, 3)), lw=1.3, zorder=3)
            mx, my = (A[i, 0] + B[j, 0]) / 2, (A[i, 1] + B[j, 1]) / 2
            ax.text(mx + 0.03, my + 0.06, f"{c:.2f}", fontsize=9, color=MUTED)
        else:
            p = A[i] if j < 0 else B[j]
            m = (p[0] + p[1]) / 2
            ax.plot([p[0], m], [p[1], m], color=CLAY, ls=(0, (2, 2)), lw=1.4,
                    zorder=3)
            ax.text((p[0] + m) / 2 + 0.06, (p[1] + m) / 2, f"{c:.2f}",
                    fontsize=9, color=CLAY)

    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    note(ax, (1.72, 0.62), "unmatched → diagonal\n(cost = persistence/√2)",
         fs=9.5, color=CLAY, ha="center")
    note(ax, (1.72, 2.72), f"W = {W:.3f}", fs=12, color=INK)

    provenance(META["id"],
               f"persim.wasserstein on 3+3 illustrative points: W = {W:.6f}, "
               f"{len(match)} matched pairs (p=1, L2 ground metric)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
