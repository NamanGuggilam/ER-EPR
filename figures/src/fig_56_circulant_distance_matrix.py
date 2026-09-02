"""fig 56 — the distance matrix is circulant. [DATA] square.

The real 24×24 matrix rung4.circle_distance_matrix builds for one cell.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data
from rung4 import circle_distance_matrix

META = dict(id=56, name="circulant_distance_matrix", slide=79, type="data",
            data_source="rung4.circle_distance_matrix on the cached "
                        "declared-C profile, N=18, β=5",
            caption="The thermal-circle distance matrix is circulant: 576 "
                    "entries carrying only 12 independent numbers.")


def build():
    from matplotlib.colors import LinearSegmentedColormap
    N, beta = 18, 5.0
    prof = data.profiles()["er_G"][(N, beta, "declared")]
    D = circle_distance_matrix(prof, data.N_TAU)
    ghat = data.ghat_profile(prof)

    cmap = LinearSegmentedColormap.from_list("deck", ["#FFFFFF", ICE, LIFT,
                                                      NAVY, INK])
    fig, ax = new_fig("square", figsize=(5.4, 5.2))
    style_axes(ax, grid_axis=None)
    im = ax.imshow(D, cmap=cmap, origin="upper", zorder=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("d = 1/Ĝ")
    cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.8)
    ax.set_xlabel("τ index  j"); ax.set_ylabel("τ index  i")
    ax.set_xticks([0, 6, 12, 18, 23]); ax.set_yticks([0, 6, 12, 18, 23])
    ax.text(0.5, -0.20, "determined by 12 numbers", transform=ax.transAxes,
            ha="center", fontsize=11.5, color=AMBER, weight="bold")
    ax.text(0.5, -0.28, "every entry is d(m), m = min(|i−j|, 24−|i−j|)",
            transform=ax.transAxes, ha="center", fontsize=9.5, color=MUTED)

    provenance(META["id"],
               f"N={N}, β={beta} declared-C profile → 24×24 circulant matrix; "
               f"d(1)={1 / ghat[0]:.6f}, d(8)={1 / ghat[7]:.6f}, "
               f"d(12)={1 / ghat[-1]:.6f}; "
               f"{len(np.unique(np.round(D, 12)))} distinct values incl. 0")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
