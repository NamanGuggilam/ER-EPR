"""fig 75 — one boundary vs two. [SCHEMATIC + DATA] full-width.

The most important figure in Part IX. LEFT is the real circulant matrix this
work used (from the cache). RIGHT is the PROPOSED two-boundary construction:
that experiment has not been run, so its matrix is an illustration of the
block structure, explicitly labelled as a proposal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data
from rung4 import circle_distance_matrix

META = dict(id=75, name="one_vs_two_boundary", slide=95, type="data",
            data_source="LEFT: real circulant matrix (profiles_cache.pkl, "
                        "N=18 β=5). RIGHT: illustrative block matrix for the "
                        "proposed two-boundary construction (not yet run).",
            caption="One boundary gives a circulant matrix and a theorem-"
                    "determined barcode; two boundaries give a block matrix "
                    "the theorem does not cover, where H₁ could carry "
                    "information.")


def build():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("deck", ["#FFFFFF", ICE, LIFT,
                                                      NAVY, INK])

    prof = data.profiles()["er_G"][(18, 5.0, "declared")]
    D1 = circle_distance_matrix(prof, data.N_TAU)

    # Illustrative two-boundary block matrix: two thermal circles, each
    # circulant within itself, coupled across by a different (weaker,
    # non-circulant) law. NOT a computed physical correlator.
    n = 12
    g = data.ghat_profile(prof)[:n // 2 + 1]
    idx = np.arange(n)
    m = np.abs(idx[:, None] - idx[None, :])
    m = np.minimum(m, n - m)
    blk = np.zeros((n, n))
    nz = m > 0
    blk[nz] = 1.0 / g[np.clip(m[nz] - 1, 0, len(g) - 1)]
    cross = 0.55 + 0.45 * np.abs(np.subtract.outer(
        np.linspace(-1, 1, n), np.linspace(1, -1, n))) / 2
    D2 = np.block([[blk, cross], [cross.T, blk]])

    fig, axes = new_fig("full", ncols=2, figsize=(11.5, 5.0))
    titles = ["what we did  —  one boundary",
              "what should be done  —  two boundaries"]
    mats = [D1, D2]
    for ax, M, t in zip(axes, mats, titles):
        style_axes(ax, grid_axis=None)
        ax.imshow(M, cmap=cmap, aspect="equal", zorder=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 1.14, t, transform=ax.transAxes, ha="center",
                fontsize=12, color=INK, weight="bold")

    axes[1].axhline(11.5, color=AMBER, lw=2.0)
    axes[1].axvline(11.5, color=AMBER, lw=2.0)
    for xy, lab in (((0.25, 0.75), "L–L"), ((0.75, 0.25), "R–R"),
                    ((0.75, 0.75), "L–R"), ((0.25, 0.25), "L–R")):
        axes[1].text(xy[0], xy[1], lab, transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=11, color="white",
                     weight="bold", zorder=5)

    notes_l = ["circulant — depends only on separation",
               "filtration forced through C₂₄(1..k)",
               "theorem applies ⇒ one H₁ bar, two scalars",
               "H₁ cannot carry information"]
    notes_r = ["block structure — L–L, R–R, and L–R",
               "not circulant: no forced nesting",
               "theorem does NOT apply",
               "H₁ could carry information"]
    for ax, ns, col in ((axes[0], notes_l, MUTED), (axes[1], notes_r, TEAL)):
        for i, s in enumerate(ns):
            ax.text(0.5, -0.10 - 0.085 * i, s, transform=ax.transAxes,
                    ha="center", fontsize=10,
                    color=col if i < 2 else (CLAY if col is MUTED else TEAL),
                    weight="bold" if i == 3 else "normal")
    axes[1].text(0.5, -0.46, "(proposed construction — not yet run; matrix "
                 "shown is illustrative)", transform=axes[1].transAxes,
                 ha="center", fontsize=9, color=MUTED, style="italic")
    axes[0].text(0.5, -0.46, f"(real matrix, N=18, β=5)",
                 transform=axes[0].transAxes, ha="center", fontsize=9,
                 color=MUTED, style="italic")

    provenance(META["id"],
               f"LEFT: real 24×24 circulant matrix from profiles_cache.pkl "
               f"(N=18, β=5, declared C), {len(np.unique(np.round(D1, 12)))} "
               f"distinct values. RIGHT: illustrative {D2.shape[0]}×"
               f"{D2.shape[1]} block matrix for the proposed two-boundary "
               f"construction — NOT computed physics, labelled as such")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
