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

    # Two rows: matrices on top, commentary in its own axes below, so every
    # label is inside a real axes and constrained_layout reserves space for
    # it (text hung below an axes gets clipped at save time).
    fig = plt.figure(figsize=(11.5, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.35])
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    notes_ax = [fig.add_subplot(gs[1, i]) for i in range(2)]

    titles = ["what we did  —  one boundary",
              "what should be done  —  two boundaries"]
    for ax, M, t in zip(axes, (D1, D2), titles):
        style_axes(ax, grid_axis=None)
        ax.imshow(M, cmap=cmap, aspect="equal", zorder=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 1.06, t, transform=ax.transAxes, ha="center",
                fontsize=12.5, color=INK, weight="bold")

    axes[1].axhline(11.5, color=AMBER, lw=2.2)
    axes[1].axvline(11.5, color=AMBER, lw=2.2)
    # labels in block corners, clear of the bright diagonal
    for xy, lab in (((0.06, 0.94), "L–L"), ((0.94, 0.06), "R–R"),
                    ((0.94, 0.94), "L–R"), ((0.06, 0.06), "L–R")):
        axes[1].text(xy[0], xy[1], lab, transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=11, color=INK,
                     weight="bold", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.22", fc="white",
                               ec="none", alpha=0.88))

    notes_l = [("circulant — depends only on separation", MUTED, "normal"),
               ("filtration forced through C₂₄(1..k)", MUTED, "normal"),
               ("theorem applies ⇒ one H₁ bar, two scalars", CLAY, "normal"),
               ("H₁ cannot carry information", CLAY, "bold"),
               ("(real matrix, N = 18, β = 5)", MUTED, "italic")]
    notes_r = [("block structure — L–L, R–R and L–R", TEAL, "normal"),
               ("not circulant: no forced nesting", TEAL, "normal"),
               ("theorem does NOT apply", TEAL, "normal"),
               ("H₁ could carry information", TEAL, "bold"),
               ("(proposed — not yet run; matrix is illustrative)",
                MUTED, "italic")]
    for ax, ns in zip(notes_ax, (notes_l, notes_r)):
        blank_axes(ax); ax.set_aspect("auto")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for i, (s, col, wt) in enumerate(ns):
            ax.text(0.5, 0.92 - 0.21 * i, s, ha="center", va="center",
                    fontsize=10 if wt != "italic" else 9, color=col,
                    weight="bold" if wt == "bold" else "normal",
                    style="italic" if wt == "italic" else "normal")

    provenance(META["id"],
               f"LEFT: real 24×24 circulant matrix from profiles_cache.pkl "
               f"(N=18, β=5, declared C), {len(np.unique(np.round(D1, 12)))} "
               f"distinct values. RIGHT: illustrative {D2.shape[0]}×"
               f"{D2.shape[1]} block matrix for the proposed two-boundary "
               f"construction — NOT computed physics, labelled as such")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
