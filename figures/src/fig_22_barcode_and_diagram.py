"""fig 22 — barcode and persistence diagram. [SCHEMATIC] full-width.

The two panels show the same real diagram of a 12-point ring, drawn the two
standard ways, with two features linked by guide lines.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import vr

META = dict(id=22, name="barcode_and_diagram", slide=31, type="schematic",
            data_source="12-point ring; ripser H₀/H₁ (same data both panels)",
            caption="Barcode and persistence diagram are two renderings of "
                    "one object: a bar [b, d) is the point (b, d).")


def build():
    from ripser import ripser
    P = vr.ring(n=12, r=1.0, jitter=0.055, seed=5)
    dgms = ripser(vr.pdist(P), distance_matrix=True, maxdim=1)["dgms"]
    H0 = dgms[0][np.isfinite(dgms[0]).all(axis=1)]
    H1 = dgms[1]
    H0 = H0[np.argsort(H0[:, 1])][-6:]        # six longest H0 bars

    fig, (axl, axr) = new_fig("full", ncols=2)

    style_axes(axl, grid_axis="x")
    for i, (b, d) in enumerate(H0):
        bar_h(axl, i, b, d, color=NAVY, lw=5)
    for j, (b, d) in enumerate(H1):
        bar_h(axl, len(H0) + j + 0.4, b, d, color=AMBER, lw=5)
    axl.set_yticks([])
    axl.spines["left"].set_visible(False)
    axl.set_xlabel("t")
    axl.set_ylim(-0.9, len(H0) + len(H1) + 0.6)
    note(axl, (0.02, len(H0) - 0.4), "H₀", color=NAVY, fs=11, ha="left",
         weight="bold")
    note(axl, (0.02, len(H0) + 0.9), "H₁", color=AMBER, fs=11, ha="left",
         weight="bold")

    style_axes(axr)
    lim = float(max(H1[:, 1].max(), H0[:, 1].max())) * 1.15
    axr.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1.0, zorder=2)
    axr.plot(H0[:, 0], H0[:, 1], "o", color=NAVY, zorder=4, label="H₀")
    axr.plot(H1[:, 0], H1[:, 1], "o", color=AMBER, zorder=5, label="H₁")
    axr.set_xlabel("birth"); axr.set_ylabel("death")
    axr.set_xlim(-0.05, lim); axr.set_ylim(0, lim)
    axr.legend(loc="lower right")
    note(axr, (lim * 0.62, lim * 0.55), "diagonal:\nzero persistence",
         color=MUTED, fs=9.5)

    # guide lines linking the same two features across panels
    for (b, d), y, col in ((H1[0], len(H0) + 0.4, AMBER),
                           (H0[-1], len(H0) - 1, NAVY)):
        con = plt.matplotlib.patches.ConnectionPatch(
            xyA=(d, y), coordsA=axl.transData,
            xyB=(b, d), coordsB=axr.transData,
            color=col, lw=0.9, alpha=0.5, linestyle=(0, (3, 3)))
        fig.add_artist(con)

    provenance(META["id"],
               f"12-point ring: {len(H0)} longest H₀ bars + {len(H1)} H₁ bar "
               f"[{H1[0,0]:.4f}, {H1[0,1]:.4f}] shown in both renderings")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
