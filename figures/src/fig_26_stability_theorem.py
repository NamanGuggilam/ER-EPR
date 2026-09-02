"""fig 26 — stability. [SCHEMATIC] half-slide.

The perturbed barcode is computed: the distance matrix really is perturbed by
ε and re-run through ripser, and the observed bar shift is reported.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import vr

META = dict(id=26, name="stability_theorem", slide=35, type="schematic",
            data_source="12-point ring; ε-perturbed distance matrix re-run "
                        "through ripser (shift measured, not asserted)",
            caption="Stability: perturbing the distance matrix by ε moves "
                    "every bar by at most ε.")


def build():
    from ripser import ripser
    rng = np.random.default_rng(1)
    P = vr.ring(n=12, r=1.0, jitter=0.055, seed=5)
    D = vr.pdist(P)
    eps = 0.12
    pert = rng.uniform(-eps, eps, D.shape)
    pert = np.triu(pert, 1); pert = pert + pert.T
    D2 = np.clip(D + pert, 0, None)
    np.fill_diagonal(D2, 0.0)

    h1a = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1][0]
    h1b = ripser(D2, distance_matrix=True, maxdim=1)["dgms"][1][0]
    shift = float(np.max(np.abs(h1b - h1a)))

    fig, ax = new_fig("half")
    style_axes(ax, grid_axis="x")
    bar_h(ax, 2.0, h1a[0], h1a[1], color=NAVY, lw=13)
    bar_h(ax, 1.0, h1b[0], h1b[1], color=AMBER, lw=13)
    ax.text(h1a[0] - 0.03, 2.0, "original", ha="right", va="center",
            fontsize=10.5, color=NAVY)
    ax.text(h1b[0] - 0.03, 1.0, "perturbed", ha="right", va="center",
            fontsize=10.5, color=AMBER)
    for x0, x1, y in ((h1a[0], h1b[0], 2.0), (h1a[1], h1b[1], 2.0)):
        ax.annotate("", xy=(x1, y - 0.5), xytext=(x0, y - 0.5),
                    arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.2))
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("t")
    ax.set_xlim(0.30, 2.05); ax.set_ylim(0.2, 3.5)
    note(ax, (1.18, 3.05), f"‖D − D′‖∞ = ε = {eps:.2f}", fs=11, color=INK)
    ax.text(1.18, 0.62, f"largest endpoint shift = {shift:.3f}  ≤  ε",
            ha="center", va="center", fontsize=11, color=TEAL, weight="bold",
            bbox=halo())

    provenance(META["id"],
               f"ε = {eps} perturbation of a 12-point ring distance matrix; "
               f"H₁ bar [{h1a[0]:.4f},{h1a[1]:.4f}] → "
               f"[{h1b[0]:.4f},{h1b[1]:.4f}], max shift {shift:.4f} ≤ ε")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
