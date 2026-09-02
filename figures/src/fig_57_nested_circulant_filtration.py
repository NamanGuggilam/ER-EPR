"""fig 57 — the filtration IS the circulant sequence. [SCHEMATIC + DATA]
full-width.

Graphs are the real circulant graphs C₂₄(1..k); thresholds printed beneath
are the actual 1/Ĝ(k) of a real cell. This is the structure theorem, drawn.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=57, name="nested_circulant_filtration", slide=80, type="data",
            data_source="thresholds 1/Ĝ(k) from profiles_cache.pkl "
                        "(N=18, β=5, declared C); graphs are C₂₄(1..k)",
            caption="Raising the threshold walks through the circulant graphs "
                    "C₂₄(1..k) and nothing else; at k = 8 the loop dies.")


def build():
    N, beta = 18, 5.0
    prof = data.profiles()["er_G"][(N, beta, "declared")]
    ghat = data.ghat_profile(prof)
    thr = 1.0 / ghat

    ks = [1, 4, 7, 8, 9]
    n = data.N_TAU
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    P = np.c_[np.cos(th), np.sin(th)]

    fig, axes = new_fig("full", ncols=5, figsize=(11.5, 3.9))
    for ax, k in zip(axes, ks):
        blank_axes(ax)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.95, 1.35)
        boundary = (k == 8)
        col = AMBER if boundary else NAVY
        for i in range(n):
            for step in range(1, k + 1):
                j = (i + step) % n
                ax.plot(P[[i, j], 0], P[[i, j], 1], color=col,
                        lw=0.55 if k > 4 else 0.9, alpha=0.55, zorder=2)
        ax.plot(P[:, 0], P[:, 1], "o", color=INK, ms=3.4, zorder=4)
        ax.text(0, 1.22, f"k = {k}", ha="center", fontsize=11,
                color=col, weight="bold" if boundary else "normal")
        ax.text(0, -1.42, f"t = 1/Ĝ({k}) = {thr[k - 1]:.3f}", ha="center",
                fontsize=9.5, color=MUTED)
        ax.text(0, -1.75, "H₁ = ℝ" if k < 8 else "H₁ = 0", ha="center",
                fontsize=10.5, color=TEAL if k < 8 else CLAY,
                weight="bold" if boundary else "normal")

    axes[3].annotate("k/n = 1/3\nthe loop dies", xy=(0, 1.05),
                     xytext=(0, 1.95), ha="center", fontsize=10.5,
                     color=AMBER, weight="bold",
                     arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.5))
    for ax in axes:
        ax.set_ylim(-1.95, 2.25)

    provenance(META["id"],
               f"N={N}, β={beta}, declared C: thresholds 1/Ĝ(k) = "
               f"{', '.join(f'k={k}:{thr[k - 1]:.4f}' for k in ks)}; "
               f"filtration passes only through C₂₄(1..k)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
