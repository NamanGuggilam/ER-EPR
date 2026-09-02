"""fig 62 — where SYK crosses the conformal value. [DATA] half-slide.

Error bars are the realisation spread and are deliberately visible: the
β = 40 points are noisy and that should not be hidden.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=62, name="crossing_per_N", slide=84, type="data",
            data_source="profiles_cache.pkl — Ĝ_SYK(1) mean ± std per cell",
            caption="SYK's Ĝ(1) crosses the conformal value between β = 20 "
                    "and β = 40 at every N; the β = 40 points carry large "
                    "realisation spread.")


def build():
    syk = data.ghat_syk()
    cols = [NAVY, LIFT, TEAL, AMBER]

    fig, ax = new_fig("half")
    crossings = {}
    for N, col in zip(data.N_VALUES, cols):
        y = np.array([syk[(N, b)][0] for b in data.BETAS])
        e = np.array([syk[(N, b)][1] for b in data.BETAS])
        ax.errorbar(data.BETAS, y, yerr=e, fmt="o-", color=col, ms=6,
                    capsize=3, lw=1.8, zorder=4, label=f"N = {N}")
        for i in range(3):
            if (y[i] - data.CONF_G1) * (y[i + 1] - data.CONF_G1) < 0:
                lb1, lb2 = np.log(data.BETAS[i]), np.log(data.BETAS[i + 1])
                t = (data.CONF_G1 - y[i]) / (y[i + 1] - y[i])
                crossings[N] = float(np.exp(lb1 + t * (lb2 - lb1)))
    ax.axhline(data.CONF_G1, color=CLAY, ls=(0, (6, 3)), lw=1.8, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("β")
    ax.set_ylabel("Ĝ_SYK(1)")
    ax.set_xticks(data.BETAS)
    ax.set_xticklabels([f"{b:g}" for b in data.BETAS])
    ax.legend(loc="upper left", fontsize=9.5)
    ax.text(5.2, data.CONF_G1 + 0.13, f"conformal {data.CONF_G1:.4f}",
            fontsize=10, color=CLAY)
    ax.text(0.97, 0.06,
            "crossings:  " + ",  ".join(f"N={N}: β≈{crossings[N]:.1f}"
                                        for N in data.N_VALUES),
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)

    provenance(META["id"],
               "interpolated crossings of Ĝ(1)=2.767905: " +
               ", ".join(f"N={N}: β≈{crossings[N]:.1f}" for N in crossings) +
               "; β=40 spreads " +
               ", ".join(f"N={N}: ±{syk[(N, 40.0)][1]:.3f}"
                         for N in data.N_VALUES))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
