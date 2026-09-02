"""fig 32 — the conformal profile. [DATA: analytic] half-slide.

Ĝ(m) = [sin(πm/24)]^(−2Δ) / [sin(π/2)]^(−2Δ), Δ = 1/4 — the C → ∞ shape,
computed. Ĝ(1) and Ĝ(8) are singled out because they turn out to carry the
entire rung-4 comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=32, name="conformal_profile", slide=41, type="data",
            data_source="analytic conformal shape [sin(πm/24)]^(−1/2), computed",
            caption="The conformal (C → ∞) correlation profile on the "
                    "24-point thermal circle; Ĝ(1) and Ĝ(8) are the two "
                    "values the whole comparison reduces to.")


def build():
    m = np.arange(1, 13)
    G = np.sin(np.pi * m / data.N_TAU) ** (-2 * data.DELTA)
    Ghat = G / G[-1]
    g1, g8 = float(Ghat[0]), float(Ghat[7])

    fig, ax = new_fig("half")
    ax.plot(m, Ghat, "-o", color=NAVY, zorder=3)
    for mm, gg, lab in ((1, g1, f"Ĝ(1) = {g1:.4f}"),
                        (8, g8, f"Ĝ(8) = {g8:.4f}")):
        ax.plot([mm], [gg], "o", color=AMBER, ms=10, zorder=5)
        ax.annotate(lab, xy=(mm, gg),
                    xytext=(mm + 1.4, gg + (0.45 if mm == 1 else 0.35)),
                    fontsize=10.5, color=AMBER,
                    arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.4))
    ax.set_xlabel("separation  m   (τ = mβ/24)")
    ax.set_ylabel("Ĝ(m) = G(m)/G(β/2)")
    ax.set_xticks(m)
    ax.set_ylim(0.9, 3.3)

    assert abs(g1 - data.CONF_G1) < 1e-12 and abs(g8 - data.CONF_G8) < 1e-12
    provenance(META["id"],
               f"analytic conformal profile, Δ={data.DELTA}, n_tau="
               f"{data.N_TAU}: Ĝ(1)={g1:.6f}, Ĝ(8)={g8:.6f} "
               f"(matches data.CONF_G1/CONF_G8)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
