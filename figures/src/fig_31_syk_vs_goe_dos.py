"""fig 31 — SYK vs a bandwidth-matched GOE control. [DATA] full-width.

The GOE control is built by rung4.build_goe_control — the SAME code path the
extraction gate used, with the same seed convention — so this figure shows
the comparison the gate actually made.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data
import rung4

META = dict(id=31, name="syk_vs_goe_dos", slide=40, type="data",
            data_source="rung4/data/spectra_N18_R20.npz + "
                        "rung4.build_goe_control(18, R, spectra) — the "
                        "extraction gate's own control path",
            caption="At N = 18 the SYK spectral edge is indistinguishable "
                    "from a bandwidth-matched GOE ensemble: no Schwarzian "
                    "edge at accessible N.")


def build():
    N = 18
    syk, fname = data.spectra(N)
    R = syk.shape[0]
    goe = rung4.build_goe_control(N, R, syk)

    def edge(sp):
        return (sp - sp[:, :1]).ravel()

    e_syk, e_goe = edge(syk), edge(goe)
    cut = float(np.quantile(e_syk, 0.45))

    fig, axes = new_fig("full", ncols=2, figsize=(11.5, 4.5))

    ax = axes[0]
    bins = np.linspace(0, cut, 55)
    ax.hist(e_syk, bins=bins, color=AMBER, alpha=0.85, density=True,
            label="SYK", zorder=3)
    ax.hist(e_goe, bins=bins, histtype="step", color=MUTED, ls="--", lw=2.0,
            density=True, label="GOE (bandwidth-matched)", zorder=4)
    ax.set_xlabel("ε = E − E₀   (spectral edge)")
    ax.set_ylabel("density of states")
    ax.legend(loc="upper left")
    ax.set_xlim(0, cut)

    ax = axes[1]
    q = np.linspace(0.001, 0.45, 300)
    ax.plot(np.quantile(e_syk, q), np.quantile(e_goe, q), color=NAVY, zorder=3)
    lo, hi = 0, max(np.quantile(e_syk, 0.45), np.quantile(e_goe, 0.45))
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1.2, zorder=2)
    ax.set_xlabel("SYK edge quantile")
    ax.set_ylabel("GOE edge quantile")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.text(0.05, 0.90, "quantile–quantile:\npoints on the diagonal means\n"
                        "the two edges have the same shape",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")

    axes[0].annotate("no Schwarzian edge at accessible N",
                     xy=(0.52, 0.55), xycoords="axes fraction",
                     fontsize=11, color=AMBER, weight="bold", ha="left")

    ks = float(np.max(np.abs(
        np.searchsorted(np.sort(e_syk), np.sort(e_goe)) / len(e_syk)
        - np.arange(len(e_goe)) / len(e_goe))))
    provenance(META["id"],
               f"SYK N={N} from {fname} (R={R}, dim={syk.shape[1]}) vs "
               f"rung4.build_goe_control(N={N}, R={R}) — same path as the "
               f"extraction gate; edge region ε ≤ {cut:.4f} (45th pct); "
               f"KS-style max quantile gap = {ks:.4f}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
