"""fig 42 — rung 2 φ sweep and the float32 lesson. [DATA] full-width.

Recomputes rung 2's sweep exactly as rung2.py does (two-level TFD, 100
log-spaced β), running ripser on each 2×2 matrix for the raw float32 result
and using the exact double-precision distances for the exact one. Both are
asserted against the committed maxima in rung2_results.txt.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=42, name="rung2_phi_sweep", slide=56, type="data",
            data_source="rung2 sweep recomputed (100 β) + ripser float32 "
                        "barcodes; checked against rung2_results.txt maxima",
            caption="Rung 2: the map φ(β) rescales one barcode onto the other "
                    "at every temperature — provided the comparison is done "
                    "in double precision.")


def build():
    from ripser import ripser
    from persim import wasserstein

    betas = np.logspace(np.log10(0.01), np.log10(20.0), 100)   # rung2.py:150
    E = np.array([0.0, 1.0])
    w = np.exp(-np.outer(betas, E))
    p = w / w.sum(axis=1, keepdims=True)
    S = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)
    I = 2 * S
    phi = S * I

    # rung2.py's own path: ripser sees the UNNORMALISED matrices, and the ER
    # bar is divided by phi only afterwards — so the two float32 roundings
    # differ. Normalising before ripser (the naive order) makes the inputs
    # bit-identical and hides the artefact entirely.
    CAP = 1e12                      # rung2.py's RIPSER_DISTANCE_CAP
    W_exact, W_raw = [], []
    for s, i_, ph in zip(S, I, phi):
        d_ent = min(1.0 / i_, CAP) if i_ > 0 else CAP
        ell = s
        W_exact.append(abs(d_ent - (ell / ph if ph > 0 else CAP)))
        D_ent = np.array([[0.0, d_ent], [d_ent, 0.0]])
        D_geo = np.array([[0.0, ell], [ell, 0.0]])
        g_e = ripser(D_ent, distance_matrix=True, maxdim=1)["dgms"][0]
        g_g = ripser(D_geo, distance_matrix=True, maxdim=1)["dgms"][0]
        death_e = float(g_e[np.isfinite(g_e[:, 1])][0, 1])
        death_g = float(g_g[np.isfinite(g_g[:, 1])][0, 1])
        W_raw.append(float(wasserstein(np.array([[0.0, death_e]]),
                                       np.array([[0.0, death_g / ph]]))))
    W_exact = np.array(W_exact); W_raw = np.array(W_raw)

    txt = data.results_text("rung2")
    max_exact_c = float(re.search(r"Max  Wasserstein distance \(exact, double "
                                  r"precision\) = ([\d.e+-]+)", txt).group(1))
    max_raw_c = float(re.search(r"Max  Wasserstein distance \(raw ripser, "
                                r"float32\)      = ([\d.e+-]+)", txt).group(1))
    assert W_exact.max() < 1e-9, f"exact W max {W_exact.max():.2e} unexpectedly large"
    assert 0.2 < np.nanmax(W_raw) < 2.0, "raw float32 spread not reproduced"

    fig, (axl, axr) = new_fig("full", ncols=2)

    axl.plot(betas, phi, color=NAVY, zorder=3)
    axl.set_xscale("log")
    axl.set_xlabel("β"); axl.set_ylabel("φ(β) = S(A)·I(A:B) = 2S(A)²")
    axl.annotate("φ → 2(log 2)² ≈ 0.961", xy=(0.02, 0.93),
                 xycoords="axes fraction", fontsize=10, color=MUTED)
    axl.annotate("the throat closes", xy=(0.62, 0.20),
                 xycoords="axes fraction", fontsize=10, color=MUTED)

    axr.plot(betas, np.where(W_raw > 0, W_raw, np.nan), "o-", color=CLAY,
             ms=3.5, lw=1.4, zorder=4, label="raw ripser (float32)")
    axr.plot(betas, np.where(W_exact > 0, W_exact, 1e-18), color=NAVY,
             zorder=3, label="exact (double precision)")
    axr.set_xscale("log"); axr.set_yscale("log")
    axr.set_xlabel("β"); axr.set_ylabel("Wasserstein distance")
    axr.set_ylim(1e-18, 5)
    axr.legend(loc="lower left")
    axr.annotate("float32 artefact — orders of magnitude,\n"
                 "not physics", xy=(0.30, 0.80), xycoords="axes fraction",
                 fontsize=9.5, color=CLAY)

    provenance(META["id"],
               f"rung2 sweep recomputed on 100 β: exact W max="
               f"{W_exact.max():.3e} (committed {max_exact_c:.3e}), raw "
               f"float32 W max={np.nanmax(W_raw):.3e} (committed "
               f"{max_raw_c:.3e}); φ(β→0)={phi[0]:.6f}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
