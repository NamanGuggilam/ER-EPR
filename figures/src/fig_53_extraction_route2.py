"""fig 53 — extraction Route 2: the entropy slope goes negative. [DATA]
half-slide.

Recomputed from the cached spectra through rung4's own entropy code path,
and cross-checked against the C values the committed extract run printed.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data
import rung4

META = dict(id=53, name="extraction_route2", slide=75, type="data",
            data_source="cached spectra through rung4.entropy_curves / "
                        "fit_entropy_window; checked against rung4_results.txt",
            caption="Route 2: regressing the Schwarzian entropy term gives a "
                    "negative C at every N — an unphysical stiffness, "
                    "reported as data.")


def build():
    txt = data.results_text("rung4")
    committed = {}
    for N in data.N_VALUES:
        # first match is the SYK block; the GOE control block follows
        m = re.search(rf"N={N:>2}  Route2 C=([-+]?\d+\.\d+)\+-", txt)
        if not m:
            raise RuntimeError(
                f"could not parse the committed Route-2 C for N={N} from "
                f"rung4_results.txt; regenerate with "
                f"'.venv/bin/python rung4/rung4.py extract'")
        committed[N] = float(m.group(1))

    betas = np.geomspace(1.0, 40.0, 48)
    lo, hi = rung4.BETA_HEADLINE
    fig, ax = new_fig("half")
    cols = [NAVY, LIFT, TEAL, AMBER]
    fitted = {}
    for N, col in zip(data.N_VALUES, cols):
        sp, _ = data.spectra(N)
        S = rung4.entropy_curves(sp, betas)
        Sm = S.mean(axis=0)
        y = Sm + 1.5 * np.log(betas)
        C_fit = rung4.fit_entropy_window(Sm, betas, rung4.BETA_HEADLINE)
        fitted[N] = C_fit
        assert abs(C_fit - committed[N]) < 5e-4, (
            f"recomputed Route-2 C for N={N} is {C_fit:.4f} but the committed "
            f"run says {committed[N]:.4f}")
        m = (betas >= lo) & (betas <= hi)
        x = 1.0 / betas
        ax.plot(x, y - y[m][0], color=col, lw=1.4, alpha=0.55, zorder=3)
        ax.plot(x[m], y[m] - y[m][0], "o", color=col, ms=4, zorder=4,
                label=f"N = {N}   C = {C_fit:+.4f}")
        p = np.polyfit(x[m], y[m] - y[m][0], 1)
        ax.plot(x[m], np.polyval(p, x[m]), color=col, lw=2.4, zorder=5)

    ax.set_xlabel("T = 1/β")
    ax.set_ylabel("S + (3/2) ln β   (offset)")
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.97, 0.94, "slope = 4π²C\nfitted slope is negative at every N",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=CLAY, weight="bold")
    ax.axvspan(1 / hi, 1 / lo, color=ICE, alpha=0.35, zorder=1)
    ax.text((1 / hi + 1 / lo) / 2, ax.get_ylim()[0] * 0.92,
            f"fit window β ∈ [{lo:g}, {hi:g}]", ha="center", fontsize=9,
            color=MUTED)

    provenance(META["id"],
               f"recomputed Route 2 from cached spectra: C = "
               f"{ {N: round(fitted[N], 4) for N in fitted} }, matching the "
               f"committed values { {N: committed[N] for N in committed} } "
               f"to <5e-4; all negative")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
