"""fig 66 — no Maldacena-Stanford power law. [DATA] half-slide.

The weighted power-law fit is recomputed here from the same four fitted
cells the battery used, and checked against the battery's reported b.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=66, name="power_law_fit", slide=89, type="data",
            data_source="cfit_out2.txt — the four β=40 fits, refitted here",
            caption="C_fit against N: the exponent is 2.16 ± 2.40 with a "
                    "confidence band spanning orders of magnitude, and the "
                    "points are not even monotone in N.")


def build():
    tab = data.cfit_table()
    txt = (Path(data.RUNG4) / "diagnostics" / "cfit_out2.txt").read_text()
    m = re.search(r"b = ([-\d.]+) \+- ([\d.]+)\s+a = ([\d.]+)", txt)
    chi = re.search(r"chi2/dof = ([\d.]+)/(\d+)", txt)
    if not m:
        raise RuntimeError("could not parse the power-law fit from cfit_out2.txt")
    b_rep, sb_rep, a_rep = float(m.group(1)), float(m.group(2)), float(m.group(3))

    Ns = np.array([N for N in data.N_VALUES if tab[(N, 40.0)]["C1"]])
    C = np.array([tab[(N, 40.0)]["C1"] for N in Ns])
    sC = np.array([tab[(N, 40.0)]["sC1"] for N in Ns])

    lnN, lnC = np.log(Ns), np.log(C)
    w = (C / sC) ** 2
    A = np.vstack([np.ones_like(lnN), lnN]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ (A.T @ W @ lnC)
    resid = lnC - A @ coef
    chi2 = float(resid @ W @ resid)
    dof = len(Ns) - 2
    err = np.sqrt(np.diag(cov) * max(chi2 / dof, 1.0))
    b, sb = float(coef[1]), float(err[1])
    assert abs(b - b_rep) < 0.02, f"refit b={b:.3f} vs reported {b_rep:.3f}"

    fig, ax = new_fig("half")
    xs = np.linspace(11, 19, 60)
    ax.fill_between(xs, np.exp(coef[0]) * xs ** (b - sb),
                    np.exp(coef[0]) * xs ** (b + sb), color=AMBER, alpha=0.16,
                    lw=0, zorder=2, label=f"fit  b = {b:.2f} ± {sb:.2f}")
    ax.plot(xs, np.exp(coef[0]) * xs ** b, color=AMBER, lw=2.0, zorder=4)
    ax.plot(xs, 0.01003 * xs, color=NAVY, ls=(0, (6, 3)), lw=2.0, zorder=5,
            label="Maldacena–Stanford  b = 1, a = 0.01003")
    ax.errorbar(Ns, C, yerr=sC, fmt="o", color=CLAY, ms=8, capsize=3, lw=1.4,
                zorder=6, label="C_fit  (β = 40)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel("C_fit")
    ax.set_xticks([12, 14, 16, 18]); ax.set_xticklabels(["12", "14", "16", "18"])
    ax.legend(loc="lower left", fontsize=8.5)
    ax.text(0.97, 0.95, f"χ²/dof = {chi2:.1f}/{dof}\n"
            f"points not monotone in N", transform=ax.transAxes, ha="right",
            va="top", fontsize=10, color=CLAY, weight="bold")

    provenance(META["id"],
               f"refit of the four β=40 C_fit values {list(np.round(C, 4))} "
               f"at N={list(Ns)}: b={b:.3f}±{sb:.3f} (battery reported "
               f"{b_rep}±{sb_rep}), a={np.exp(coef[0]):.5f} (reported "
               f"{a_rep}), χ²/dof={chi2:.1f}/{dof}; MS predicts b=1, a=0.01003")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
