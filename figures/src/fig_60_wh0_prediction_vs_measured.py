"""fig 60 — closed form vs measurement for W_H₀. [DATA] half-slide.

Measured W_H₀ (committed split table) against the closed form
23·|Δ(1/Ĝ(1))| computed from the cached profiles. The strongest validation
figure in the paper: a prediction, not a fit.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=60, name="wh0_prediction_vs_measured", slide=83, type="data",
            data_source="split_out.txt (measured W_H₀) vs 23·|Δ(1/Ĝ(1))| "
                        "from profiles_cache.pkl — all 16 cells × 4 tags",
            caption="Measured W_H₀ against its closed form: 64 cells on the "
                    "identity line, confirming the barcode reduces to Ĝ(1).")


def build():
    split = data.split_table()
    pr = data.profiles()
    tags = ["declared", "x5", "x0.2", "conformal"]
    tag_col = {"declared": NAVY, "x5": MUTED, "x0.2": MUTED,
               "conformal": ICE}

    # The measured entry is the MEAN over realisations of a per-realisation
    # Wasserstein. Because |·| is nonlinear, the prediction must be averaged
    # the same way: mean_r 23·|f_r − f_ER|, NOT 23·|mean_r f_r − f_ER|.
    pred, meas, cols = [], [], []
    for N in data.N_VALUES:
        for bi, beta in enumerate(data.BETAS):
            P = pr["syk_G"][N][:, bi, :]
            f_syk = P[:, -1] / P[:, 0]                   # 1/Ĝ(1) per realisation
            for t in tags:
                g = data.ghat_profile(pr["er_G"][(N, beta, t)])
                pred.append(float(np.mean(23.0 * np.abs(f_syk - 1.0 / g[0]))))
                meas.append(split[(N, beta)]["W0"][t])
                cols.append(tag_col[t])
    pred, meas = np.array(pred), np.array(meas)
    resid = np.abs(pred - meas)

    fig, ax = new_fig("half")
    lim = max(pred.max(), meas.max()) * 1.06
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1.3, zorder=2)
    for c in set(cols):
        m = [i for i, cc in enumerate(cols) if cc == c]
        ax.plot(pred[m], meas[m], "o", color=c, ms=6, alpha=0.85, zorder=4)
    ax.set_xlabel("closed form   23 · |Δ(1/Ĝ(1))|")
    ax.set_ylabel("measured  W_H₀")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.text(0.04, 0.90, f"{len(pred)} cells × tags\n"
            f"max |difference| = {resid.max():.1e}\n"
            f"(the committed table is printed to 4 dp,\n"
            f"so ≤ 5e−5 is exact to recorded precision)",
            transform=ax.transAxes, fontsize=9.5, color=TEAL, va="top",
            weight="bold")
    ax.text(0.96, 0.06, "y = x", transform=ax.transAxes, ha="right",
            fontsize=10, color=MUTED)

    provenance(META["id"],
               f"{len(pred)} (cell, tag) points: max |measured − 23·"
               f"|Δ(1/Ĝ(1))|| = {resid.max():.3e}, mean {resid.mean():.3e}; "
               f"measured range [{meas.min():.4f}, {meas.max():.4f}]")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
