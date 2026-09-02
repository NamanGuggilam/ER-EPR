"""fig 64 — C_fit depends on temperature. [DATA] half-slide.

C = √2·α_S·N carries no temperature dependence, so any slope at all is a
failure of the dictionary.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=64, name="cfit_beta_dependence", slide=87, type="data",
            data_source="cfit_out2.txt, Ĝ(8) target (the only target with "
                        "fits at two temperatures)",
            caption="The fitted C moves with temperature by 29–158% of its "
                    "mean, though the declared dictionary makes it "
                    "temperature-independent.")


def build():
    tab = data.cfit_table()
    cols = [NAVY, LIFT, TEAL, AMBER]

    fig, ax = new_fig("half")
    spreads = {}
    for N, col in zip(data.N_VALUES, cols):
        pts = [(b, tab[(N, b)]["C8"]) for b in data.BETAS
               if tab[(N, b)]["C8"] is not None]
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        spread = (max(ys) - min(ys)) / np.mean(ys) * 100
        spreads[N] = spread
        ax.plot(xs, ys, "o-", color=col, ms=7, zorder=4,
                label=f"N = {N}   ({spread:.0f}% spread)")
        ax.axhline(data.declared_C(N), color=col, ls=(0, (2, 3)), lw=1.0,
                   alpha=0.7, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("β")
    ax.set_ylabel("C_fit   (target Ĝ(8))")
    ax.set_xticks([20, 40]); ax.set_xticklabels(["20", "40"])
    ax.legend(loc="upper left", fontsize=9.5)
    # the band between the N=12 curve and the N=16/18 curves is the only
    # empty region; putting this at the bottom laid it across the dotted
    # C(N) lines and the N=14/16 curves
    ax.text(0.5, 0.53,
            "dotted lines: declared C(N)\n"
            "a dictionary C cannot depend on β at all",
            transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
            color=CLAY, weight="bold")

    provenance(META["id"],
               "C_fit(Ĝ8) spread across β at fixed N: " +
               ", ".join(f"N={N}: {spreads[N]:.1f}%" for N in spreads) +
               "; values " + str({N: [tab[(N, b)]["C8"] for b in data.BETAS
                                      if tab[(N, b)]["C8"] is not None]
                                  for N in spreads}))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
