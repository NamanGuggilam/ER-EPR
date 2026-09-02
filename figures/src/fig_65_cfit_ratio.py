"""fig 65 — one C cannot fit two separations. [DATA] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=65, name="cfit_ratio", slide=88, type="data",
            data_source="cfit_out2.txt — C_fit(Ĝ1)/C_fit(Ĝ8) at β = 40",
            caption="Fitting the nearest-neighbour scale and the β/3 scale "
                    "demands different couplings — a shape failure, "
                    "independent of the value of C.")


def build():
    tab = data.cfit_table()
    Ns, ratios = [], []
    for N in data.N_VALUES:
        r = tab[(N, 40.0)]
        if r["C1"] and r["C8"]:
            Ns.append(N); ratios.append(r["C1"] / r["C8"])

    fig, ax = new_fig("half")
    x = np.arange(len(Ns))
    ax.bar(x, ratios, color=AMBER, width=0.55, zorder=3)
    ax.axhline(1.0, color=MUTED, ls="--", lw=1.6, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f"N = {n}" for n in Ns])
    ax.set_ylabel("C_fit(Ĝ(1))  /  C_fit(Ĝ(8))")
    ax.set_ylim(0, max(ratios) * 1.22)
    for xi, r in zip(x, ratios):
        ax.text(xi, r + max(ratios) * 0.03, f"{r:.3f}", ha="center",
                fontsize=10.5, color=INK)
    ax.text(0.02, 1.06, "agreement", fontsize=10, color=MUTED)
    ax.text(0.97, 0.90,
            "one C must fit the whole profile;\n"
            "these differ by up to 7.8×",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=CLAY, weight="bold")
    ax.text(0.97, 0.62, "this failure does not depend on C at all —\n"
                        "it is a failure of shape",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=MUTED, style="italic")

    provenance(META["id"],
               "C_fit(Ĝ1)/C_fit(Ĝ8) at β=40: " +
               ", ".join(f"N={n}: {r:.3f}" for n, r in zip(Ns, ratios)))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
