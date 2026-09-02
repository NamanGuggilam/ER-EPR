"""fig 04 — binary entropy. [DATA: analytic, computed here] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=4, name="binary_entropy", slide=8, type="data",
            data_source="analytic S(p) = -p ln p - (1-p) ln(1-p), computed",
            caption="Binary entropy, maximised at p = ½ where S = log 2.")


def build():
    p = np.linspace(1e-6, 1 - 1e-6, 2001)
    S = -p * np.log(p) - (1 - p) * np.log(1 - p)
    smax = float(S.max())

    fig, ax = new_fig("half")
    ax.plot(p, S, color=NAVY, zorder=3)
    ax.plot([0.5], [np.log(2)], "o", color=AMBER, zorder=4)
    ax.axhline(np.log(2), color=MUTED, ls="--", lw=1.0, zorder=2)
    ax.annotate("p = ½,  S = log 2 = 0.6931",
                xy=(0.5, np.log(2)), xytext=(0.56, 0.42), fontsize=10,
                color=AMBER,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.4))
    ax.set_xlabel("p")
    ax.set_ylabel("S(p)   (nats)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.78)

    provenance(META["id"], f"analytic binary entropy on 2001 points; "
                           f"max S = {smax:.6f} at p = 0.5 (log 2 = {np.log(2):.6f})")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
