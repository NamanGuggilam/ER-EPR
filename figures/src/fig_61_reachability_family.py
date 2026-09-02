"""fig 61 — reachability. [DATA] full-width.

The Schwarzian family's Ĝ(1) against β/C (from the committed sweep) with the
conformal infimum, and all 16 measured SYK values overlaid. Twelve sit below
the floor; the four β = 40 cells sit inside the family's range.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=61, name="reachability_family", slide=84, type="data",
            data_source="cfit_out2.txt Ĝ_Schwarzian(1) sweep + "
                        "profiles_cache.pkl SYK values",
            caption="The Schwarzian family is bounded below by the conformal "
                    "value 2.7679; SYK sits beneath that floor for every "
                    "β ≤ 20 and only enters the family's range at β = 40.")


def build():
    txt = data.uv_text() if False else None
    p = Path(data.RUNG4) / "diagnostics" / "cfit_out2.txt"
    if not p.exists():
        raise data.MissingData(
            f"MISSING: {p}\n  regenerate with: .venv/bin/python "
            f"rung4/diagnostics/cfit_battery.py")
    rows = re.findall(r"beta=\s*([\d.]+) beta/C=\s*([\d.]+) \(C=[\d.eE+-]+\): "
                      r"Ghat\(1\) = ([\d.]+)", p.read_text())
    if not rows:
        raise RuntimeError("could not parse the Ĝ(1) sweep from cfit_out2.txt")
    boc = np.array([float(r[1]) for r in rows])
    g1 = np.array([float(r[2]) for r in rows])
    o = np.argsort(boc)
    boc_u, idx = np.unique(boc[o], return_index=True)
    g1_u = g1[o][idx]

    syk = data.ghat_syk()

    fig, ax = new_fig("full", figsize=(11.5, 4.8))
    ax.plot(boc_u, g1_u, "-o", color=NAVY, zorder=4,
            label="Schwarzian family  Ĝ(1)  (function of β/C alone)")
    ax.axhline(data.CONF_G1, color=AMBER, ls=(0, (6, 3)), lw=2.0, zorder=3)
    ax.axhspan(0.9, data.CONF_G1, color=CLAY, alpha=0.07, zorder=1)

    cols = {5.0: "#9AB0D4", 10.0: LIFT, 20.0: NAVY, 40.0: CLAY}
    for beta in data.BETAS:
        xs, ys, es = [], [], []
        for N in data.N_VALUES:
            m, s, *_ = syk[(N, beta)]
            xs.append(beta / data.declared_C(N)); ys.append(m); es.append(s)
        ax.errorbar(xs, ys, yerr=es, fmt="s", color=cols[beta], ms=7,
                    capsize=3, lw=1.2, zorder=5, label=f"SYK, β = {beta:g}")

    ax.set_xscale("log")
    ax.set_xlabel("β / C")
    ax.set_ylabel("Ĝ(1)   (nearest-neighbour correlation ratio)")
    ax.set_ylim(0.9, 8.2)
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)
    ax.text(0.035, data.CONF_G1 + 0.16,
            f"conformal infimum  {data.CONF_G1:.4f}", fontsize=10.5,
            color=AMBER, weight="bold")
    ax.text(0.035, 1.55, "unreachable by ANY C\n(12 of 16 cells)",
            fontsize=10.5, color=CLAY, weight="bold")

    below = sum(1 for k in syk if syk[k][0] < data.CONF_G1)
    provenance(META["id"],
               f"Schwarzian Ĝ(1) sweep over β/C ∈ [{boc_u.min():g}, "
               f"{boc_u.max():g}] (min {g1_u.min():.6f} ≥ conformal "
               f"{data.CONF_G1:.6f}); {below}/16 SYK cells below the "
               f"infimum, {16 - below} above (all β=40)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
