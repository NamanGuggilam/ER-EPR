"""fig 67 — refining the grid makes agreement worse. [DATA] half-slide.

Per-pair gap = W_H₀ / (n_τ − 1), which removes the trivial growth from
having more H₀ bars, so what is left is a real increase in disagreement.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=67, name="uv_ntau_sweep", slide=90, type="data",
            data_source="rung4/diagnostics/uv_out.txt, n_τ sweep at N=18, β=5",
            caption="Refining the time grid makes the disagreement worse — "
                    "the opposite of a discretisation artefact — because "
                    "finer grids probe further into the UV.")


def build():
    txt = data.uv_text()
    blocks = re.findall(
        r"n_tau=\s*(\d+) \(tau_min=([\d.]+).*?\n"
        r"\s*ER H1 bar = \[([\d.]+), ([\d.]+)\]; SYK H1 birth mean = ([\d.]+)\n"
        r"\s*vs declared:\s+W_H0 =\s*([\d.]+) .*?W_H1 = ([\d.]+)", txt)
    if len(blocks) < 3:
        raise RuntimeError(
            "could not parse the n_tau sweep from uv_out.txt; regenerate "
            "with '.venv/bin/python rung4/diagnostics/uv_diagnostics.py'")
    n = np.array([int(b[0]) for b in blocks])
    tmin = np.array([float(b[1]) for b in blocks])
    er_b = np.array([float(b[2]) for b in blocks])
    syk_b = np.array([float(b[4]) for b in blocks])
    W0 = np.array([float(b[5]) for b in blocks])
    W1 = np.array([float(b[6]) for b in blocks])
    gap = W0 / (n - 1)

    fig, ax = new_fig("half")
    ax.plot(n, gap, "o-", color=AMBER, ms=8, zorder=5,
            label="per-pair gap  W_H₀/(n_τ−1)")
    ax.plot(n, W1, "s--", color=NAVY, ms=7, zorder=4, label="W_H₁")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n])
    ax.set_xlabel("n_τ  (points on the thermal circle)")
    ax.set_ylabel("disagreement per pair")
    ax.legend(loc="upper left", fontsize=9.5)
    for x, g, t in zip(n, gap, tmin):
        ax.annotate(f"{g:.3f}\nτ_min = {t:.3f}", xy=(x, g), xytext=(0, -30),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=MUTED)
    ax.set_ylim(0.20, 0.62)
    ax.text(0.97, 0.06, "finer grid ⇒ worse agreement",
            transform=ax.transAxes, ha="right", fontsize=10.5, color=CLAY,
            weight="bold")

    provenance(META["id"],
               f"n_τ={list(n)}, τ_min={list(tmin)}: W_H₀={list(W0)}, "
               f"per-pair gap={list(np.round(gap, 4))}, W_H₁={list(W1)}; "
               f"ER birth {list(er_b)} vs SYK birth {list(syk_b)}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
