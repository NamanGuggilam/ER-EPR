"""fig 51 — Check E: small-C dense-grid cross-check. [DATA] half-slide."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=51, name="check_e_dense_grid", slide=72, type="data",
            data_source="rung4_checks_results.txt, Check E table",
            caption="Check E: adaptive quadrature agrees with an independent "
                    "dense grid to machine precision in the C ≈ 0.1–0.2 "
                    "regime where the physics actually sits.")


def build():
    txt = data.results_text("checks")
    rows = re.findall(r"^\s*([\d.]+) \|\s*([\d.]+) \|\s*([\d.e+-]+) \|"
                      r"\s*([\d.e+-]+) \|\s*([\d.e+-]+) \|\s*([\d.e+-]+) \|",
                      txt, flags=re.M)
    rows = [r for r in rows if float(r[0]) <= 1.0]
    if len(rows) < 4:
        raise RuntimeError(
            "could not parse Check E table from rung4_checks_results.txt; "
            "regenerate with '.venv/bin/python rung4/rung4_checks.py E'")
    C = np.array([float(r[0]) for r in rows])
    tb = np.array([float(r[1]) for r in rows])
    rel = np.array([float(r[4]) for r in rows])
    floor = 1e-17
    shown = np.where(rel > 0, rel, floor)

    fig, ax = new_fig("half")
    labels = [f"C = {c:g}\nτ/β = {t:g}" for c, t in zip(C, tb)]
    xs = np.arange(len(rows))
    ax.bar(xs, shown, color=NAVY, width=0.55, zorder=3)
    ax.axhline(1e-6, color=CLAY, ls="--", lw=1.6, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|quad − dense grid| / G")
    ax.set_ylim(floor / 3, 1e-4)
    ax.text(0.02, 2e-6, "acceptance threshold  1e−6", fontsize=10, color=CLAY)
    ax.text(0.02, 0.72,
            "Checks A–D lived at C ∈ [25, 200];\nthe physics sits at C ≈ 0.1–0.2",
            transform=ax.transAxes, fontsize=10, color=MUTED, va="top")

    provenance(META["id"],
               f"Check E: C={list(C)}, τ/β={list(tb)}, relative differences="
               f"{list(rel)} (all ≤ 1.3e-15, threshold 1e-6)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
