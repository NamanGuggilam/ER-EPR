"""fig 68 — excluding the shortest separations. [DATA] half-slide.

DIAGNOSTIC ONLY. This convention was not adopted anywhere.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=68, name="uv_truncation", slide=91, type="data",
            data_source="rung4/diagnostics/uv_out.txt, m_min clamp at N=18, β=5",
            caption="Dropping the shortest separations improves agreement "
                    "monotonically, locating the disagreement in the UV. "
                    "Diagnostic only — not adopted.")


def build():
    txt = data.uv_text()
    blocks = re.findall(
        r"m_min=(\d+): ER H1 bar = \[([\d.]+), ([\d.]+)\]\n"
        r"\s*vs declared:\s+W_H0 =\s*([\d.]+) .*?W_H1 = ([\d.]+).*?\n"
        r"\s*vs conformal:\s+W_H0 =\s*([\d.]+) .*?W_H1 = ([\d.]+)", txt)
    if len(blocks) < 3:
        raise RuntimeError(
            "could not parse the m_min clamp from uv_out.txt; regenerate "
            "with '.venv/bin/python rung4/diagnostics/uv_diagnostics.py'")
    m = np.array([int(b[0]) for b in blocks])
    W0d = np.array([float(b[3]) for b in blocks])
    W0c = np.array([float(b[5]) for b in blocks])

    fig, ax = new_fig("half")
    x = np.arange(len(m))
    ax.bar(x - 0.17, W0d, width=0.34, color=NAVY, zorder=3,
           label="vs declared C")
    ax.bar(x + 0.17, W0c, width=0.34, color=ICE, zorder=3,
           label="vs conformal")
    ax.set_xticks(x)
    ax.set_xticklabels([f"m ≥ {v}" for v in m])
    ax.set_xlabel("shortest separation retained in the filtration")
    ax.set_ylabel("W_H₀")
    ax.legend(loc="upper right", fontsize=9.5)
    for xi, v in zip(x, W0d):
        ax.text(xi - 0.17, v + 0.18, f"{v:.2f}", ha="center", fontsize=10,
                color=INK)
    ax.set_ylim(0, max(W0d) * 1.30)
    ax.text(0.5, 0.90, "DIAGNOSTIC ONLY — NOT ADOPTED",
            transform=ax.transAxes, ha="center", fontsize=12, color=CLAY,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F7E7E4", ec=CLAY,
                      lw=1.6))

    provenance(META["id"],
               f"m_min={list(m)}: W_H₀ vs declared = {list(W0d)}, "
               f"vs conformal = {list(W0c)} (N=18, β=5, n_τ=24)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
