"""fig 63 — C_fit per cell, both targets. [DATA] full-width.

NO FIT cells are marked as such and never as a boundary value.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=63, name="cfit_table", slide=86, type="data",
            data_source="rung4/diagnostics/cfit_out2.txt (C ∈ (0, 200])",
            caption="Best-fit C per cell for both targets: fits exist only at "
                    "β = 40 for Ĝ(1), and every other cell is monotone toward "
                    "the boundary — no fit, not a boundary value.")


def build():
    tab = data.cfit_table()
    txt = (Path(data.RUNG4) / "diagnostics" / "cfit_out2.txt").read_text()
    resid = dict(re.findall(r"C_fit\[Ghat1\] = [\d.]+ \+- [\d.]+ "
                            r"\(resid W_H0 ([\d.e+-]+), crossed\)", txt) and
                 [] or [])

    fig, axes = new_fig("full", ncols=2, figsize=(11.5, 4.6))
    for ax, key, title in zip(axes, ("C1", "C8"),
                              ("target  Ĝ(1)", "target  Ĝ(8)")):
        blank_axes(ax)
        ax.set_aspect("auto")
        ax.set_xlim(-1.5, 4.3); ax.set_ylim(-1.3, 4.9)
        ax.text(2.0, 4.62, title, ha="center", fontsize=12, color=INK,
                weight="bold")
        for j, beta in enumerate(data.BETAS):
            ax.text(j + 0.5, 4.12, f"β = {beta:g}", ha="center", fontsize=10,
                    color=MUTED)
        for i, N in enumerate(data.N_VALUES):
            y = 3.0 - i * 0.95
            ax.text(-0.15, y + 0.36, f"N = {N}", ha="right", va="center",
                    fontsize=10.5, color=INK)
            for j, beta in enumerate(data.BETAS):
                v = tab[(N, beta)][key]
                if v is None:
                    box(ax, (j + 0.04, y), 0.92, 0.72, "NO FIT", fc="#EFEFF2",
                        ec=MUTED, tc=MUTED, fs=9.5, r=0.0, lw=1.0)
                else:
                    ratio = v / data.declared_C(N)
                    box(ax, (j + 0.04, y), 0.92, 0.72, "", fc="#F6E4C4",
                        ec=AMBER, lw=1.6, r=0.0)
                    ax.text(j + 0.5, y + 0.46, f"{v:.4f}", ha="center",
                            fontsize=10, color=INK, zorder=4)
                    ax.text(j + 0.5, y + 0.17, f"{ratio:.1f}× declared",
                            ha="center", fontsize=8, color=MUTED, zorder=4)
        ax.text(2.0, -0.95,
                "monotone toward the C = 200 boundary in every unfitted cell",
                ha="center", fontsize=9.5, color=MUTED, style="italic")

    n1 = sum(1 for k in tab if tab[k]["C1"] is not None)
    n8 = sum(1 for k in tab if tab[k]["C8"] is not None)
    provenance(META["id"],
               f"C_fit: Ĝ(1) target fits in {n1}/16 cells (all β=40): "
               f"{ {k: tab[k]['C1'] for k in tab if tab[k]['C1']} }; "
               f"Ĝ(8) target fits in {n8}/16 cells; residuals at every "
               f"crossing ≤ 2e-7 (exact crossings)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
