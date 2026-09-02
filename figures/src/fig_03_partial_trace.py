"""fig 03 — partial trace of the Bell state. [SCHEMATIC] half-slide.

The matrices shown are computed exactly (not drawn by hand).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=3, name="partial_trace", slide=7, type="schematic",
            data_source="exact |Φ⁺⟩⟨Φ⁺| and its partial trace (computed)",
            caption="Tracing out B sends the pure Bell density matrix to the "
                    "maximally mixed ρ_A = I/2.")


def build():
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho = np.outer(psi, psi)                      # basis |00>,|01>,|10>,|11>
    rho_A = np.einsum("ikjk->ij", rho.reshape(2, 2, 2, 2))

    fig, axes = new_fig("half", ncols=2, figsize=(5.6, 3.6),
                        gridspec_kw={"width_ratios": [2, 1]})
    for a in axes:
        blank_axes(a)

    ax = axes[0]
    ax.set_xlim(-0.6, 4.2); ax.set_ylim(-1.15, 4.6)
    labels = ["00", "01", "10", "11"]
    for i in range(4):
        for j in range(4):
            v = rho[i, j]
            fc = ICE if v > 0 else "none"
            box(ax, (j, 3 - i), 1, 1, f"{v:.1f}" if v else "0",
                fc=fc, ec=GRID, lw=0.9, fs=10, r=0.0,
                tc=INK if v else MUTED)
        note(ax, (-0.32, 3.5 - i), labels[i], color=MUTED, fs=9)
        note(ax, (i + 0.5, 4.3), labels[i], color=MUTED, fs=9)
    # the 2x2 blocks that get summed
    for bi in range(2):
        ax.add_patch(plt.Rectangle((2 * bi, 2 - 2 * bi), 2, 2, fill=False,
                                   ec=AMBER, lw=2.0, zorder=6))
    note(ax, (2.0, -0.75), "|Φ⁺⟩⟨Φ⁺|   (pure)", color=INK, fs=10.5)

    arrow(ax, (4.35, 2.0), (5.05, 2.0), color=AMBER, lw=1.8)

    ax = axes[1]
    ax.set_xlim(-1.5, 2.6); ax.set_ylim(-2.6, 3.1)
    note(ax, (0.9, 2.55), "Tr_B", color=AMBER, fs=11, weight="bold")
    for i in range(2):
        for j in range(2):
            v = rho_A[i, j]
            box(ax, (j, 1 - i), 1, 1, f"{v:.1f}" if v else "0",
                fc=ICE if v else "none", ec=GRID, lw=0.9, fs=10, r=0.0,
                tc=INK if v else MUTED)
    note(ax, (1.0, -0.55), "ρ_A = I/2", color=INK, fs=11)
    note(ax, (1.0, -1.35), "maximally mixed", color=AMBER, fs=10.5,
         weight="bold")
    note(ax, (1.0, -2.15), "S(A) = log 2", color=MUTED, fs=10)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
