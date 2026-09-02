"""fig 05 — thermofield double construction. [SCHEMATIC] full-width.

Link opacities are the true Boltzmann amplitudes e^(-βEₙ/2) at β = 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=5, name="tfd_construction", slide=10, type="schematic",
            data_source="link opacities = e^(-βEₙ/2), β=1, Eₙ=0..4 (computed)",
            caption="The thermofield double pairs level n on the left with "
                    "level n on the right; tracing out R leaves a thermal state.")


def build():
    fig, axes = new_fig("full", ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})
    for a in axes:
        blank_axes(a)

    E = np.arange(5.0)
    beta = 1.0
    amp = np.exp(-beta * E / 2)
    amp = amp / amp.max()

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 6.6)
    xl, xr, w = 1.1, 8.9, 1.9
    for n, En in enumerate(E):
        y = 0.7 + n * 1.15
        ax.plot([xl - w / 2, xl + w / 2], [y, y], color=NAVY, lw=2.4, zorder=3)
        ax.plot([xr - w / 2, xr + w / 2], [y, y], color=NAVY, lw=2.4, zorder=3)
        ax.plot([xl + w / 2, xr - w / 2], [y, y], color=AMBER,
                lw=2.6, alpha=float(amp[n]), zorder=2)
        note(ax, (xl - w / 2 - 0.42, y), f"|{n}⟩", color=MUTED, fs=9.5)
        note(ax, (xr + w / 2 + 0.42, y), f"|{n}⟩", color=MUTED, fs=9.5)
    note(ax, (xl, 6.25), "L", fs=13, color=INK, weight="bold")
    note(ax, (xr, 6.25), "R", fs=13, color=INK, weight="bold")
    note(ax, (5.0, 0.02), "link opacity ∝ e^(−βEₙ/2)", color=AMBER, fs=10.5)
    note(ax, (5.0, -0.5), "|TFD(β)⟩ = Z^(−½) Σₙ e^(−βEₙ/2) |n⟩_L |n⟩_R",
         color=INK, fs=11)

    ax = axes[1]
    ax.set_xlim(0, 8); ax.set_ylim(-0.6, 6.6)
    arrow(ax, (0.35, 3.4), (2.3, 3.4), color=AMBER, lw=2.0)
    note(ax, (1.3, 3.95), "Tr_R", color=AMBER, fs=12, weight="bold")
    for n, En in enumerate(E):
        y = 0.7 + n * 1.15
        p = float(np.exp(-beta * En))
        ax.plot([3.0, 3.0 + 3.6 * p / np.exp(0.0)], [y, y],
                color=NAVY, lw=7, solid_capstyle="butt", zorder=3)
        note(ax, (2.62, y), f"|{n}⟩", color=MUTED, fs=9.5)
    note(ax, (4.6, 0.02), "ρ_L = e^(−βH)/Z", color=INK, fs=11)
    note(ax, (4.6, -0.5), "a thermal state at temperature 1/β",
         color=MUTED, fs=10.5, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
