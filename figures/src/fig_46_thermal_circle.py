"""fig 46 — the 24-point thermal circle. [SCHEMATIC] square."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=46, name="thermal_circle", slide=67, type="schematic",
            data_source="none (schematic); n_tau = 24 as in rung4",
            caption="Euclidean time is a circle: only the separation "
                    "m = min(|i−j|, 24−|i−j|) matters, and m = 1 and m = 8 "
                    "are the two that will carry the result.")


def build():
    n = data.N_TAU
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    P = np.c_[np.cos(th), np.sin(th)]

    fig, ax = new_fig("square")
    blank_axes(ax)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.75, 1.5)

    ax.plot(np.cos(np.linspace(0, 2 * np.pi, 400)),
            np.sin(np.linspace(0, 2 * np.pi, 400)), color=GRID, lw=1.5,
            zorder=1)
    ax.plot(P[:, 0], P[:, 1], "o", color=NAVY, ms=7, zorder=3)

    for i, lab in ((0, "τ = 0"), (6, "β/4"), (12, "β/2"), (18, "3β/4")):
        ax.text(P[i, 0] * 1.19, P[i, 1] * 1.19, lab, ha="center", va="center",
                fontsize=9.5, color=MUTED)

    for m, col, lab in ((1, AMBER, "m = 1"), (8, AMBER, "m = 8")):
        ax.plot(P[[0, m], 0], P[[0, m], 1], color=col, lw=2.6, zorder=4)
        ax.plot(P[[0, m], 0], P[[0, m], 1], "o", color=col, ms=9, zorder=5)
        mid = P[[0, m]].mean(axis=0) * 1.0
        ax.text(mid[0] * 0.72, mid[1] * 0.72, lab, ha="center", va="center",
                fontsize=10.5, color=col, weight="bold")

    note(ax, (0, -1.42), "m = min(|i − j|,  24 − |i − j|)", fs=11.5,
         color=INK)
    note(ax, (0, -1.70), "distance depends on separation only", fs=10,
         color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
