"""fig 41 — the rung 1 circularity. [SCHEMATIC] full-width.

Key slide: the loop must be obvious in one second.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=41, name="rung1_tautology", slide=53, type="schematic",
            data_source="none (schematic)",
            caption="Rung 1's agreement is definitional: φ is defined as the "
                    "ratio it is then verified to reproduce.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 5.0))
    blank_axes(ax)
    ax.set_xlim(0, 20); ax.set_ylim(-0.6, 8.0)

    cx, cy, rx, ry = 10.0, 3.9, 6.5, 2.5
    steps = [
        (cx, cy + ry, "d_ent ≡ 1/I(A:B)\nis DEFINED"),
        (cx + rx, cy, "φ ≡ ℓ / d_ent\nis DEFINED"),
        (cx, cy - ry, "φ is APPLIED\nto the ER bar"),
        (cx - rx, cy, "the result EQUALS\nd_ent"),
    ]
    for (x, y, lab) in steps:
        box(ax, (x - 2.15, y - 0.72), 4.3, 1.44, lab, fc=CARD, ec=CLAY,
            tc=INK, fs=11, lw=1.8)

    th = np.linspace(0, 2 * np.pi, 400)
    for k in range(4):
        a0 = np.pi / 2 - k * np.pi / 2
        a1 = a0 - np.pi / 2
        seg = np.linspace(a0 - 0.30, a1 + 0.30, 60)
        ax.plot(cx + rx * np.cos(seg), cy + ry * np.sin(seg), color=CLAY,
                lw=2.2, zorder=3)
        xm, ym = cx + rx * np.cos(seg[-1]), cy + ry * np.sin(seg[-1])
        dx = cx + rx * np.cos(seg[-1] - 0.04) - xm
        dy = cy + ry * np.sin(seg[-1] - 0.04) - ym
        ax.annotate("", xy=(xm + dx * 3, ym + dy * 3), xytext=(xm, ym),
                    arrowprops=dict(arrowstyle="-|>", color=CLAY, lw=2.2))

    note(ax, (cx, cy + 0.42), "no physics is consulted", fs=12.5, color=CLAY,
         weight="bold")
    note(ax, (cx, cy - 0.12), "between the definition and the measurement",
         fs=11, color=CLAY)
    note(ax, (cx, -0.25),
         "the Wasserstein distance of 0 is an identity, not a result",
         fs=11, color=INK, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
