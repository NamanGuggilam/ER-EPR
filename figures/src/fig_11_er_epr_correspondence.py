"""fig 11 — the ER=EPR correspondence. [SCHEMATIC] full-width.

The conceptual anchor of the talk: entanglement on the left, geometry on the
right, one equals sign between them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=11, name="er_epr_correspondence", slide=16, type="schematic",
            data_source="none (schematic)",
            caption="ER = EPR: two entangled many-body systems (left) and two "
                    "boundaries joined by a wormhole (right) are proposed to be "
                    "the same object described two ways.")


def build():
    fig, ax = new_fig("full")
    blank_axes(ax)
    ax.set_xlim(0, 20); ax.set_ylim(0, 7.6)

    # --- EPR side: two many-body systems, many entanglement links
    rng = np.random.default_rng(3)
    for cx, lab in ((2.6, "L"), (6.6, "R")):
        circle(ax, (cx, 4.0), 1.35, fc=CARD, ec=NAVY, lw=2.0, zorder=2)
        ang = np.linspace(0, 2 * np.pi, 9)[:-1] + (0.2 if lab == "R" else 0.0)
        ax.plot(cx + 0.95 * np.cos(ang), 4.0 + 0.95 * np.sin(ang), "o",
                color=NAVY, ms=6, zorder=4)
        note(ax, (cx, 2.15), lab, fs=13, color=INK, weight="bold")
    for k in range(5):
        y0 = 4.0 + 0.78 * np.sin(2 * np.pi * k / 5)
        y1 = 4.0 + 0.78 * np.sin(2 * np.pi * k / 5 + 0.8)
        wavy(ax, (4.02, y0), (5.22, y1), amp=0.085, cycles=3, color=AMBER,
             lw=1.5)
    note(ax, (4.6, 6.35), "entanglement", fs=12, color=AMBER, weight="bold")
    note(ax, (4.6, 1.15), "two entangled quantum systems", fs=11, color=MUTED)

    # --- the claim
    note(ax, (10.0, 4.05), "=", fs=42, color=AMBER, weight="bold")

    # --- ER side: two boundary mouths joined by a throat (surface of
    # revolution seen edge-on; mouths drawn as ellipses, not discs)
    from matplotlib.patches import Ellipse
    x0, x1, mouth, waist = 13.2, 17.8, 1.32, 0.46
    ts = np.linspace(0, 1, 300)
    xs = x0 + ts * (x1 - x0)
    prof = mouth - (mouth - waist) * np.sin(np.pi * ts)
    ax.fill_between(xs, 4.0 - prof, 4.0 + prof, color=ICE, alpha=0.60,
                    zorder=2, lw=0)
    for sgn in (+1, -1):
        ax.plot(xs, 4.0 + sgn * prof, color=LIFT, lw=2.4, zorder=3)
    for cx, lab in ((x0, "L"), (x1, "R")):
        ax.add_patch(Ellipse((cx, 4.0), 0.62, 2 * mouth, fc="none", ec=NAVY,
                             lw=2.4, zorder=5))
        note(ax, (cx, 2.15), lab, fs=13, color=INK, weight="bold")
    note(ax, (15.4, 6.35), "wormhole throat", fs=12, color=LIFT, weight="bold")
    note(ax, (15.4, 1.15), "one connected geometry", fs=11, color=MUTED)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
