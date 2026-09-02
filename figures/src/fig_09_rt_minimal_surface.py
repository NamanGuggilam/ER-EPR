"""fig 09 — Ryu-Takayanagi minimal surface. [SCHEMATIC] square.

The geodesic is the true hyperbolic geodesic of the Poincaré disc: the arc
orthogonal to the boundary through the two endpoints (centre 1/cos φ,
radius tan φ for half-opening angle φ) — computed, not sketched.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=9, name="rt_minimal_surface", slide=14, type="schematic",
            data_source="exact Poincaré-disc geodesic (computed)",
            caption="The Ryu–Takayanagi prescription: the entropy of a "
                    "boundary region is the area of the bulk geodesic that "
                    "hangs from its endpoints.")


def build():
    fig, ax = new_fig("square")
    blank_axes(ax)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.45, 1.55)

    circle(ax, (0, 0), 1.0, fc="none", ec=NAVY, lw=2.4, zorder=3)

    phi = np.deg2rad(52.0)          # half opening angle of region A
    d, r = 1.0 / np.cos(phi), np.tan(phi)     # orthogonal-circle geodesic
    a0, a1 = np.pi - phi, np.pi + phi
    arcA = np.linspace(a0, a1, 300)
    ax.plot(np.cos(arcA), np.sin(arcA), color=AMBER, lw=4.0, zorder=5,
            solid_capstyle="round")

    t = np.linspace(0, 2 * np.pi, 900)
    gx, gy = -d + r * np.cos(t), r * np.sin(t)
    keep = gx ** 2 + gy ** 2 <= 1.0
    ax.plot(gx[keep], gy[keep], color=NAVY, lw=2.6, zorder=5)

    th = np.linspace(a0, a1, 200)
    fill_x = np.concatenate([np.cos(th), gx[keep][::-1]])
    fill_y = np.concatenate([np.sin(th), gy[keep][::-1]])
    ax.fill(fill_x, fill_y, color=ICE, alpha=0.75, zorder=2, lw=0)

    note(ax, (-1.16, 0.0), "A", fs=15, color=AMBER, weight="bold")
    note(ax, (-0.13, 0.0), "γ_A", fs=13, color=NAVY, weight="bold")
    note(ax, (0.0, -1.30), "S(A) = Area(γ_A) / 4Gℏ", fs=12, color=INK)
    note(ax, (0.62, 0.72), "bulk", fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
