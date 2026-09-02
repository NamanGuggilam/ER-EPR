"""fig 34 — JT / AdS₂ in the Poincaré disc. [DATA: computed metric] square.

Geodesics are true hyperbolic geodesics (circles orthogonal to the boundary);
the cutoff curve is a wiggly curve at small proper distance from the edge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=34, name="jt_ads2_geometry", slide=43, type="data",
            data_source="Poincaré-disc geodesics (orthogonal circles) + "
                        "finite-cutoff boundary curve, computed",
            caption="JT gravity on AdS₂: the bulk geometry is rigid — all the "
                    "dynamics lives in the shape of the finite-cutoff "
                    "boundary curve.")


def geodesic(a0, a1, npts=300):
    """Hyperbolic geodesic of the unit disc joining boundary angles a0, a1."""
    mid, half = (a0 + a1) / 2, abs(a1 - a0) / 2
    if abs(np.sin(half)) < 1e-12:
        t = np.linspace(-1, 1, npts)
        return t * np.cos(a0), t * np.sin(a0)
    d, r = 1.0 / np.cos(half), np.tan(half)
    cx, cy = d * np.cos(mid), d * np.sin(mid)
    th = np.linspace(0, 2 * np.pi, 1400)
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    keep = x ** 2 + y ** 2 <= 1.0
    return x[keep], y[keep]


def build():
    fig, ax = new_fig("square")
    blank_axes(ax)
    ax.set_xlim(-1.42, 1.42); ax.set_ylim(-1.5, 1.42)

    t = np.linspace(0, 2 * np.pi, 900)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.2, ls="--", zorder=2)

    rng = np.random.default_rng(4)
    angs = np.linspace(0, 2 * np.pi, 13)[:-1]
    for i in range(len(angs)):
        for j in range(i + 2, len(angs), 3):
            gx, gy = geodesic(angs[i], angs[j])
            ax.plot(gx, gy, color=ICE, lw=1.1, zorder=1)

    eps = 0.085
    wig = 1.0 - eps * (1.0 + 0.30 * np.sin(5 * t + 0.6)
                       + 0.18 * np.sin(9 * t + 2.0))
    ax.plot(wig * np.cos(t), wig * np.sin(t), color=AMBER, lw=2.8, zorder=5)

    note(ax, (0, 0.14), "rigid bulk", fs=12, color=NAVY, weight="bold")
    note(ax, (0, -0.22), "constant negative curvature", fs=10, color=MUTED)
    note(ax, (0, 1.24), "boundary curve at finite cutoff", fs=10.5,
         color=AMBER, weight="bold")
    note(ax, (0, -1.32),
         "all the dynamics is in the shape of this curve", fs=10.5,
         color=INK, style="italic")

    provenance(META["id"],
               f"Poincaré-disc geodesics as orthogonal circles "
               f"(centre 1/cos φ, radius tan φ); cutoff curve at mean radius "
               f"1 − ε with ε = {eps}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
