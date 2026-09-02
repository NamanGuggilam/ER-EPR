"""fig 10 — Einstein-Rosen bridge. [DATA: computed embedding] full-width.

Flamm's paraboloid: the exact isometric embedding of the t = const, θ = π/2
slice of Schwarzschild, z(r) = ±2√(r_s(r − r_s)) with r_s = 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=10, name="einstein_rosen_bridge", slide=15, type="data",
            data_source="Flamm paraboloid z = ±2√(r_s(r−r_s)), r_s=1 (computed)",
            caption="The Einstein–Rosen bridge: two asymptotically flat "
                    "sheets joined at a throat that pinches off too fast to cross.")


def build():
    rs = 1.0
    r = np.linspace(rs, 6.0, 160)
    ph = np.linspace(0, 2 * np.pi, 160)
    R, PH = np.meshgrid(r, ph)
    X, Y = R * np.cos(PH), R * np.sin(PH)
    Z = 2.0 * np.sqrt(rs * (R - rs))

    fig = plt.figure(figsize=SIZES["full"], constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    for sgn, col in ((+1, NAVY), (-1, LIFT)):
        ax.plot_surface(X, Y, sgn * Z, rstride=4, cstride=4, color=col,
                        alpha=0.30, linewidth=0.35, edgecolor=col,
                        shade=False, zorder=2)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rs * np.cos(th), rs * np.sin(th), np.zeros_like(th),
            color=AMBER, lw=3.0, zorder=6)

    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((1, 1, 0.85))
    lim = 6.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-5.2, 5.2)

    ax.text2D(0.50, 0.955, "two asymptotically flat sheets",
              transform=ax.transAxes, ha="center", fontsize=10.5, color=MUTED)
    ax.text2D(0.50, 0.50, "throat", transform=ax.transAxes, ha="center",
              fontsize=11, color=AMBER, weight="bold")
    ax.text2D(0.50, 0.045,
              "non-traversable — the throat pinches faster than light can cross",
              transform=ax.transAxes, ha="center", fontsize=10.5, color=INK)

    provenance(META["id"],
               f"Flamm paraboloid computed on a {R.shape[0]}×{R.shape[1]} grid, "
               f"r ∈ [{rs}, 6], r_s = {rs}; throat circumference radius = {rs}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
