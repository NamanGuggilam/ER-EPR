"""fig 44 — why points on a line have no H₁. [SCHEMATIC] full-width.

The chord shown is computed: for the four points drawn and the threshold
drawn, the pair that closes the would-be 4-cycle really is within t.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=44, name="chordal_indifference_proof", slide=63,
            type="schematic",
            data_source="4 points on ℝ with an explicit threshold; the chord "
                        "is verified against that threshold",
            caption="Points on a line give an indifference graph, which is "
                    "chordal: every long cycle has a chord, so the clique "
                    "complex is contractible and H₁ is exactly zero.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 5.0))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(-0.6, 10.6); ax.set_ylim(-2.5, 5.4)

    x = np.array([0.6, 2.4, 4.0, 5.6])
    t = 3.4
    y0 = 3.6
    ax.plot([-0.2, 10.2], [y0, y0], color=GRID, lw=1.5, zorder=1)
    ax.plot(x, [y0] * 4, "o", color=INK, ms=10, zorder=4)
    for i, xi in enumerate(x):
        ax.text(xi, y0 + 0.45, f"x{i}", ha="center", fontsize=10, color=MUTED)
    ax.text(10.2, y0 + 0.45, "ℝ", ha="right", fontsize=12, color=MUTED)

    ax.annotate("", xy=(x[0] + t, y0 - 0.55), xytext=(x[0], y0 - 0.55),
                arrowprops=dict(arrowstyle="<|-|>", color=AMBER, lw=1.6))
    ax.text(x[0] + t / 2, y0 - 0.95, f"threshold t = {t}", ha="center",
            fontsize=10.5, color=AMBER)

    # would-be 4-cycle
    V = np.array([[7.2, 4.4], [9.4, 4.4], [9.4, 2.2], [7.2, 2.2]])
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ax.plot(V[[a, b], 0], V[[a, b], 1], color=NAVY, lw=2.4, zorder=3)
    close = abs(x[0] - x[2]) <= t                      # the chord x0-x2
    assert close, "the chord pair is not within t — the drawing would lie"
    ax.plot(V[[0, 2], 0], V[[0, 2], 1], color=CLAY, lw=2.4, ls="--", zorder=4)
    ax.plot(V[:, 0], V[:, 1], "o", color=INK, ms=9, zorder=5)
    for lab, (px, py) in zip(["x0", "x1", "x2", "x3"], V):
        ax.text(px + (0.30 if px > 8 else -0.30), py, lab, ha="center",
                va="center", fontsize=10, color=MUTED)
    ax.text(8.3, 1.55, f"the chord x0–x2 exists\n(|x0 − x2| = {abs(x[0]-x[2]):.1f} ≤ t)",
            ha="center", fontsize=10.5, color=CLAY)
    ax.text(8.3, 5.05, "a would-be 4-cycle", ha="center", fontsize=11,
            color=NAVY)

    steps = ["points on a line", "⇒ indifference graph",
             "⇒ chordal: every cycle ≥ 4 has a chord",
             "⇒ clique complex is contractible", "⇒ H₁ = 0 exactly"]
    for i, s in enumerate(steps):
        ax.text(0.6, 1.6 - 0.62 * i, s, ha="left", fontsize=11,
                color=INK if i < 4 else CLAY,
                weight="bold" if i == 4 else "normal")

    note(ax, (5.3, -2.25),
         "curvature and triangle-filling arguments were tried first — both "
         "are wrong; chordality is the reason",
         fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
