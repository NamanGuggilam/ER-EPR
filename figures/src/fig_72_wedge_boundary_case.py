"""fig 72 — the k/n = 1/3 boundary across n. [DATA] half-slide.

For each n the death index k* (the smallest k with k/n ≥ 1/3) and the branch
of Theorem 4.3 that applies are COMPUTED with exact rational arithmetic.
"""
import sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=72, name="wedge_boundary_case", slide=85, type="data",
            data_source="Theorem 4.3 branch evaluated exactly for n = 21..26",
            caption="Which branch applies depends on n: every n divisible by "
                    "3 lands exactly on k/n = 1/3 and gets the wedge branch, "
                    "the rest get an odd sphere.")


def build():
    ns = list(range(21, 27))
    rows = []
    for n in ns:
        k = -(-n // 3)                       # ceil(n/3): first k with k/n ≥ 1/3
        r = Fraction(k, n)
        boundary = (r == Fraction(1, 3))
        if boundary:
            branch = f"⋁^{n - 2 * k - 1} S²"
        else:
            for l in range(1, 60):
                lo, hi = Fraction(l, 2 * l + 1), Fraction(l + 1, 2 * l + 3)
                if lo < r < hi:
                    branch = f"S^{2 * l + 1}"
                    break
            else:
                raise RuntimeError(f"no branch matched n={n}, k={k}")
        rows.append((n, k, r, boundary, branch))

    fig, ax = new_fig("half", figsize=(6.2, 4.2))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(-0.9, 6.3); ax.set_ylim(-1.5, 6.4)

    heads = ["n", "k*", "k*/n", "branch of Thm 4.3"]
    xs = [0.0, 1.15, 2.5, 4.6]
    for x, h in zip(xs, heads):
        ax.text(x, 5.75, h, ha="center", fontsize=10, color=MUTED)
    for i, (n, k, r, boundary, branch) in enumerate(rows):
        y = 5.0 - i * 0.82
        col = AMBER if boundary else MUTED
        if boundary:
            box(ax, (-0.62, y - 0.31), 6.7, 0.66, "", fc="#FBF1DE", ec=AMBER,
                lw=1.4, r=0.0)
        ax.text(xs[0], y, str(n), ha="center", fontsize=10.5, color=INK,
                zorder=4, weight="bold" if boundary else "normal")
        ax.text(xs[1], y, str(k), ha="center", fontsize=10.5, color=INK,
                zorder=4)
        ax.text(xs[2], y, f"{r}" + ("  = 1/3" if boundary else ""),
                ha="center", fontsize=10.5, color=col, zorder=4,
                weight="bold" if boundary else "normal")
        ax.text(xs[3], y, branch, ha="center", fontsize=10.5,
                color=AMBER if boundary else NAVY, zorder=4,
                weight="bold" if boundary else "normal")

    hits = [n for n, _, _, b, _ in rows if b]
    ax.text(2.7, -0.55, f"exact boundary at n = {', '.join(map(str, hits))}",
            ha="center", fontsize=10.5, color=AMBER, weight="bold")
    ax.text(2.7, -1.15,
            "every n divisible by 3 lands on the boundary — the open point "
            "in the proof",
            ha="center", fontsize=9.5, color=INK, style="italic")

    provenance(META["id"],
               "; ".join(f"n={n}: k*={k}, k*/n={r}, "
                         f"{'BOUNDARY ' if b else ''}{br}"
                         for n, k, r, b, br in rows))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
