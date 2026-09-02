"""fig 27 — Adamaszek-Adams classification at n = 24. [DATA] full-width.

Theorem 4.3 of arXiv:1503.03669: for 0 ≤ k < n/2,

    Cl(C_n^k) ≃ S^(2l+1)              if l/(2l+1) < k/n < (l+1)/(2l+3)
              ≃ ⋁^(n-2k-1) S^(2l)     if k/n = l/(2l+1)

The branch is DECIDED numerically for each k with exact rational arithmetic
(Fraction), not assumed from the pattern. k = n/2 is outside the theorem's
range: there every pair is within reach, so the complex is the full simplex.
"""
import sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=27, name="adamaszek_adams_classification", slide=35,
            type="data",
            data_source="Theorem 4.3, arXiv:1503.03669, evaluated with exact "
                        "rational arithmetic at n=24, k=1..12",
            caption="The homotopy type of the circulant clique complex at "
                    "n = 24: a circle up to k = 7, a wedge of 7 spheres "
                    "exactly at k/n = 1/3, then odd spheres invisible to H₁.")


SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup(i):
    return str(i).translate(SUP)


def classify(n, k):
    """Return (label, kills_H1) by testing the theorem's branches exactly."""
    r = Fraction(k, n)
    if r >= Fraction(1, 2):
        return "full\nsimplex", True
    for l in range(0, 60):
        lo, hi = Fraction(l, 2 * l + 1), Fraction(l + 1, 2 * l + 3)
        if r == lo:
            return f"⋁{sup(n - 2 * k - 1)} S{sup(2 * l)}", True
        if lo < r < hi:
            return f"S{sup(2 * l + 1)}", (2 * l + 1) != 1
    raise RuntimeError(f"no branch of Theorem 4.3 matched n={n}, k={k}")


def build():
    n = 24
    ks = list(range(1, n // 2 + 1))
    labels, kills = zip(*[classify(n, k) for k in ks])

    fig, ax = new_fig("full", figsize=(11.5, 4.4))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(-0.1, len(ks)); ax.set_ylim(-1.55, 2.5)

    for i, (k, lab, kill) in enumerate(zip(ks, labels, kills)):
        boundary = (Fraction(k, n) == Fraction(1, 3))
        fc = "#F6E4C4" if boundary else CARD
        ec = AMBER if boundary else NAVY
        box(ax, (i + 0.06, 0.0), 0.88, 1.35, "", fc=fc, ec=ec,
            lw=2.2 if boundary else 1.2)
        ax.text(i + 0.5, 0.94, f"k = {k}", ha="center", fontsize=9.5,
                color=INK, zorder=4)
        ax.text(i + 0.5, 0.60, lab, ha="center", va="center",
                fontsize=13 if len(lab) <= 4 else 10.5,
                color=AMBER if boundary else NAVY, zorder=4,
                weight="bold" if boundary else "normal")
        ax.text(i + 0.5, 0.16, f"k/n = {Fraction(k, n)}", ha="center",
                fontsize=8.5, color=MUTED, zorder=4)
        ax.text(i + 0.5, -0.38, "H₁ = ℝ" if not kill else "H₁ = 0",
                ha="center", fontsize=9.5,
                color=TEAL if not kill else CLAY, zorder=4)

    ax.annotate("k/n = 1/3 exactly — H₁ dies here",
                xy=(7.5, 1.42), xytext=(7.5, 2.15),
                ha="center", fontsize=11.5, color=AMBER, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))
    note(ax, (3.6, -1.05), "one loop, for every k below the boundary",
         fs=10.5, color=TEAL)
    note(ax, (10.0, -1.05), "higher spheres — invisible at maxdim = 1",
         fs=10.5, color=MUTED)

    kill_at = [k for k, kl in zip(ks, kills) if kl][0]
    provenance(META["id"],
               f"Theorem 4.3 evaluated exactly (Fraction) at n={n}: "
               f"S¹ for k=1..{kill_at - 1}; k={kill_at} (k/n=1/3) gives "
               f"{labels[kill_at - 1].strip()}; then "
               f"{', '.join(f'k={k}:{l}' for k, l in zip(ks[kill_at:], labels[kill_at:]))}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
