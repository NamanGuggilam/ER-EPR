"""fig 50 — Check D: is β/C a real knob? [DATA] half-slide.

Both tests parsed from the committed Check D output. Test 1's caveat is
drawn in the figure: equality there is a manifest symmetry of the integrand,
so it is a regression check rather than evidence.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=50, name="check_d_scale_invariance", slide=71, type="data",
            data_source="rung4_checks_results.txt, Check D tests 1 and 2",
            caption="Check D: G depends on β and C only through β/C — "
                    "test 1 is a manifest symmetry (a regression check), "
                    "test 2 shows the ratio genuinely matters.")


def build():
    txt = data.results_text("checks")
    blk = txt[txt.index("CHECK D"):txt.index("CHECK D PASS")]
    t1 = re.findall(r"\(beta=([\d.]+), C=([\d.]+)\): G = ([\d.e+-]+)", blk)
    t2 = re.findall(r"\(beta=([\d.]+), C=([\d.]+), beta/C=([\d.]+)\): "
                    r"G = ([\d.e+-]+)", blk)
    if len(t1) < 3 or len(t2) < 3:
        raise RuntimeError(
            "could not parse Check D from rung4_checks_results.txt; "
            "regenerate with '.venv/bin/python rung4/rung4_checks.py D'")

    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.2)

    note(ax, (5.0, 8.75), "test 1 — same β/C = 0.02", fs=11.5, color=TEAL,
         weight="bold")
    for i, (b, c, g) in enumerate(t1[:3]):
        y = 7.85 - 0.62 * i
        note(ax, (0.35, y), f"β = {float(b):g},  C = {float(c):g}", fs=10.5,
             color=INK, ha="left")
        note(ax, (9.65, y), f"G = {float(g):.12e}", fs=10.5, color=NAVY,
             ha="right")
    note(ax, (5.0, 5.75), "identical to 13 digits  (spread 0.00e+00)",
         fs=11, color=TEAL)
    note(ax, (5.0, 5.15),
         "a manifest symmetry of the integrand — a regression check,\n"
         "not evidence that β/C is a real knob",
         fs=9.5, color=CLAY, weight="bold")

    ax.plot([0.3, 9.7], [4.35, 4.35], color=GRID, lw=1.2)

    note(ax, (5.0, 3.85), "test 2 — different β/C", fs=11.5, color=AMBER,
         weight="bold")
    for i, (b, c, boc, g) in enumerate(t2[:3]):
        y = 2.95 - 0.62 * i
        note(ax, (0.35, y), f"β/C = {float(boc):g}", fs=10.5, color=INK,
             ha="left")
        note(ax, (9.65, y), f"G = {float(g):.12e}", fs=10.5, color=AMBER,
             ha="right")
    note(ax, (5.0, 0.75), "G changes by 41% — the ratio genuinely matters",
         fs=11, color=AMBER)

    provenance(META["id"],
               f"Check D test 1: {[g for _, _, g in t1[:3]]} (spread 0); "
               f"test 2: {[g for *_, g in t2[:3]]} at β/C = "
               f"{[b for *_, b, _ in [(a, c, d, e) for a, c, d, e in t2[:3]]]}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
