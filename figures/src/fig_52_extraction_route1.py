"""fig 52 — extraction Route 1: no DOS plateau. [DATA] half-slide.

Band-resolved C from the committed extract run: a plateau across bands would
be a measurement of C, and there is none at any N.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=52, name="extraction_route1", slide=74, type="data",
            data_source="rung4_results.txt extract stage, Route 1 band tables",
            caption="Route 1: the fitted C drifts by a factor of 1.6–4.3 "
                    "across fitting bands at every N — no plateau, so no "
                    "measurement of C.")


def build():
    txt = data.results_text("rung4")
    block = txt[:txt.index("4. THE C(N) GATE")]
    out = {}
    for N in data.N_VALUES:
        m = re.search(rf"N={N:>2}  Route1 C=[\d.]+\+-[\d.]+  plateau "
                      rf"max/min=([\d.]+).*?\n\s*bands: (.*)", block)
        if not m:
            raise RuntimeError(
                f"could not parse Route 1 bands for N={N} from "
                f"rung4_results.txt; regenerate with "
                f"'.venv/bin/python rung4/rung4.py extract'")
        ratio = float(m.group(1))
        bands = re.findall(r"([\d.]+-[\d.]+):([\d.]+)", m.group(2))
        out[N] = (ratio, [b[0] for b in bands],
                  np.array([float(b[1]) for b in bands]))

    fig, ax = new_fig("half")
    cols = [NAVY, LIFT, TEAL, AMBER]
    labels = out[data.N_VALUES[0]][1]
    x = np.arange(len(labels))
    for (N, col) in zip(data.N_VALUES, cols):
        ratio, _, vals = out[N]
        ax.plot(x, vals, "o-", color=col, zorder=3,
                label=f"N = {N}   (max/min = {ratio:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=20)
    ax.set_xlabel("edge quantile band used for the fit")
    ax.set_ylabel("fitted C")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.03, 0.10, "a plateau would give C — there isn't one\n"
                        "(tolerance for 'plateau' was max/min < 1.25)",
            transform=ax.transAxes, fontsize=10, color=CLAY, weight="bold")

    provenance(META["id"],
               "; ".join(f"N={N}: bands={list(np.round(out[N][2], 4))}, "
                         f"max/min={out[N][0]}" for N in data.N_VALUES))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
