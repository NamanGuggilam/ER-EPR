"""fig 40 — rung 1 barcodes. [DATA] full-width.

Values are parsed from the committed rung1_results.txt, using the EXACT
double-precision bars the run printed alongside the float32 ones ripser
reports.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=40, name="rung1_barcodes", slide=52, type="data",
            data_source="rung1_results.txt (exact double-precision bars)",
            caption="Rung 1: the two H₀ barcodes coincide exactly once the "
                    "ER bar is rescaled by φ.")


def build():
    txt = data.results_text("rung1")

    def grab(pattern, label):
        m = re.search(pattern, txt)
        if not m:
            raise RuntimeError(
                f"could not parse {label} from rung1_results.txt — the "
                f"results format changed; regenerate with "
                f"'.venv/bin/python rung1.py'")
        return float(m.group(1))

    # the EXACT double-precision bars the run printed (not the float32 ones)
    d_ent = grab(r"H0 \(EPR side\) finite bars, exact\s*:\s*\[\[0\.0, "
                 r"([\d.]+)\]\]", "exact EPR bar")
    ell = grab(r"H0 \(ER side\)\s+finite bars, exact, raw\s*:\s*\[\[0\.0, "
               r"([\d.]+)\]\]", "exact ER bar")
    ell_norm = grab(r"exact, normalized by phi\s*:\s*\[\[0\.0, ([\d.]+)\]\]",
                    "phi-normalised ER bar")
    scale = grab(r"scale_factor = ell_throat / d_ent = ([\d.]+)", "scale")
    raw_epr = grab(r"H0 \(EPR side\) finite bars, as reported by ripser\s*:"
                   r"\s*\[\[0\.0, ([\d.]+)\]\]", "float32 EPR bar")
    raw_er = grab(r"H0 \(ER side\)\s+finite bars, as reported by ripser\s*:"
                  r"\s*\[\[0\.0, ([\d.]+)\]\]", "float32 ER bar")
    W = grab(r"Wasserstein distance \(normalized\)\s*=\s*([\d.e+-]+)",
             "normalised Wasserstein")

    fig, ax = new_fig("full", figsize=(11.5, 4.4))
    style_axes(ax, grid_axis="x")
    ax.set_aspect("auto")

    bars = [("EPR   d = 1/I(A:B)", d_ent, NAVY, 3),
            ("ER    ℓ = S(A)", ell, LIFT, 2),
            ("ER rescaled by φ", ell_norm, AMBER, 1)]
    for lab, x, col, y in bars:
        bar_h(ax, y, 0.0, x, color=col, lw=15)
        ax.text(-0.012, y, lab, ha="right", va="center", fontsize=11,
                color=col)
        ax.text(x + 0.008, y, f"{x:.10f}", ha="left", va="center",
                fontsize=10.5, color=INK)

    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("t")
    ax.set_xlim(-0.30, 0.98); ax.set_ylim(0.3, 4.35)
    note(ax, (0.42, 3.85), f"φ = S(A)·I(A:B) = 2(log 2)² = {scale:.7f}",
         fs=11.5, color=INK)
    note(ax, (0.42, 0.72),
         f"Wasserstein after rescaling = {W:.1e}   (exact, double precision)",
         fs=11, color=AMBER, weight="bold")
    note(ax, (0.42, 0.45),
         f"ripser reports these bars in float32 ({raw_epr:.7f}, {raw_er:.7f}); "
         "the isomorphism check uses the exact values",
         fs=9.5, color=MUTED, style="italic")

    provenance(META["id"],
               f"rung1_results.txt exact bars: d_ent={d_ent}, ell={ell}, "
               f"φ={scale}, ell/φ={ell_norm} (= d_ent to "
               f"{abs(ell_norm - d_ent):.1e}), W={W:.1e}; ripser float32 bars "
               f"{raw_epr}, {raw_er}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
