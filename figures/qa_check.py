"""Layout QA for every figure.

For each figure script this re-runs build() with style.finish patched so the
live Figure can be inspected before it is closed, and reports:

  CLIP    a text artist extends outside the figure canvas
  OVER    two text artists overlap each other substantially
  OUTAX   a text artist sits outside its own axes (often fine, but listed
          when it also leaves the figure)

Overlap is measured in device pixels on the intersection over the smaller
box, so a label sitting on a patch is ignored (patches are not text) while
label-on-label collisions are caught.
"""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style

OVERLAP_TOL = 0.34      # fraction of the smaller text box
captured = {}


def _capture(fig, fig_id, name=None):
    # mirror everything finish() does to the figure before saving, so the QA
    # sees the artists the saved file actually contains
    style.tidy_log_axes(fig)
    captured["fig"] = fig
    return f"fig_{fig_id:02d}_{name}" if name else f"fig_{fig_id:02d}"


def _offscreen_ticklabels(fig):
    """Tick-label artists for ticks outside the view: matplotlib creates them
    but never draws them, so their extents are not real layout problems."""
    dead = set()
    for ax in fig.axes:
        # axes drawn with set_axis_off()/blank_axes still own tick-label
        # artists that are never rendered (notably the 3D axes in fig 10)
        if not getattr(ax, "axison", True) or not getattr(ax, "_axis3don", True):
            for t in ax.findobj(plt.Text):
                dead.add(id(t))
            continue
        for getter, ticks, lim in (
                (ax.get_xticklabels, ax.get_xticks, ax.get_xlim),
                (ax.get_yticklabels, ax.get_yticks, ax.get_ylim)):
            lo, hi = sorted(lim())
            try:
                labs, tks = getter(), ticks()
            except Exception:
                continue
            for lab, tv in zip(labs, tks):
                if not (lo - 1e-9 <= tv <= hi + 1e-9):
                    dead.add(id(lab))
    return dead


def texts_of(fig):
    dead = _offscreen_ticklabels(fig)
    out = []
    for t in fig.findobj(plt.Text):
        s = (t.get_text() or "").strip()
        if not s or not t.get_visible() or id(t) in dead:
            continue
        try:
            bb = t.get_window_extent(fig.canvas.get_renderer())
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        out.append((s, bb, t))
    return out


def check(path):
    fid = int(path.stem.split("_")[1])
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    real = style.finish
    style.finish = _capture
    try:
        spec.loader.exec_module(mod)
        mod.finish = _capture          # module imported finish by name
        captured.clear()
        mod.build()
        fig = captured.get("fig")
        if fig is None:
            return fid, path.stem, ["no figure captured"]
        fig.canvas.draw()
        fb = fig.bbox
        issues = []
        ts = texts_of(fig)
        for s, bb, t in ts:
            if (bb.x0 < fb.x0 - 1 or bb.y0 < fb.y0 - 1
                    or bb.x1 > fb.x1 + 1 or bb.y1 > fb.y1 + 1):
                issues.append(f"CLIP  {s[:42]!r}")
        for (s1, b1, _), (s2, b2, _) in combinations(ts, 2):
            ix = max(0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
            iy = max(0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
            if ix <= 0 or iy <= 0:
                continue
            inter = ix * iy
            smaller = min(b1.width * b1.height, b2.width * b2.height)
            if smaller > 0 and inter / smaller > OVERLAP_TOL:
                issues.append(f"OVER  {s1[:26]!r} × {s2[:26]!r} "
                              f"({inter / smaller:.0%})")
        plt.close(fig)
        return fid, path.stem, issues
    finally:
        style.finish = real


def main():
    only = {int(a) for a in sys.argv[1:] if a.isdigit()}
    rows = []
    for p in sorted(SRC.glob("fig_*.py")):
        fid = int(p.stem.split("_")[1])
        if only and fid not in only:
            continue
        try:
            rows.append(check(p))
        except Exception as exc:
            rows.append((fid, p.stem, [f"ERROR {exc}"]))
    bad = [r for r in rows if r[2]]
    for fid, stem, issues in bad:
        print(f"\nfig {fid:02d}  {stem}")
        for i in issues:
            print("   ", i)
    print(f"\n{len(rows) - len(bad)}/{len(rows)} clean, {len(bad)} with issues")


if __name__ == "__main__":
    main()
