"""Build every figure, then write manifest.json, provenance.txt, BLOCKED.md
and contact_sheet.png.

    .venv/bin/python figures/build_all.py            # everything
    .venv/bin/python figures/build_all.py 45 46 47   # just these ids
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(only=None):
    scripts = sorted(SRC.glob("fig_*.py"))
    if only:
        scripts = [p for p in scripts if int(p.stem.split("_")[1]) in only]
    if not only:
        (HERE / "provenance.txt").write_text("")   # fresh provenance per build

    entries, blocked, rows = [], [], []
    for path in scripts:
        fid = int(path.stem.split("_")[1])
        try:
            mod = load(path)
            stem = mod.build()
            m = mod.META
            entries.append({"figure_id": m["id"], "filename": f"{stem}.png",
                            "pdf": f"{stem}.pdf", "slide_number": m.get("slide"),
                            "type": m["type"], "data_source": m["data_source"],
                            "caption": m["caption"]})
            rows.append((fid, path.stem, "OK", ""))
        except Exception as exc:
            first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            blocked.append((fid, path.stem, str(exc)))
            rows.append((fid, path.stem, "BLOCKED", first))
            if "--traceback" in sys.argv:
                traceback.print_exc()

    if not only:
        (HERE / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
        write_blocked(blocked)
        contact_sheet(entries)

    print(f"\n{'id':>3}  {'figure':<44} {'status':<8} note")
    print("-" * 96)
    for fid, stem, status, note in rows:
        print(f"{fid:>3}  {stem:<44} {status:<8} {note[:40]}")
    print(f"\n{len(rows) - len(blocked)}/{len(rows)} OK, {len(blocked)} blocked")
    return entries, blocked


def write_blocked(blocked):
    p = HERE / "BLOCKED.md"
    if not blocked:
        p.write_text("# Blocked figures\n\nNone — every figure built from "
                     "real data.\n")
        return
    lines = ["# Blocked figures", "",
             "These figures could not be built because their [DATA] source is "
             "missing. No numbers were invented and no schematic was "
             "substituted.", ""]
    for fid, stem, msg in blocked:
        lines += [f"## fig {fid:02d} — {stem}", "", "```", msg.strip(), "```", ""]
    p.write_text("\n".join(lines) + "\n")


def contact_sheet(entries, ncol=6):
    pngs = [(e["figure_id"], HERE / e["filename"]) for e in entries]
    pngs = [(i, p) for i, p in pngs if p.exists()]
    if not pngs:
        return
    nrow = int(np.ceil(len(pngs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.9, nrow * 2.35),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, (fid, path) in zip(axes, pngs):
        ax.imshow(mpimg.imread(path))
        ax.set_title(f"{fid:02d}", fontsize=9, color="#161F38", pad=2)
    fig.savefig(HERE / "contact_sheet.png", dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:] if a.isdigit()]
    main(ids or None)
