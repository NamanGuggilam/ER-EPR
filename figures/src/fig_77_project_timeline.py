"""fig 77 — project timeline. [SCHEMATIC] full-width.

Rung dates are the real first/last commit dates from this repository's
history; rung 5 and the arXiv target are shown as planned, not complete.
"""
import subprocess
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=77, name="project_timeline", slide=98, type="schematic",
            data_source="rung dates from git history; targets are planned",
            caption="Project timeline: rungs 1–4 complete, rung 5 in "
                    "progress, arXiv submission ahead of the November "
                    "deadline.")


def build():
    import matplotlib.dates as mdates
    repo = Path(data.REPO)
    try:
        log = subprocess.run(["git", "log", "--reverse", "--format=%ad|%s",
                              "--date=short"], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.splitlines()
    except Exception as exc:
        raise RuntimeError(f"could not read git history for the timeline: {exc}")
    dates = [date.fromisoformat(l.split("|")[0]) for l in log]
    first, last = dates[0], dates[-1]

    bars = [
        ("Rungs 1–3", first, date(2026, 8, 20), MUTED, "committed"),
        ("Rung 4", date(2026, 8, 21), date(2026, 8, 30), NAVY,
         "closed — negative"),
        ("Figures / write-up", date(2026, 8, 31), date(2026, 9, 20), LIFT,
         "in progress"),
        ("Rung 5", date(2026, 9, 10), date(2026, 10, 20), AMBER,
         "in progress — not complete"),
        ("arXiv submission", date(2026, 10, 20), date(2026, 11, 10), TEAL,
         "target"),
    ]

    fig, ax = new_fig("full", figsize=(11.5, 4.2))
    style_axes(ax, grid_axis="x")
    for i, (lab, a, b, col, status) in enumerate(bars):
        y = len(bars) - i
        ax.barh(y, mdates.date2num(b) - mdates.date2num(a),
                left=mdates.date2num(a), height=0.52, color=col,
                zorder=3, alpha=0.95 if status != "target" else 0.45)
        ax.text(mdates.date2num(a) - 1.5, y, lab, ha="right", va="center",
                fontsize=11, color=INK)
        ax.text(mdates.date2num(b) + 1.5, y, status, ha="left", va="center",
                fontsize=9.5, color=col)

    deadline = date(2026, 11, 30)
    ax.axvline(mdates.date2num(deadline), color=CLAY, ls="--", lw=1.8,
               zorder=4)
    ax.text(mdates.date2num(deadline) - 2, 0.55, "November deadline",
            ha="right", fontsize=10.5, color=CLAY, weight="bold")
    ax.axvline(mdates.date2num(last), color=MUTED, ls=":", lw=1.4, zorder=4)
    ax.text(mdates.date2num(last) - 2, 5.75, "today", ha="right",
            fontsize=9.5, color=MUTED)

    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(0.3, 6.1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlim(mdates.date2num(date(2026, 8, 1)),
                mdates.date2num(date(2026, 12, 10)))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
