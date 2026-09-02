"""fig 06 — TFD entanglement entropy vs β. [DATA] half-slide.

Recomputes rung 2's two-level model (E₀=0, E₁=1) and asserts agreement with
the values committed in rung2_results.txt before plotting.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=6, name="tfd_entropy_vs_beta", slide=11, type="data",
            data_source="rung2 two-level TFD recomputed; checked against "
                        "rung2_results.txt table",
            caption="Entanglement entropy of the two-level thermofield double "
                    "across temperature: a smooth crossover, not a transition.")


def S_of_beta(beta):
    E = np.array([0.0, 1.0])
    w = np.exp(-np.outer(beta, E))
    p = w / w.sum(axis=1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1)


def build():
    # regression against the committed table (rung2_results.txt section 4)
    txt = data.results_text("rung2")
    checked = []
    for line in txt.splitlines():
        m = re.match(r"\s*([\d.]+) \|\s*[\d.]+ \|\s*[\d.]+ \|\s*[\d.e-]+ \|"
                     r"\s*([\d.]+) \|", line)
        if m:
            b, s_committed = float(m.group(1)), float(m.group(2))
            s_now = float(S_of_beta(np.array([b]))[0])
            assert abs(s_now - s_committed) < 5e-6, (
                f"recomputed S(A) at beta={b} is {s_now:.6f}, committed "
                f"{s_committed:.6f} — model mismatch, not a plotting bug")
            checked.append((b, s_committed))

    beta = np.geomspace(0.01, 20.0, 600)
    S = S_of_beta(beta)

    fig, ax = new_fig("half")
    ax.plot(beta, S, color=NAVY, zorder=3)
    ax.axhline(np.log(2), color=MUTED, ls="--", lw=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("β")
    ax.set_ylabel("S(A)   (nats)")
    ax.set_ylim(0, 0.79)
    note(ax, (0.028, 0.725), "log 2", color=MUTED, fs=10, ha="left")
    ax.plot([1.0], [float(S_of_beta(np.array([1.0]))[0])], "o", color=AMBER,
            zorder=4)
    ax.annotate("crossover at β ≈ 1\nsmooth — not a phase transition\n"
                "(Betti numbers never change)",
                xy=(1.0, 0.582), xytext=(0.03, 0.30),
                textcoords="axes fraction", ha="left", va="center",
                fontsize=10, color=AMBER,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.4))

    provenance(META["id"],
               f"two-level TFD (E=0,1) recomputed on 600 log-spaced β in "
               f"[0.01,20]; {len(checked)} committed values from "
               f"rung2_results.txt reproduced to <5e-6 "
               f"(e.g. β=1 → S={S_of_beta(np.array([1.0]))[0]:.5f})")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
