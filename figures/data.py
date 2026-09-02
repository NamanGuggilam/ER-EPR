"""Data access for [DATA] figures.

Every loader either returns real numbers from this repo or raises MissingData
with an explicit statement of what is missing and how to regenerate it.
Nothing here invents, interpolates, or substitutes values.
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNG4 = REPO / "rung4"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(RUNG4))


class MissingData(RuntimeError):
    """Raised when a [DATA] figure's source does not exist."""


def _need(path, how):
    if not Path(path).exists():
        raise MissingData(f"MISSING: {path}\n  regenerate with: {how}")
    return Path(path)


# --- rung 4 caches -------------------------------------------------------
def profiles():
    """{'syk_G': {N: (R, nbeta, nsep)}, 'er_G': {(N,beta,tag): prof}, ...}"""
    p = _need(RUNG4 / "profiles_cache.pkl",
              ".venv/bin/python rung4/diagnostics/split_wasserstein.py")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def diagrams():
    """{('syk',N,beta,r) | ('er',N,beta,tag): [H0, H1]} — 224 entries."""
    p = _need(RUNG4 / "diagram_cache.pkl",
              ".venv/bin/python rung4/diagnostics/split_wasserstein.py")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def schwarzian_memo():
    """{(tau, beta, C): G} memo from the C_fit battery."""
    p = _need(RUNG4 / "schG_memo.pkl",
              ".venv/bin/python rung4/diagnostics/cfit_battery.py")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def spectra(N):
    """Cached SYK eigenvalue array, shape (R, dim)."""
    hits = sorted(RUNG4.glob(f"data/spectra_N{N}_R*.npz"))
    if not hits:
        raise MissingData(
            f"MISSING: rung4/data/spectra_N{N}_R*.npz\n"
            f"  regenerate with: .venv/bin/python rung4/rung4.py extract --N={N}")
    with np.load(hits[0]) as z:
        return z["spectra"], hits[0].name


# --- committed results files --------------------------------------------
def results_text(which):
    """Verbatim text of a committed results file."""
    names = {"rung1": REPO / "rung1_results.txt",
             "rung2": REPO / "rung2_results.txt",
             "rung3": REPO / "rung3_results.txt",
             "rung4": RUNG4 / "rung4_results.txt",
             "checks": RUNG4 / "rung4_checks_results.txt"}
    p = _need(names[which], f"rerun the {which} stage")
    return p.read_text()


# --- parsed tables from the committed record ----------------------------
BETAS = [5.0, 10.0, 20.0, 40.0]
N_VALUES = [12, 14, 16, 18]
N_TAU = 24
DELTA = 0.25
ALPHA_S = 0.00709
CONF_G1 = float(np.sin(np.pi / N_TAU) ** (-2 * DELTA))       # 2.767905
CONF_G8 = float(np.sin(np.pi * 8 / N_TAU) ** (-2 * DELTA))   # 1.074570


def declared_C(N):
    return ALPHA_S * N / (1.0 / np.sqrt(2.0))


def split_table():
    """The H0/H1 split table (per cell, per tag) from the committed
    diagnostics output. Returns {(N,beta): {'W0': {...}, 'W1': {...},
    'boc': float}}."""
    p = _need(RUNG4 / "diagnostics" / "split_out.txt",
              ".venv/bin/python rung4/diagnostics/split_wasserstein.py")
    rows = {}
    for line in p.read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s*\|" + r"\s*([-\d.]+)\s+([-\d.]+)\s*\|" * 4,
                     line)
        if m:
            g = m.groups()
            N, beta, boc = int(g[0]), float(g[1]), float(g[2])
            vals = [float(x) for x in g[3:]]
            tags = ["declared", "x5", "x0.2", "conformal"]
            rows[(N, beta)] = {
                "boc": boc,
                "W0": {t: vals[2 * i] for i, t in enumerate(tags)},
                "W1": {t: vals[2 * i + 1] for i, t in enumerate(tags)}}
    if len(rows) != 16:
        raise MissingData(
            f"split_out.txt parsed {len(rows)}/16 cells — format changed;\n"
            f"  regenerate with: .venv/bin/python rung4/diagnostics/split_wasserstein.py")
    return rows


def cfit_table():
    """C_fit results parsed from the committed battery output.
    {(N,beta): {'C1': float|None, 'C8': float|None, 'sC1': float|None}}"""
    p = _need(RUNG4 / "diagnostics" / "cfit_out2.txt",
              ".venv/bin/python rung4/diagnostics/cfit_battery.py")
    txt = p.read_text().splitlines()
    out, cur = {}, None
    for line in txt:
        m = re.match(r"\s*N=(\d+) beta=\s*(\d+): C_fit\[Ghat1\] = (.*)", line)
        if m:
            N, beta, rest = int(m.group(1)), float(m.group(2)), m.group(3)
            cur = (N, beta)
            out[cur] = {"C1": None, "sC1": None, "C8": None}
            v = re.match(r"([\d.]+) \+- ([\d.]+)", rest)
            if v:
                out[cur]["C1"] = float(v.group(1))
                out[cur]["sC1"] = float(v.group(2))
            continue
        m8 = re.match(r"\s*C_fit\[Ghat8\] = ([\d.]+) ", line)
        if m8 and cur:
            out[cur]["C8"] = float(m8.group(1))
    if len(out) != 16:
        raise MissingData(
            f"cfit_out2.txt parsed {len(out)}/16 cells — format changed;\n"
            f"  regenerate with: .venv/bin/python rung4/diagnostics/cfit_battery.py")
    return out


def uv_text():
    return _need(RUNG4 / "diagnostics" / "uv_out.txt",
                 ".venv/bin/python rung4/diagnostics/uv_diagnostics.py").read_text()


# --- derived quantities computed from the caches ------------------------
def ghat_syk(pr=None):
    """{(N,beta): (mean_g1, std_g1, mean_g8, std_g8)} from cached profiles."""
    pr = pr or profiles()
    out = {}
    for N in N_VALUES:
        for bi, beta in enumerate(BETAS):
            P = pr["syk_G"][N][:, bi, :]
            g1, g8 = P[:, 0] / P[:, -1], P[:, 7] / P[:, -1]
            out[(N, beta)] = (float(g1.mean()), float(g1.std()),
                              float(g8.mean()), float(g8.std()))
    return out


def ghat_profile(prof):
    """Ghat(m), m = 1..12, from a raw G profile."""
    return np.asarray(prof) / prof[-1]
