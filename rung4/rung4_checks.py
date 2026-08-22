"""
ER=EPR Project — Rung 4: Checks A-D validation suite for the Schwarzian
ER side (rung4/schwarzian.py)
========================================================================

Four checks, definitions fixed by the validated (pre-loss) rung-4 work.
Do not weaken or reinterpret them.

  Check A — free-energy calibration. Compare ln Z(beta) against the
    Schwarzian free-energy asymptotic
        ln Z ~ 2 pi^2 C / beta + 1.5 log(2 pi C / beta) + const
    across several beta at fixed C. Validates the FUNCTIONAL FORM of
    rho(k) = k sinh(2 pi k): a wrong power of k makes the residual
    DRIFT with beta rather than settle at a constant offset (the
    overall multiplicative constant in rho is irrelevant — it cancels
    in the barcode). PASS = residual settles toward a constant.
    Control columns with rho = k^0 sinh and k^2 sinh demonstrate the
    drift a wrong power would produce.

  Check B — conformal reduction (the SHAPE check, most important).
    As C -> infinity, G_Delta(tau) must reduce to the conformal form
        G_conf(tau) = [pi / (beta sin(pi tau / beta))]^(2 Delta).
    Compare the SHAPE only — G(tau)/G(tau_ref) against
    G_conf(tau)/G_conf(tau_ref) — because the two carry different
    overall normalization conventions (the absolute ratio approaches a
    constant, not 1). PASS = shape deviation shrinks as O(beta/C) at
    the predicted rate. Scoped to C <= 200: above that the quadrature
    fails, and this is a physics decision, not a shortcut — the
    deep-classical regime C >> 200 coincides with tree level anyway,
    so no new physics lives there.

  Check C — beta/C -> 0 barcode collapse. Build the shape-normalized
    Schwarzian distance matrix d = -log|G| on n points of the thermal
    circle, sweep C up toward 200 at fixed beta, and compute the
    Wasserstein distance between its H1 barcode and the tree-level
    frozen-circle (conformal) barcode at each C. PASS = Wasserstein
    shrinks MONOTONICALLY toward 0 as C grows.
    (Shape normalization used here, applied identically to both sides:
    d_ij = log(G_max / G_ij) with G_max the largest off-diagonal
    correlator, so the nearest-neighbor distance is 0 and all
    distances are >= 0; -log|G| alone is offset by the arbitrary
    normalization convention, which this removes.)

  Check D — is beta/C a real independent knob? Two tests:
    (1) same beta/C via different (beta, C) pairs at the same
        tau/beta -> G must come out EQUAL;
    (2) same tau/beta, different beta/C -> G must DIFFER.
    PASS = (1) equal within convergence error, (2) different far
    outside it. Confirms the ER side is genuinely temperature-
    dependent, not a frozen circle.

Usage:  python rung4/rung4_checks.py [A B C D]
        (default: A B — checks C and D are gated until A/B are approved)
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schwarzian import (  # noqa: E402
    C_MAX, DELTA_SYK, compute_G, compute_log_Z, k_star, log_sinh,
)

RESULTS_PATH = Path(__file__).resolve().parent / "rung4_checks_results.txt"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()  # long runs are watched live; don't sit in block buffers

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def g_conformal(tau, beta, delta=DELTA_SYK):
    """Tree-level conformal two-point function (shape only matters)."""
    return (np.pi / (beta * np.sin(np.pi * tau / beta))) ** (2.0 * delta)


# ======================================================================
# CHECK A — free-energy calibration
# ======================================================================
def _log_Z_power(p, beta, C, epsrel, limit):
    """log of Integral_0^inf dk k^p sinh(2 pi k) exp(-beta k^2/2C),
    same saddle-hint + max-subtraction treatment as compute_log_Z.
    p=1 is the true rho; p=0 and p=2 are the wrong-power controls."""
    kst = k_star(beta, C)
    kmax = kst + 40.0 * np.sqrt(C / beta) + 50.0
    kgrid = np.linspace(1e-9, kmax, 4001)

    def logf(k):
        k = np.asarray(k, dtype=float)
        with np.errstate(divide="ignore"):
            lp = np.where(k > 0, p * np.log(np.where(k > 0, k, 1.0)), -np.inf)
        return lp + log_sinh(2.0 * np.pi * k) - beta * k ** 2 / (2.0 * C)

    ref = float(np.max(logf(kgrid)))
    val, _ = quad(lambda k: np.exp(logf(k) - ref), 0.0, kmax,
                  points=[kst], epsabs=1e-14, epsrel=epsrel, limit=limit)
    return np.log(val) + ref


def check_A():
    section("CHECK A — free-energy calibration: ln Z vs Schwarzian asymptotic")
    C = 100.0
    betas = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]  # descending: C/beta grows
    print(f"C = {C}, beta sweep = {betas}  (C/beta from {C/betas[0]:.0f} "
          f"to {C/betas[-1]:.0f})")
    print("residual R(beta) = lnZ_num - [2 pi^2 C/beta + 1.5 log(2 pi C/beta)]")
    print("PASS = R settles toward a constant as C/beta grows; the wrong-power")
    print("controls (rho = k^0 sinh, k^2 sinh) drift instead.\n")

    rows = []
    for beta in betas:
        lnZ_f = compute_log_Z(beta, C, epsrel=1e-11, limit=400)
        lnZ_c = compute_log_Z(beta, C, epsrel=1e-8, limit=150)
        conv = abs(np.exp(lnZ_c - lnZ_f) - 1.0)
        asym = 2.0 * np.pi ** 2 * C / beta + 1.5 * np.log(2.0 * np.pi * C / beta)
        R = lnZ_f - asym
        R0 = _log_Z_power(0, beta, C, 1e-11, 400) - asym
        R2 = _log_Z_power(2, beta, C, 1e-11, 400) - asym
        rows.append((beta, lnZ_f, R, R0, R2, conv))

    print(f"{'beta':>6} | {'lnZ (fine)':>14} | {'R (rho=k sinh)':>14} | "
          f"{'R (k^0 ctrl)':>12} | {'R (k^2 ctrl)':>12} | {'conv_err':>9}")
    print("-" * 84)
    for beta, lnZ, R, R0, R2, conv in rows:
        print(f"{beta:6.3f} | {lnZ:14.5f} | {R:14.6f} | {R0:12.6f} | "
              f"{R2:12.6f} | {conv:9.2e}")

    R_vals = np.array([r[2] for r in rows])
    dR = np.abs(np.diff(R_vals))
    print(f"\nSuccessive |Delta R|, true rho: "
          + "  ".join(f"{d:.2e}" for d in dR))
    for tag, idx in (("k^0 control", 3), ("k^2 control", 4)):
        v = np.array([r[idx] for r in rows])
        print(f"Total drift over sweep, {tag}: {v[-1]-v[0]:+.4f} "
              f"(true rho: {R_vals[-1]-R_vals[0]:+.6f})")
    # "Settles toward a constant, doesn't drift": the pass test is on DRIFT
    # across the sweep, not on successive deltas shrinking -- for
    # rho = k sinh(2 pi k) the asymptotic is exact up to e^{-2 pi^2 C/beta}
    # half-line corrections (~1e-214 even at beta=4), so the residual is
    # already constant at machine precision and successive deltas are pure
    # float noise. Pass = true-rho drift negligible both absolutely and
    # relative to the drift the wrong-power controls exhibit.
    drift_true = abs(R_vals[-1] - R_vals[0])
    drift_ctrl = min(abs(rows[-1][i] - rows[0][i]) for i in (3, 4))
    settling = drift_true < 1e-3 and drift_true < 1e-2 * drift_ctrl
    print(f"\nCHECK A PASS (residual settles at a constant, controls drift): "
          f"{settling}  [true-rho drift {drift_true:.1e} vs control drift "
          f"{drift_ctrl:.2f}]")
    return settling


# ======================================================================
# CHECK B — conformal reduction (shape check)
# ======================================================================
def check_B():
    section("CHECK B — conformal reduction: shape of G vs tree level as C grows")
    beta = 1.0
    C_values = [25.0, 50.0, 100.0, 200.0]  # scope: C <= 200 (see docstring)
    tau_fracs = [0.05, 0.1, 0.2, 0.3, 0.4]
    tau_ref_frac = 0.5
    print(f"beta = {beta}, C sweep = {C_values}, tau/beta grid = {tau_fracs}, "
          f"tau_ref/beta = {tau_ref_frac}")
    print("deviation(C) = max_tau | [G(tau)/G(tau_ref)] / "
          "[G_conf(tau)/G_conf(tau_ref)] - 1 |")
    print("PASS = deviation shrinks as O(beta/C): ratio ~0.5 per C doubling.\n")

    devs, convs = [], []
    for C in C_values:
        t0 = time.time()
        res_ref = compute_G(tau_ref_frac * beta, beta, C)
        max_conv = res_ref["conv_err"]
        dev_C = 0.0
        for f in tau_fracs:
            res = compute_G(f * beta, beta, C)
            max_conv = max(max_conv, res["conv_err"])
            shape_sch = res["G"] / res_ref["G"]
            shape_conf = (g_conformal(f * beta, beta)
                          / g_conformal(tau_ref_frac * beta, beta))
            dev = abs(shape_sch / shape_conf - 1.0)
            dev_C = max(dev_C, dev)
            print(f"  C={C:6.1f}  tau/beta={f:4.2f}: shape_Sch={shape_sch:12.8f}"
                  f"  shape_conf={shape_conf:12.8f}  |ratio-1|={dev:.3e}"
                  f"  conv_err={res['conv_err']:.1e}")
        devs.append(dev_C)
        convs.append(max_conv)
        print(f"  C={C:6.1f}: deviation = {dev_C:.4e}, max conv_err = "
              f"{max_conv:.1e}  [{time.time()-t0:.0f}s]")

    print(f"\n{'C':>6} | {'deviation':>11} | {'ratio to prev':>13} | "
          f"{'C*deviation':>11} | {'max conv_err':>12}")
    print("-" * 66)
    ratios = []
    for i, C in enumerate(C_values):
        r = devs[i] / devs[i - 1] if i else float("nan")
        if i:
            ratios.append(r)
        print(f"{C:6.1f} | {devs[i]:11.4e} | {r:13.3f} | "
              f"{C * devs[i]:11.4e} | {convs[i]:12.1e}")

    # O(beta/C) rate: log-log slope of deviation vs 1/C should be ~ +1
    slope = float(np.polyfit(np.log(1.0 / np.array(C_values)),
                             np.log(np.array(devs)), 1)[0])
    print(f"\nlog-log slope of deviation vs beta/C: {slope:.3f}  "
          f"(O(beta/C) prediction: 1; per-doubling ratios: "
          + ", ".join(f"{r:.3f}" for r in ratios) + ")")
    ok = bool(np.all(np.diff(devs) < 0)) and 0.7 < slope < 1.3 \
        and max(convs) < 1e-6
    print(f"CHECK B PASS (monotone shrink at O(beta/C) rate, converged): {ok}")
    return ok


# ======================================================================
# CHECK C — beta/C -> 0 barcode collapse            [GATED: not run yet]
# ======================================================================
def check_C():
    section("CHECK C — barcode collapse onto the tree-level frozen circle")
    from ripser import ripser
    from persim import wasserstein

    beta = 1.0
    C_values = [25.0, 50.0, 100.0, 200.0]
    n_circle = 16  # points on the thermal circle; separations j*beta/n, j=1..n/2
    print(f"beta = {beta}, C sweep = {C_values}, {n_circle} points on the "
          f"thermal circle")
    print("d_ij = log(G_max / G(|tau_i - tau_j|_circle))  [shape-normalized, "
          "same prescription both sides]")

    seps = [j * beta / n_circle for j in range(1, n_circle // 2 + 1)]

    def distance_matrix(g_of_sep):
        g = {j + 1: g_of_sep(s) for j, s in enumerate(seps)}
        gmax = max(g.values())
        D = np.zeros((n_circle, n_circle))
        for i in range(n_circle):
            for k in range(i + 1, n_circle):
                j = min(k - i, n_circle - (k - i))
                D[i, k] = D[k, i] = np.log(gmax / g[j])
        return D

    D_tree = distance_matrix(lambda s: g_conformal(s, beta))
    h1_tree = ripser(D_tree, distance_matrix=True, maxdim=1)["dgms"][1]
    h1_tree = h1_tree[np.isfinite(h1_tree[:, 1])]
    print(f"tree-level frozen-circle H1 barcode: {h1_tree.tolist()}")

    w_prev, monotone = None, True
    for C in C_values:
        cache = {}

        def g_sch(s, C=C, cache=cache):
            if s not in cache:
                cache[s] = compute_G(s, beta, C)
            return cache[s]["G"]

        D = distance_matrix(g_sch)
        max_conv = max(v["conv_err"] for v in cache.values())
        h1 = ripser(D, distance_matrix=True, maxdim=1)["dgms"][1]
        h1 = h1[np.isfinite(h1[:, 1])]
        w = float(wasserstein(h1, h1_tree))
        print(f"  C={C:6.1f}: H1 bars={len(h1)}, Wasserstein to tree = "
              f"{w:.6e}, max conv_err = {max_conv:.1e}")
        if w_prev is not None and w >= w_prev:
            monotone = False
        w_prev = w
    print(f"CHECK C PASS (Wasserstein shrinks monotonically toward 0): "
          f"{monotone}")
    return monotone


# ======================================================================
# CHECK D — beta/C as the real independent knob      [GATED: not run yet]
# ======================================================================
def check_D():
    section("CHECK D — is beta/C a real independent knob?")
    tau_frac = 0.2

    # Test 1: same beta/C (=1/50), same tau/beta, different (beta, C) -> EQUAL
    pairs_same = [(1.0, 50.0), (2.0, 100.0), (4.0, 200.0)]
    print(f"Test 1 — same beta/C = {pairs_same[0][0]/pairs_same[0][1]}, "
          f"tau/beta = {tau_frac}, different (beta, C):")
    vals = []
    for beta, C in pairs_same:
        res = compute_G(tau_frac * beta, beta, C)
        vals.append(res)
        print(f"  (beta={beta}, C={C}): G = {res['G']:.12e}  "
              f"conv_err = {res['conv_err']:.1e}")
    rel_spread = max(abs(v["G"] / vals[0]["G"] - 1.0) for v in vals[1:])
    tol = 100.0 * max(max(v["conv_err"] for v in vals), 1e-12)
    equal_ok = rel_spread < max(tol, 1e-8)
    print(f"  max relative spread = {rel_spread:.2e}  (equal: {equal_ok})")

    # Test 2: same tau/beta, different beta/C -> DIFFER
    pairs_diff = [(1.0, 50.0), (1.0, 100.0), (1.0, 200.0)]
    print(f"\nTest 2 — same tau/beta = {tau_frac}, different beta/C:")
    vals2 = []
    for beta, C in pairs_diff:
        res = compute_G(tau_frac * beta, beta, C)
        vals2.append(res)
        print(f"  (beta={beta}, C={C}, beta/C={beta/C:.3f}): "
              f"G = {res['G']:.12e}  conv_err = {res['conv_err']:.1e}")
    min_diff = min(abs(vals2[i]["G"] / vals2[0]["G"] - 1.0)
                   for i in range(1, len(vals2)))
    differ_ok = min_diff > 1e-3
    print(f"  min relative difference vs first = {min_diff:.2e}  "
          f"(differ: {differ_ok})")

    ok = equal_ok and differ_ok
    print(f"CHECK D PASS (test 1 equal, test 2 different): {ok}")
    return ok


# ======================================================================
if __name__ == "__main__":
    requested = [a.upper() for a in sys.argv[1:]] or ["A", "B"]
    checks = {"A": check_A, "B": check_B, "C": check_C, "D": check_D}
    unknown = [a for a in requested if a not in checks]
    if unknown:
        sys.exit(f"unknown check(s): {unknown}; choose from A B C D")

    log_file = open(RESULTS_PATH, "a")
    _real_stdout = sys.stdout
    sys.stdout = Tee(_real_stdout, log_file)
    print(f"\n############ rung4_checks run: {' '.join(requested)} "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')}) ############")

    t0 = time.time()
    outcomes = {name: checks[name]() for name in requested}

    section("OUTCOMES")
    for name, ok in outcomes.items():
        print(f"  Check {name}: {'PASS' if ok else 'FAIL'}")
    print(f"Total time: {time.time()-t0:.0f}s")
    print(f"Results appended to: {RESULTS_PATH}")

    sys.stdout = _real_stdout
    log_file.close()
