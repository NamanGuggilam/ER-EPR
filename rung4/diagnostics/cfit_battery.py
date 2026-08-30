"""Final rung-4 computation battery. Numbers only.

1. Ghat_Schwarzian(1) vs beta/C sweep at ALL four betas (per-cell
   reachability restatement), with a collapse check across beta at equal
   beta/C (Check D's scaling).
2. Ghat_SYK(1) and Ghat_SYK(8) for all 16 cells from cache (mean +- std),
   confirmation of the user's inverted-W_H0 reconstruction for N=18, and
   the beta where SYK crosses the conformal value, per N.
3. C_fit = argmin_C W_H0 per cell over C in (0, 200]:
   (a) C_fit, residual, crossed-vs-floored; monotone-to-boundary => NO FIT
   (b) spread of C_fit across beta at fixed N
   (c) power-law fit C_fit = a*N^b (weighted, with uncertainties)
   (d) C_fit / C_declared
   (e) same using Ghat(8); ratio C_fit(G1)/C_fit(G8) per cell
4. Raw scalars per cell: 1/Ghat(1), 1/Ghat(8) for SYK and for the
   Schwarzian at C_fit.
"""
import pickle
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, "/Users/naman/projects/ER-EPR")
sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")

from rung4 import declared_C, BETA_SWEEP
from schwarzian import compute_G

CACHE = "/private/tmp/claude-501/-Users-naman/d2153a54-68ef-4744-a61e-58afea02c44d/scratchpad/profiles_cache.pkl"
N_VALUES = [12, 14, 16, 18]
BETAS = list(BETA_SWEEP)          # [5, 10, 20, 40]
N_TAU = 24
DELTA = 0.25
CONF_G1 = float(np.sin(np.pi / N_TAU) ** (-2 * DELTA))          # 2.767905
CONF_G8 = float(np.sin(np.pi * 8 / N_TAU) ** (-2 * DELTA))      # 1.074570

with open(CACHE, "rb") as fh:
    cache = pickle.load(fh)
syk_G = cache["syk_G"]

MEMO_PATH = CACHE.replace("profiles_cache", "schG_memo")
try:
    with open(MEMO_PATH, "rb") as fh:
        _memo = pickle.load(fh)
except FileNotFoundError:
    _memo = {}


class ConvError(Exception):
    pass


def save_memo():
    with open(MEMO_PATH, "wb") as fh:
        pickle.dump(_memo, fh)


def sch_G(tau, beta, C):
    key = (round(tau, 12), beta, round(C, 12))
    if key not in _memo:
        res = compute_G(tau, beta, C)
        if res["conv_err"] >= 1e-6:
            raise ConvError(f"conv_err {res['conv_err']:.1e} at tau={tau}, "
                            f"beta={beta}, C={C}")
        _memo[key] = res["G"]
        if len(_memo) % 25 == 0:
            save_memo()
    return _memo[key]


def sch_ghat(m, beta, C):
    return sch_G(m * beta / N_TAU, beta, C) / sch_G(beta / 2, beta, C)


# ------------------------------------------------------------------ part 2
print("=== 2. Ghat_SYK(1) and Ghat_SYK(8), all 16 cells (mean +- std over "
      "10 realizations) ===")
print(f"conformal references: Ghat(1) = {CONF_G1:.6f}, Ghat(8) = {CONF_G8:.6f}")
g1 = {}   # (N, beta) -> (mean, std) of Ghat_syk(1)
g8 = {}
for N in N_VALUES:
    for bi, beta in enumerate(BETAS):
        prof = syk_G[N][:, bi, :]                 # (R, 12)
        gh1 = prof[:, 0] / prof[:, -1]
        gh8 = prof[:, 7] / prof[:, -1]
        g1[(N, beta)] = (float(gh1.mean()), float(gh1.std()))
        g8[(N, beta)] = (float(gh8.mean()), float(gh8.std()))
        print(f"  N={N} beta={beta:>4g}: Ghat(1) = {gh1.mean():.4f} +- "
              f"{gh1.std():.4f}   Ghat(8) = {gh8.mean():.4f} +- {gh8.std():.4f}")

print("\nuser reconstruction check (N=18): "
      + ", ".join(f"{g1[(18, b)][0]:.3f}" for b in BETAS)
      + "  (claimed 1.303, 1.729, 2.474, 3.548)")

print("\ncrossing of conformal Ghat(1) = 2.76790, per N (linear in ln beta "
      "between bracketing betas):")
for N in N_VALUES:
    vals = [g1[(N, b)][0] for b in BETAS]
    cross = None
    for i in range(3):
        if (vals[i] - CONF_G1) * (vals[i + 1] - CONF_G1) < 0:
            lb1, lb2 = np.log(BETAS[i]), np.log(BETAS[i + 1])
            t = (CONF_G1 - vals[i]) / (vals[i + 1] - vals[i])
            cross = float(np.exp(lb1 + t * (lb2 - lb1)))
            print(f"  N={N}: crosses between beta={BETAS[i]:g} "
                  f"({vals[i]:.3f}) and beta={BETAS[i+1]:g} ({vals[i+1]:.3f}); "
                  f"interpolated beta_cross ~= {cross:.1f}")
    if cross is None:
        print(f"  N={N}: no crossing in the sweep range")

# ------------------------------------------------------------------ part 1
print("\n=== 1. Ghat_Schwarzian(1) vs beta/C at beta = 5, 10, 20, 40 ===")
XS = [0.05, 0.25, 1.0, 5.0, 25.0, 100.0, 300.0, 1000.0]   # beta/C
tab = {}
for beta in BETAS:
    for x in XS:
        C = beta / x
        if C > 200.0 or C <= 0:
            continue
        tab[(beta, x)] = sch_ghat(1, beta, C)
        print(f"  beta={beta:>4g} beta/C={x:>7g} (C={C:.4g}): "
              f"Ghat(1) = {tab[(beta, x)]:.6f}", flush=True)
print(f"{'beta/C':>8} | " + " | ".join(f"beta={b:>3g}" for b in BETAS))
for x in XS:
    row = [f"{tab[(b, x)]:8.5f}" if (b, x) in tab else "     out"
           for b in BETAS]
    print(f"{x:>8g} | " + " | ".join(row))
spread = max(np.ptp([tab[(b, x)] for b in BETAS if (b, x) in tab])
             for x in XS if sum((b, x) in tab for b in BETAS) > 1)
print(f"collapse check: max spread across beta at equal beta/C = {spread:.2e}")
mono = all(all(tab[(b, XS[i])] < tab[(b, XS[i + 1])]
               for i in range(len(XS) - 1)
               if (b, XS[i]) in tab and (b, XS[i + 1]) in tab)
           for b in BETAS)
minv = min(tab.values())
print(f"monotone increasing in beta/C at every beta: {mono}; "
      f"min over sweep = {minv:.6f} (conformal {CONF_G1:.6f})")
print("per-cell reachability (target = SYK mean Ghat(1); family range at "
      "each beta = (conformal, Ghat at C -> 0)):")
for N in N_VALUES:
    for beta in BETAS:
        t = g1[(N, beta)][0]
        print(f"  N={N} beta={beta:>4g}: SYK {t:.3f} "
              f"{'INSIDE family range (fit possible)' if t > CONF_G1 else 'BELOW conformal infimum (unreachable)'}")

# ------------------------------------------------------------------ part 3
print("\n=== 3. C_fit = argmin_C W_H0, C in (0, 200] ===", flush=True)
C_HI = 200.0


def fit_C(beta, target_f, m):
    """Root of 1/Ghat_sch(m) = target_f in C. f is increasing in C.
    Lower bracket starts at beta/C = 2000 and backs off (C doubled) on
    quadrature convergence failure. Returns (C_fit, kind)."""
    def g(lnC):
        return 1.0 / sch_ghat(m, beta, float(np.exp(lnC))) - target_f
    ghi = g(np.log(C_HI))
    if ghi < 0:
        return None, "NO FIT (monotone toward C=200 boundary)"
    c_lo = beta / 2000.0
    while c_lo < C_HI:
        try:
            glo = g(np.log(c_lo))
            break
        except ConvError:
            c_lo *= 2.0
    else:
        return None, "NO FIT (no convergent lower bracket)"
    if glo > 0:
        return None, (f"NO FIT (monotone toward small-C boundary; lowest "
                      f"evaluable C = {c_lo:.4g}, beta/C = {beta/c_lo:.0f})")
    lnc = brentq(g, np.log(c_lo), np.log(C_HI), xtol=1e-6, rtol=1e-12)
    return float(np.exp(lnc)), "crossed"


rows = {}
for N in N_VALUES:
    for beta in BETAS:
        f1m, f1s = 1.0 / g1[(N, beta)][0], None
        # SEM of f1 via per-realization 1/Ghat(1)
        bi = BETAS.index(beta)
        prof = syk_G[N][:, bi, :]
        f1_r = prof[:, -1] / prof[:, 0]
        f8_r = prof[:, -1] / prof[:, 7]
        f1m, f1sem = float(f1_r.mean()), float(f1_r.std() / np.sqrt(len(f1_r)))
        f8m, f8sem = float(f8_r.mean()), float(f8_r.std() / np.sqrt(len(f8_r)))
        C1, kind1 = fit_C(beta, f1m, 1)
        C8, kind8 = fit_C(beta, f8m, 8)
        res1 = (23.0 * abs(1.0 / sch_ghat(1, beta, C1) - f1m)
                if C1 else None)
        res8 = (23.0 * abs(1.0 / sch_ghat(8, beta, C8) - f8m)
                if C8 else None)
        sC1 = None
        if C1:
            df = (1.0 / sch_ghat(1, beta, C1 * 1.02)
                  - 1.0 / sch_ghat(1, beta, C1)) / (0.02 * C1)
            sC1 = abs(f1sem / df)
        rows[(N, beta)] = dict(C1=C1, kind1=kind1, res1=res1, sC1=sC1,
                               C8=C8, kind8=kind8, res8=res8,
                               f1m=f1m, f8m=f8m)
        c1s = (f"{C1:.4f} +- {sC1:.4f} (resid W_H0 {res1:.2e}, crossed)"
               if C1 else kind1)
        c8s = (f"{C8:.4f} (resid W_H0 {res8:.2e}, crossed)"
               if C8 else kind8)
        print(f"  N={N} beta={beta:>4g}: C_fit[Ghat1] = {c1s}", flush=True)
        print(f"                C_fit[Ghat8] = {c8s}", flush=True)

print("\n(b) C_fit spread across beta at fixed N:")
for N in N_VALUES:
    cs = [(beta, rows[(N, beta)]["C1"]) for beta in BETAS
          if rows[(N, beta)]["C1"]]
    if len(cs) <= 1:
        print(f"  N={N}: C_fit exists at "
              f"{len(cs)} beta value(s) ({[b for b, _ in cs]}); "
              f"spread across beta NOT DEFINED")
    else:
        vals = [c for _, c in cs]
        print(f"  N={N}: betas {[b for b, _ in cs]}, C_fit {vals}, "
              f"spread {max(vals)-min(vals):.4f} "
              f"({(max(vals)-min(vals))/np.mean(vals)*100:.1f}% of mean)")

print("\n(c) power law C_fit = a*N^b over cells with a fit [Ghat1 target]:")
pts = [(N, beta, rows[(N, beta)]["C1"], rows[(N, beta)]["sC1"])
       for N in N_VALUES for beta in BETAS if rows[(N, beta)]["C1"]]
if len(pts) >= 3:
    lnN = np.array([np.log(p[0]) for p in pts])
    lnC = np.array([np.log(p[2]) for p in pts])
    w = np.array([(p[2] / p[3]) ** 2 if p[3] and p[3] > 0 else 1.0
                  for p in pts])     # weight = 1/sigma_lnC^2
    A = np.vstack([np.ones_like(lnN), lnN]).T
    W = np.diag(w)
    cov = np.linalg.inv(A.T @ W @ A)
    coef = cov @ (A.T @ W @ lnC)
    resid = lnC - A @ coef
    chi2 = float(resid @ W @ resid)
    dof = len(pts) - 2
    scale = max(chi2 / dof, 1.0) if dof > 0 else 1.0
    err = np.sqrt(np.diag(cov) * scale)
    a, b = float(np.exp(coef[0])), float(coef[1])
    print(f"  points: {[(p[0], p[1], round(p[2], 4)) for p in pts]}")
    print(f"  b = {b:.3f} +- {err[1]:.3f}   a = {a:.5f} "
          f"(*/ {np.exp(err[0]):.3f})   [MS predicts b=1, a=0.01003]")
    print(f"  chi2/dof = {chi2:.1f}/{dof}")
else:
    print(f"  only {len(pts)} fitted cells — power law not fit")

print("\n(d) C_fit / C_declared per fitted cell [Ghat1]:")
for N, beta, C, s in pts:
    print(f"  N={N} beta={beta:g}: C_fit/C_decl = {C/declared_C(N):.3f}")

print("\n(e) C_fit(Ghat1)/C_fit(Ghat8) per cell where both exist:")
any8 = False
for N in N_VALUES:
    for beta in BETAS:
        r = rows[(N, beta)]
        if r["C1"] and r["C8"]:
            any8 = True
            print(f"  N={N} beta={beta:g}: {r['C1']:.4f}/{r['C8']:.4f} "
                  f"= {r['C1']/r['C8']:.3f}")
if not any8:
    print("  no cell has both fits")

# ------------------------------------------------------------------ part 4
print("\n=== 4. RAW SCALARS PER CELL: 1/Ghat(1), 1/Ghat(8) — SYK | "
      "Schwarzian at C_fit[Ghat1] ===")
for N in N_VALUES:
    for beta in BETAS:
        r = rows[(N, beta)]
        s = f"  N={N} beta={beta:>4g}: SYK ({r['f1m']:.6f}, {r['f8m']:.6f})"
        if r["C1"]:
            e1 = 1.0 / sch_ghat(1, beta, r["C1"])
            e8 = 1.0 / sch_ghat(8, beta, r["C1"])
            s += f" | Sch@C_fit={r['C1']:.4f} ({e1:.6f}, {e8:.6f})"
        else:
            s += " | Sch: NO FIT"
        print(s)

save_memo()
