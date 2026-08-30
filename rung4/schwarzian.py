"""
ER=EPR Project — Rung 4, ER side: exact Schwarzian two-point function
=======================================================================

Rung 3's ER side was a classical, frozen JT time-slice — 1-dimensional,
temperature-trivial, provably H1-free. Rung 4 replaces it with the
quantity that actually carries the JT gravity dynamics: the boundary
Euclidean two-point function computed from the exact Schwarzian theory
(Mertens–Turiaci–Verlinde, arXiv:1705.08408):

    G_Delta(tau; beta, C)
      = (1/Z) * Integral dk1 dk2  rho(k1) rho(k2)
            * exp( -tau k1^2 / 2C  -  (beta - tau) k2^2 / 2C )
            * |Gamma(Delta + i(k1+k2))|^2 |Gamma(Delta + i(k1-k2))|^2
            / Gamma(2 Delta)

    rho(k)  = k * sinh(2 pi k)
    Z(beta) = Integral dk rho(k) exp(-beta k^2 / 2C)
    Delta   = 1/4  (the SYK4 fermion dimension)

with k1, k2 integrated over (0, inf). C is the Schwarzian coupling.

NORMALIZATION SCOPE (stated honestly): the tau-dependence above is
MTV's exact result (their eqs 1.15-1.16 at C = 1/2, generalized by
E = k^2/2C), but the OVERALL tau-independent constant is convention-
dependent (MTV's measure is dk^2 sinh(2 pi k) = 2 k sinh(2 pi k) dk
vs rho = k sinh(2 pi k) here, and bilocal-operator normalizations
vary across the literature). Every rung-4 use of G is shape-gauged
(Ghat = G/G(beta/2), log-differences, or ratios), so only the shape
is load-bearing; absolute G values (e.g. the standing regression
gate 3.980e+02) are anchors of THIS implementation, not
literature-normalized numbers.

Implementation notes — every one of these was a real failure mode:

  * LOG SPACE THROUGHOUT. Individual factors (sinh(2 pi k) at the
    saddle k ~ 2 pi C / beta, the Gamma moduli) overflow float64 by
    thousands of orders of magnitude. All factors are assembled as
    log-integrands (scipy.special.loggamma; log sinh x = x - log 2
    + log1p(-exp(-2x))), a global reference max is subtracted before
    exponentiation, and it is added back only at the log-result level.

  * (s, d) ROTATION + JACOBIAN. The Gamma factors couple k1, k2 only
    through s = k1 + k2 and d = k1 - k2, so the integral is done in
    (s, d) coordinates: s in (0, inf), d in (-s, s). dk1 dk2 =
    (1/2) ds dd, so the log-integrand carries LOG_JACOBIAN = -log(2).
    Omitting the Jacobian was a real bug (a silent factor-2 error).

  * PEAK HINTS ON EVERY QUAD CALL. At large C the integrand is a very
    narrow peak far from the origin; QUADPACK silently misses it and
    falsely reports convergence. Every quad call — the Z integral AND
    both levels of the G integral — gets an explicit points=[...] hint
    at the closed-form saddle: k* = 2 pi C / beta for Z, and
    s* = 4 pi C / beta for the outer G integral (the inner d integral
    gets its own stationary-point hints, clipped into (-s, s)).
    points= requires finite limits, so the infinite ranges are cut off
    at saddle + 40 Gaussian widths + 50 (the Gamma moduli also decay
    like exp(-pi |arg|), so this cutoff is vastly conservative).

  * SELF-REPORTED CONVERGENCE. Every compute_G call runs the full
    pipeline at two quadrature resolutions (coarse and fine) and
    reports their relative difference as conv_err. Do not use a value
    whose conv_err you have not looked at.

  * VALIDATED SCOPE: C <= 200 ONLY. Above C ~ 200 QUADPACK fails even
    with peak hints; below, this implementation cross-validates to
    ~1e-7 against a dense-grid quadrature. compute_G raises on
    C > C_MAX rather than returning silently wrong numbers.
"""

import warnings

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import loggamma

DELTA_SYK = 0.25          # Delta = 1/4, the SYK4 fermion dimension
C_MAX = 200.0             # validated scope; QUADPACK fails above this
LOG_JACOBIAN = -np.log(2.0)   # dk1 dk2 = (1/2) ds dd
_LOG2 = np.log(2.0)


# ----------------------------------------------------------------------
# Log-space building blocks
# ----------------------------------------------------------------------
def log_sinh(x):
    """log(sinh(x)) for x >= 0 without overflow: x - log2 + log1p(-e^{-2x}).
    For x -> 0 falls back to log(x) (relative error O(x^2))."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-6
    xs = np.where(small, 1.0, x)  # dummy to keep log1p/exp in-range
    with np.errstate(divide="ignore"):
        out = np.where(
            small,
            np.log(np.where(x > 0, x, 1.0)) + np.where(x > 0, 0.0, -np.inf),
            xs - _LOG2 + np.log1p(-np.exp(-2.0 * xs)),
        )
    return out


def log_rho(k):
    """log rho(k) = log(k sinh(2 pi k)); -inf for k <= 0 (integrand vanishes)."""
    k = np.asarray(k, dtype=float)
    pos = k > 0
    kk = np.where(pos, k, 1.0)
    with np.errstate(divide="ignore"):
        out = np.where(pos, np.log(kk) + log_sinh(2.0 * np.pi * kk), -np.inf)
    return out


def log_gamma_abs2(delta, x):
    """log |Gamma(delta + i x)|^2 = 2 Re loggamma(delta + i x)."""
    return 2.0 * np.real(loggamma(delta + 1j * np.asarray(x, dtype=float)))


def k_star(beta, C):
    """Closed-form saddle of the Z integrand: rho(k) e^{-beta k^2/2C}
    peaks (for C/beta >> 1) where 2 pi = beta k / C."""
    return 2.0 * np.pi * C / beta


def s_star(beta, C):
    """Closed-form saddle of the outer G integrand: s* = 4 pi C / beta
    (= 2 k*, since the peak sits near k1 = k2 = k*)."""
    return 4.0 * np.pi * C / beta


def _validate(tau, beta, C):
    if not (0.0 < C <= C_MAX):
        raise ValueError(
            f"C={C} outside validated scope (0, {C_MAX}]: QUADPACK fails "
            f"above C={C_MAX} even with peak hints (see module docstring)."
        )
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if not (0.0 < tau < beta):
        raise ValueError(f"need 0 < tau < beta, got tau={tau}, beta={beta}")


# ----------------------------------------------------------------------
# Log-integrands
# ----------------------------------------------------------------------
def log_z_integrand(k, beta, C):
    return log_rho(k) - beta * np.asarray(k, dtype=float) ** 2 / (2.0 * C)


def log_g_integrand(s, d, tau, beta, C, delta):
    """Log of the FULL G-numerator integrand in (s, d) coordinates,
    including the 1/Gamma(2 Delta) normalization and the (s,d) Jacobian."""
    s = np.asarray(s, dtype=float)
    d = np.asarray(d, dtype=float)
    k1 = 0.5 * (s + d)
    k2 = 0.5 * (s - d)
    return (
        log_rho(k1)
        + log_rho(k2)
        - tau * k1 ** 2 / (2.0 * C)
        - (beta - tau) * k2 ** 2 / (2.0 * C)
        + log_gamma_abs2(delta, s)
        + log_gamma_abs2(delta, d)
        - float(np.real(loggamma(2.0 * delta)))
        + LOG_JACOBIAN
    )


def _inner_d_hints(s, tau, beta, C):
    """Stationary-point hints for the inner d integral at fixed s.
    The Gaussian part alone peaks at d = s (beta - 2 tau) / beta; the
    exp(-pi |d|) tail of |Gamma(Delta + i d)|^2 shifts the true peak by
    -+ 4 pi C / beta. Offer all candidates (plus d = 0, where the Gamma
    modulus itself peaks), clipped strictly inside (-s, s)."""
    d_gauss = s * (beta - 2.0 * tau) / beta
    shift = 4.0 * np.pi * C / beta
    cands = (0.0, d_gauss, d_gauss - shift, d_gauss + shift)
    eps = 1e-12 * max(s, 1.0)
    return sorted({d for d in cands if -s + eps < d < s - eps})


# ----------------------------------------------------------------------
# Reference maxima for max-subtraction (coarse vectorized grids)
# ----------------------------------------------------------------------
def _ref_max_z(beta, C, k_max, n=4001):
    k = np.linspace(1e-9, k_max, n)
    return float(np.max(log_z_integrand(k, beta, C)))


def _ref_max_g(tau, beta, C, delta, s_max, ns=601, nd=401):
    s = np.linspace(1e-9, s_max, ns)[:, None]
    u = np.linspace(-1.0 + 1e-9, 1.0 - 1e-9, nd)[None, :]
    vals = log_g_integrand(np.broadcast_to(s, (ns, nd)), s * u,
                           tau, beta, C, delta)
    return float(np.max(vals))


# ----------------------------------------------------------------------
# The integrals
# ----------------------------------------------------------------------
def compute_log_Z(beta, C, epsrel=1e-11, limit=400):
    """log Z(beta) = log Integral_0^inf dk rho(k) exp(-beta k^2/2C),
    via max-subtracted quad with an explicit peak hint at k*."""
    if not (0.0 < C <= C_MAX):
        raise ValueError(f"C={C} outside validated scope (0, {C_MAX}]")
    kst = k_star(beta, C)
    width = np.sqrt(C / beta)
    k_max = kst + 40.0 * width + 50.0
    ref = max(_ref_max_z(beta, C, k_max), float(log_z_integrand(kst, beta, C)))
    val, _ = quad(
        lambda k: np.exp(log_z_integrand(k, beta, C) - ref),
        0.0, k_max, points=[kst], epsabs=1e-14, epsrel=epsrel, limit=limit,
    )
    return np.log(val) + ref


def _log_g_numerator(tau, beta, C, delta, ref, s_max, epsrel, limit):
    """log of Integral ds dd exp(log_g_integrand), max-subtracted by ref,
    with peak hints on BOTH quad levels."""
    sst = s_star(beta, C)

    def inner(s):
        # Each d-slice is max-subtracted by its OWN peak (slice-local ref),
        # not just the global one: slices ~e^-30 below the global peak are
        # negligible but nonzero, and handing QUADPACK an integrand that
        # tiny (relative to the subtracted scale) makes it emit spurious
        # "probably divergent" warnings. Rescaling every slice to O(1) at
        # its peak conditions all of them identically; the slice weight
        # exp(local_ref - ref) is applied on the way out. The fine probe
        # grid (spacing <~ 0.5, resolving the O(1)-width Gamma structure)
        # also yields the slice's true peak location as an extra quad hint.
        hints = _inner_d_hints(s, tau, beta, C)
        n_probe = min(20001, max(129, int(4.0 * s)))
        probe = np.concatenate([np.asarray(hints), np.linspace(-s, s, n_probe)])
        logf_probe = log_g_integrand(s, probe, tau, beta, C, delta)
        local_ref = float(np.max(logf_probe))
        if local_ref - ref < -700.0:
            return 0.0  # slice underflows entirely relative to the global peak
        d_peak = float(probe[int(np.argmax(logf_probe))])
        eps_ = 1e-12 * max(s, 1.0)
        pts = sorted({p for p in hints + [d_peak] if -s + eps_ < p < s - eps_})
        val, _ = quad(
            lambda d: np.exp(log_g_integrand(s, d, tau, beta, C, delta) - local_ref),
            -s, s, points=pts if pts else None,
            epsabs=1e-14, epsrel=epsrel, limit=limit,
        )
        return float(np.exp(local_ref - ref)) * val

    val, _ = quad(
        inner, 0.0, s_max, points=[sst],
        epsabs=1e-14, epsrel=epsrel, limit=limit,
    )
    return np.log(val) + ref


def compute_G(tau, beta, C, delta=DELTA_SYK,
              epsrel_coarse=1e-8, limit_coarse=150,
              epsrel_fine=1e-11, limit_fine=400):
    """Exact Schwarzian Euclidean two-point function G_Delta(tau; beta, C).

    Runs the full pipeline (Z and the double integral) at two quadrature
    resolutions and self-reports conv_err = |G_coarse - G_fine| / G_fine.

    Returns a dict:
        G          -- the fine-resolution value of G_Delta(tau)
        logG       -- its log (use this if G under/overflows downstream)
        conv_err   -- relative coarse-vs-fine difference (convergence check)
        logZ       -- fine-resolution log Z(beta)
        n_warnings -- number of QUADPACK IntegrationWarnings raised (0 = clean)
    """
    _validate(tau, beta, C)
    sst = s_star(beta, C)
    s_max = sst + 40.0 * np.sqrt(2.0 * C / beta) + 50.0
    ref = _ref_max_g(tau, beta, C, delta, s_max)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)

        def one_pass(epsrel, limit):
            log_num = _log_g_numerator(tau, beta, C, delta, ref, s_max,
                                       epsrel, limit)
            log_z = compute_log_Z(beta, C, epsrel, limit)
            return log_num - log_z, log_z

        log_g_coarse, _ = one_pass(epsrel_coarse, limit_coarse)
        log_g_fine, log_z_fine = one_pass(epsrel_fine, limit_fine)
        n_warn = sum(1 for w in caught
                     if issubclass(w.category, IntegrationWarning))

    g_fine = float(np.exp(log_g_fine))
    conv_err = float(abs(np.exp(log_g_coarse - log_g_fine) - 1.0))
    return {
        "G": g_fine,
        "logG": float(log_g_fine),
        "conv_err": conv_err,
        "logZ": float(log_z_fine),
        "n_warnings": n_warn,
    }


# ----------------------------------------------------------------------
# Sanity check (the rung-4 step-3 gate): G(tau=0.1, beta=1, C=80)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TAU, BETA, C = 0.1, 1.0, 80.0
    print(f"Sanity check: G_Delta(tau={TAU}, beta={BETA}, C={C}), "
          f"Delta={DELTA_SYK}")
    res = compute_G(TAU, BETA, C)
    print(f"  G          = {res['G']:.12e}")
    print(f"  logG       = {res['logG']:.12f}")
    print(f"  logZ       = {res['logZ']:.12f}")
    print(f"  conv_err   = {res['conv_err']:.3e}   (require < 1e-6)")
    print(f"  n_warnings = {res['n_warnings']}   (QUADPACK IntegrationWarnings)")
    ok = np.isfinite(res["G"]) and res["G"] > 0 and res["conv_err"] < 1e-6
    print(f"  finite & positive & converged: {ok}")
