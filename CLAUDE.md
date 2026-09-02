# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A ladder of numerical tests of ER=EPR via persistent homology. Each rung computes an
entanglement (EPR) side and a geometry (ER) side, turns both into distance matrices, runs
ripser, and compares barcodes with persim's Wasserstein distance. Rungs 1–3 are finished
and committed (`rung1.py`–`rung3.py` + their `*_results.txt`/`*.png`); **do not modify
them**. Rung 4 (`rung4/`) is **closed as a decisive negative** (commit `3ee3035`; see
`rung4/RUNG4_REPORT_FOR_ARJUN.md` for the handoff summary and
`rung4/rung4_results.txt` for the complete verbatim record). Remaining future work is
N=20–24 via parity blocks — explicitly not started.

## Commands

Everything runs from the repo's venv; there is no test framework, build, or linter —
validation is the check suites and standing regression gates below.

```bash
.venv/bin/pip install -r requirements.txt          # numpy scipy ripser persim matplotlib

.venv/bin/python rung4/schwarzian.py               # ER-side sanity gate: G(0.1,1,80)=3.980425717581e+02
.venv/bin/python rung4/rung4_checks.py A B C D E   # validation checks (any subset; default "A B")
.venv/bin/python rung4/rung4.py extract [--smoke] [--N=12,14,16,18]
.venv/bin/python rung4/rung4.py compare [--smoke]
.venv/bin/python rung4/rung4_rescore.py            # two-convention rescore + regression vs committed table

.venv/bin/python rung1.py                          # rungs 1-3: rerunning OVERWRITES committed results in place
```

Long runs (`compare` ~53 min, rescore similar) should go in the background; spectra are
cached in `rung4/data/*.npz` and seeded rebuilds are asserted to reproduce them to <1e-8.

## Non-negotiable workflow rules (set by the project owner)

- `*_results.txt` files are **verbatim teed runs** — never hand-edit or summarize into
  them. New results are appended by running scripts. (The one labeled exception is the
  hand-written "RUNG 4 CLOSURE" section, marked as authored, not teed.)
- Work is **staged and gated**: the owner reviews each stage's results before the next
  stage runs. Never average or reinterpret a failed gate into a usable parameter.
- Physics constants must be **source-verified against papers, not memory** (this repo
  once hung a factor of √2 on whether a J in Maldacena–Stanford was plain or
  calligraphic — resolved at the PDF-font level). α_S = 0.00709 is single-source
  (derived from MS's c ≈ 0.396 N/J); say so wherever it's used.
- No retry loops: twice without a different approach = stop and name the blocker.

## Architecture

**Shared SYK builder** (`syk_model.py`): Majoranas normalized so {χi,χj} = δij·I
(χ² = I/2 — the compare stage asserts Σm W_nm = 1/2 from this), coupling variance
6J²/N³ (MS eq 2.3 at q=4). It draws from the **global NumPy RNG by design**; callers
seed externally with `np.random.seed(1000*N + r)`. Realization r of size N is
reproducible only via that convention.

**Rung 4 ER side** (`rung4/schwarzian.py`): the exact Mertens–Turiaci–Verlinde
Schwarzian two-point function. Its module docstring lists real failure modes that must
not be regressed: everything in log space; the (s,d) rotation carries
LOG_JACOBIAN = −log 2; every QUADPACK call needs explicit saddle hints
(k* = 2πC/β, s* = 4πC/β); slice-local max-subtraction. Validated scope is **C ≤ 200**
(it raises outside; quadrature genuinely fails there). `compute_G` self-reports
`conv_err` — never use a value without checking it. Absolute normalization is
implementation-anchored (shape-gauged everywhere); the standing gate value 398.04 is a
regression anchor, not a literature number.

**Rung 4 pipeline** (`rung4/rung4.py`): two stages behind one CLI. `extract` measures
C from SYK spectra (DOS sinh-edge + entropy routes, GOE control, pre-registered gate —
it failed 0/4, which is why C is *declared* from the literature in `compare`). `compare`
puts both sides on the same 24-point thermal τ circle, distance d = 1/Ghat with
Ghat = G/G(β/2), and scores W_H0+W_H1 against declared C plus ×5/÷5 and conformal
controls.

**Structural fact that constrains any future PH work here** (proved and verified
grid-wide, see `rung4/diagnostics/`): the circulant τ-circle construction with monotone
Ghat forces, via Adamaszek–Adams, exactly one H1 bar [1/Ghat(1), 1/Ghat(8)] and H0 =
n−1 copies of [0, 1/Ghat(1)] — every Wasserstein number is a function of just
(Ghat(1), Ghat(8)) per side, ~23:1 weighted toward the shortest (most UV) separation.
"H1 > 0 on both sides" is architecturally guaranteed and carries no information.
persim's `wasserstein` is order p=1 with an L2 ground metric (sum of matched costs).

**Distance conventions differ by rung** — rung 1–2: d = 1/I(A:B); rung 3: d = 1/|C_ij|;
rung 4 compare: d = 1/Ghat; rung 4 Check C: d = log(Gmax/G). Rung 4's verdict was shown
convention-robust (identical favored set under both of its conventions); don't silently
switch conventions elsewhere.

**Precision caveat**: ripser's C++ backend is float32 — raw barcode coordinates carry
~1e-7 relative error, and rung 2 documents raw-vs-exact Wasserstein differing by orders
of magnitude on 2-point inputs. Follow the existing pattern of separating float32
artifacts from physics before reporting a mismatch.

## Remote

`origin` = https://github.com/ArjunParadkar/ER-EPR. Pushing requires an account with
collaborator access to that repo (a NamanGuggilam login alone gets 403).
