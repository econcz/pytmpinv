# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [5.0.0] – 2026-11-23

Version 5.0.0 adapts TMPinv to pyclsp 5.0.0, which enforces bounds on the unknowns exactly. The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

### Added

- `zero_diagonal="exact"` in `tmpinv()`: sets the bounds of the diagonal cells to (0, 0), so that the diagonal is exactly zero (in the full and in the reduced model); a *TMPinvInputError* is raised if the bounds of a diagonal cell exclude 0 or for another string. `zero_diagonal=True` keeps its meaning (soft model rows x_kk = 0).
- `bounds` and `slack_sign` parameters of `TMPinvSolve()`: the cell bounds (one pair per cell, as normalized by `tmpinv()`) are passed to `CLSP.solve(bounds=...)`, block-wise in the reduced model; `slack_sign` restricts the slacks of `S` to be ≥ 0 (ignored in the reduced model).
- `sweep_limit` parameter of `TMPinvSolve()` (default 100, also accepted by `tmpinv()` as a keyword argument): the maximum number of block sweeps of the reduced model; a *RuntimeWarning* is issued if the sweeps do not converge within it, and a *TMPinvInputError* if it is < 1.
- `TMPinvResult.sweeps`: the number of block sweeps of the reduced model (*None* in the full model).
- `TMPinvResult.replaced`: boolean m × p mask of the cells set to `replace_value` (an optional fourth argument of `TMPinvResult()`, all *False* by default).
- Pass-through of all options of `CLSP.solve()` via keyword arguments, documented: `r`, `Z`, `rcond`, `final`, `alpha`, `scale`, `slack_sign` and CVXPY solver arguments (e.g. `solver=`); `tolerance` and `iteration_limit` are passed to CLSP.
- Bounds given as *np.nan* or ±*np.inf* denote an unbounded side (like *None*); bounds as an array of shape (m·p, 2) and as a tuple of pairs are accepted; malformed bounds (wrong length or shape, non-numeric values, low > high, low = +inf, high = −inf) raise a *TMPinvInputError*.
- Validation in the reduced model: grouped totals (`i` or `j` > 1) raise a *TMPinvInputError* instead of failing in CLSP.
- Test suite `tests/test_tmpinv.py` (26 tests; references computed with numpy, scipy and CLSP, including the equivalence of the bounded first step with scipy's BVLS on the canonical system; runs under pytest or as a script, against the installed package with `TMPINV_TEST_INSTALLED=1`, with a pyclsp source tree from `CLSP_SRC`).
- CI: a `test` job (Python 3.11 and 3.12) in the PyPI workflow, on which the release build depends, and a `tests.yml` workflow (push and pull request; Python 3.11, 3.12 and 3.13) that builds the wheel and runs the tests against the installed wheel. Both install pyclsp ≥ 5.0.0 from PyPI.
- Full MIT license text in `LICENSE` (previously only the title line).
- `CHANGELOG.md`; README sections on how TMPinv works (problem types, canonical form, bounds, Step 2, zero diagonal and symmetry, reduced models, `replace_value`), seven runnable examples with their output, an extended parameter reference, warnings and troubleshooting, limitations, and citation.

### Changed

- Bounds are enforced exactly by CLSP: by bounded-variable least squares (verified by the KKT conditions, with a scipy BVLS fallback) in Step 1 and as constraints in Step 2. Out-of-bound values can no longer occur in a CLSP estimate, also in the reduced model.
- `replace_value` applies only to the cells of a full or block model whose bounded Step 1 did not converge (`model.status["step1"]` starting with *'not converged'*) and to the cells that violate their bounds by more than `tolerance` after the symmetrization (`symmetric=True` with asymmetric bounds). *None* now means "no replacement". With `symmetric=True`, the replaced cells are mirrored, so that `result.x` stays symmetric.
- `S` is validated against the number of totals, `len(b_row)` + `len(b_col)` (previously m + p plus the internal bound rows), and its slacks are signed by default (`slack_sign=True` when `S` is supplied), so that S_kk = +1/−1 makes a total a "≤"/"≥" condition on its feasible side, as documented.
- Reduced model: the full problem is solved by block-coordinate (Gauss–Seidel) sweeps. Each block is fitted by CLSP with the other blocks fixed, i.e. with the row and column totals minus the current sums of the other blocks, its known values and its bounds (as CLSP bounds instead of additional constraint rows), without slacks; the sweeps start from the minimum-norm table of the totals (with the known values, clipped to the bounds) and stop when the largest change of a cell is below `tolerance` · max(1, max |x|). They converge toward a least-squares fit of the full problem (not necessarily the minimum-norm one); a sweep costs one CLSP fit per block. `result.model` holds the block fits of the last sweep, and `replaced` refers to them; `S` and `slack_sign` are ignored.
- `alpha` is a single value in [0, 1] and *None* means α = 1 (pyclsp 5.0.0); `rcond` is a cutoff on the singular values of A·Z (pyclsp 5.0.0); the defaults of `final`, `rcond` and `scale` are those of pyclsp.
- `result.x` is a copy of the CLSP estimate when cells are replaced, so that `result.model.x` keeps the unreplaced estimate.
- Docstrings and README describe the totals, known values and soft zero-diagonal rows as least-squares rows, the exact enforcement of the bounds, the meaning of Step 2, and the block sweeps of the reduced model.
- Requires Python ≥ 3.11 and pyclsp ≥ 5.0.0; classifiers for Python 3.11, 3.12 and 3.13; version 5.0.0 in `pyproject.toml` and `tmpinv/__init__.py`.

### Fixed

- `S` was rejected for grouped totals (`i`, `j` > 1), since the expected number of rows was computed from the dimensions of the table instead of the number of totals.
- A tuple of bound pairs was treated as a single (low, high) pair.
- The reduced model with grouped totals sliced `b_row`/`b_col` by the cell indices and failed in CLSP (now rejected with a *TMPinvInputError*).
- The reduced model did not reproduce the totals: every block was fitted independently with one slack per total, so that the assembled table overshot the totals (40 × 40 table with `reduced=(20, 20)`: row sums 1.9 to 2.7 times the row totals, R² ≈ −4.2, vs. R² = 0.28 for the full model). With the block sweeps, the totals are reproduced (relative error 2·10⁻⁹, R² = 0.27; README, Example 5).

- `zero_diagonal=True` with a known diagonal cell in `M` silently removed the soft row x_kk = 0, since the rows were merged by their coefficients only (in pyclsp and in the reduced model): 2 × 2 totals of 2 with x₁₁ = 3 gave [[2, 0.5], [0.5, 1]] (squared residual 7) instead of the least-squares table [[1.2, 1.1], [1.1, 0.6]] (5.4). All rows are now kept; the reduced model accepts repeated cells in `M` and passes every row of a cell to its block with its own value (as in the full model, so that ρ, the NRMSE and the refinement with `r` > 1 refer to the supplied rows; a one-block reduced model equals the full model).

### Removed

- The iterative bound-handling heuristic of `tmpinv()` (freezing the in-bound cells as known values in `M`, re-solving up to `iteration_limit` times, and replacing the remaining out-of-bound cells by `replace_value`).
- The internal keyword arguments `b_lim` and `C_lim` of `TMPinvSolve()` (bounds as constraint rows with slacks), and the bound rows and slack columns built for each block of the reduced model.
- The block slacks of the reduced model (one slack per block-wise total, `S` = I), replaced by the block sweeps.
- Support for Python 3.10.

### Breaking

- pyclsp ≥ 5.0.0 and Python ≥ 3.11 are required.
- Estimates with `bounds` differ from 4.0.0: they are exact bounded least-squares estimates (Step 1) and, with `final=True`, the bounded table selected by α (Step 2), instead of the result of the heuristic.
- `replace_value` has a new scope (non-converged fits and bound violations after the symmetrization), and `replace_value=None` keeps the estimate instead of producing *NaN*.
- `alpha` lists and the NRMSE rule for `alpha=None` are removed in pyclsp 5.0.0: a list, tuple or array raises a *CLSPError*, and *None* means α = 1. A float `rcond` has a different effective cutoff (pyclsp 5.0.0).
- `TMPinvSolve()` no longer accepts `b_lim`/`C_lim`; it takes normalized `bounds` (one pair per cell) and `slack_sign`.
- `S` must have `len(b_row)` + `len(b_col)` rows, and its slacks are signed by default (`slack_sign=False` restores free slacks).
- The reduced model is estimated by block-coordinate sweeps (different estimates, no block slacks, `slack_sign` ignored, `result.model` from the last sweep) and rejects grouped totals.
- `zero_diagonal=True` no longer drops the row x_kk = 0 of a known diagonal cell, and `M` with repeated cells is accepted in the reduced model (estimates differ in these cases).

## [4.0.0] – 2026-06-06

- Bug fixes.

## [3.5.0] – 2026-04-22

- Cross-platform condition-number tolerance in CLSP (`cond_tolerance`).

## [1.6.0] – 2026-01-28

- Better boundary handling in the reduced model.

## [1.4.2] – 2025-12-21

- GitHub Actions workflow for PyPI publishing.

## [1.4.0] – 2025-11-17

- `TMPinvResult.summary()` and refreshed README.

## [1.3.2] – 2025-11-06

- README update.

## [1.3.1] – 2025-10-25

- Improved bound handling via the CLSP design matrix.

## [1.2.0] – 2025-09-16

- Input checks, zero diagonal, and symmetric flag.

## [1.0.0] – 2025-08-23

- Initial commit.
