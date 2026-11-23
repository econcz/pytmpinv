# Tabular Matrix Problems via Pseudoinverse Estimation

The **Tabular Matrix Problems via Pseudoinverse Estimation (TMPinv)** is a two-stage estimation method that reformulates structured table-based systems — such as allocation problems, transaction matrices, and input–output tables — as structured least-squares problems. Based on the [Convex Least Squares Programming (CLSP)](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") framework, TMPinv solves systems with row and column constraints, block structure, and optionally reduced dimensionality by (1) constructing a canonical constraint form and applying a pseudoinverse-based projection, followed by (2) a convex-programming refinement stage to improve fit, coherence, and regularization (e.g., via Lasso, Ridge, or Elastic Net). Bounds on the cells (e.g., non-negativity, upper limits or fixed cells) are enforced exactly by CLSP; the totals and known values are fitted in least squares (see [How TMPinv Works](#how-tmpinv-works) and [Limitations](#limitations)). All calculations are performed in numpy.float64 precision.

The method is described in Bolotov (2025), see [Bibliography](#bibliography). Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

## Contents

- [Installation](#installation)
- [How TMPinv Works](#how-tmpinv-works): [Problem Types](#problem-types), [The Canonical Form](#the-canonical-form), [Bounds](#bounds), [Step 2](#step-2), [Zero Diagonal and Symmetry](#zero-diagonal-and-symmetry), [Reduced Models](#reduced-block-wise-models), [`replace_value`](#replace_value)
- [Examples](#examples):
  [1. Quick Example](#1-quick-example-an-input-output-type-table-with-x--0-and-known-cells),
  [2. Bounds](#2-lower-upper-and-cell-specific-bounds),
  [3. Zero Diagonal](#3-a-zero-diagonal-soft-rows-vs-exact-zeros),
  [4. Inconsistent Totals](#4-inconsistent-totals-ρ-residual-and-status),
  [5. Reduced Models](#5-reduced-block-wise-models),
  [6. Reading the Results](#6-reading-the-results-and-diagnostics),
  [7. `replace_value`](#7-replace_value-and-replaced)
- [User Reference](#user-reference): [TMPinv Parameters](#tmpinv-parameters), [CLSP Parameters](#clsp-parameters), [Returns](#returns)
- [Warnings and Troubleshooting](#warnings-and-troubleshooting)
- [Limitations](#limitations)
- [What's New in 5.0.0](#whats-new-in-500)
- [Citing](#citing)
- [Acknowledgements](#acknowledgements) · [Bibliography](#bibliography) · [License](#license)

## Installation

```bash
pip install pytmpinv
```

The package requires Python ≥ 3.11 (tested on Python 3.11, 3.12 and 3.13) and depends on numpy and [pyclsp](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") ≥ 5.0.0 (installed automatically, together with scipy, cvxpy and the solvers ECOS, OSQP, SCS and CLARABEL).

## How TMPinv Works

### In Plain Terms

A tabular matrix problem asks for the cells of a table X ∈ ℝ^{m×p} when only some aggregate information is known: the row totals, the column totals and, possibly, some cells. The problem is usually underdetermined (there are more cells than totals) and, if the totals come from different sources, inconsistent (the row totals and the column totals do not add up to the same grand total).

`tmpinv()` writes the problem as an allocation problem (AP (TM)) of CLSP, `CLSP().solve(problem="ap", ...)`, and returns the estimated table together with the fitted CLSP object(s). CLSP solves it in two steps:

1. **Step 1** computes the minimum-norm least-squares solution with a pseudoinverse. If `bounds` on the cells are set, the bounded least-squares problem is solved instead (bounded-variable least squares, an active-set method that moves cells to and from their bounds).
2. **Step 2** keeps the best achievable fit of Step 1 fixed and, among all tables that attain it, chooses the one closest to the unbounded Step 1 estimate in a Lasso (L1), Ridge (L2) or Elastic Net sense, subject to the bounds. With the default α = 1 (Ridge), this is the minimum-norm (bounded) least-squares table.

Please note that the totals and the known values are fitted *in least squares*: if they are consistent and compatible with the bounds, they are reproduced exactly (up to rounding); otherwise, they are traded off against each other. Only the `bounds` (and the signs of the slacks of `S`) are enforced exactly. A hard-constraint mode is not implemented in CLSP.

### Problem Types

TMPinv covers the following variants of the problem, which can be combined:

- **Row and column totals** (`b_row`, `b_col`): the sums of the rows and of the columns of X, both required (at least 2 entries each).
- **Grouped totals** (`i`, `j`): each entry of `b_row` is the sum of `i` consecutive rows, each entry of `b_col` the sum of `j` consecutive columns, so that m = `len(b_row)`·`i` and p = `len(b_col)`·`j`.
- **Known values** (`M`, `b_val`): linear restrictions on the cells, typically the selection of known cells (rows of an identity matrix) with their values in `b_val`.
- **Inequality totals** (`S`): a total can be a "≤" or "≥" condition instead of an equality (a diagonal sign matrix, signed slacks).
- **Bounds** (`bounds`): lower and/or upper bounds on all cells or on each cell, e.g. non-negativity `(0, None)`; lo = hi fixes a cell.
- **Zero diagonal** (`zero_diagonal`): x_kk = 0, e.g. for trade or flow matrices, as soft model rows (`True`) or exactly via bounds (`"exact"`).
- **Symmetry** (`symmetric`): a symmetric table, obtained by averaging the estimate with its transpose.
- **Reduced (block-wise) models** (`reduced`): a large table is estimated block by block, by repeated sweeps over smaller problems on contiguous blocks.

### The Canonical Form

```text
A z = b,      A = [ C  S ]      z = [ x ]      b = [ b_row ]
                  [ M  0 ]          [ y ]          [ b_col ]
                                                   [ b_val ]
```

- x ∈ ℝ^{m·p} is X stored row by row (`x[k]` is the cell (k // p, k % p)), returned as the m × p matrix `result.x`;
- `C` are the constraint rows built by CLSP from m, p, `i` and `j`: `len(b_row)` row-sum rows followed by `len(b_col)` column-sum rows;
- `S` (optional) holds the coefficients of the slacks y of the totals: S_kk = +1 makes total k a "≤" condition and S_kk = −1 a "≥" condition, since the slacks are restricted to y ≥ 0 (`slack_sign=True`, the default when `S` is supplied);
- `M` are the model rows with right-hand side `b_val`, e.g. the known cells; with `zero_diagonal=True`, the rows x_kk = 0 are added to `M` (and the zeros to `b`) by CLSP;
- the bounds lb ≤ x ≤ ub are not rows of `A`: they are constraints of both steps.

The fitted CLSP object is `result.model` (a list of CLSP objects in the reduced model): `model.A` and `model.b` are the canonical system above, and `model.status`, `model.rho` and `model.residual` report the outcome of the fit (see Example 6). For the mathematical details of both steps, including the numerical-rank policy (`rcond`) and the verification of Step 2, see the [pyclsp documentation](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming").

### Bounds

`bounds` are passed to CLSP and enforced exactly in both steps:

- in Step 1 by bounded-variable least squares (BVLS: the Lawson–Hanson active-set method in the bounded form of Stark and Parker, with pseudoinverse subproblems on the free cells), whose result is verified by the Karush–Kuhn–Tucker (KKT) conditions of the bounded least-squares problem; if the active set does not converge, the same problem is solved by `scipy.optimize.lsq_linear(..., method="bvls")` with a *RuntimeWarning*;
- in Step 2 as constraints of the convex program, whose solution is verified and clipped to the bounds.

An out-of-bound cell can therefore not occur in the CLSP estimate. A single pair `(lo, hi)` applies to all cells, a list of m·p pairs to each cell; *None*, *np.nan* and ±*np.inf* denote an unbounded side. If the totals and known values cannot be met within the bounds, the bounds are kept and the totals are fitted in least squares: `model.rho` > 0 is then the smallest residual attainable under the bounds (Example 2).

In version 4.0.0, bounds were handled by a heuristic: the cells within their bounds were frozen as known values (rows of `M`), the model was re-estimated, and the cells that remained out of bounds were replaced by `replace_value` (*np.nan* by default). This heuristic is removed in 5.0.0.

### Step 2

With `final=True` (the default), CLSP computes the fitted vector w* of the bounded least-squares problem of Step 1, its residual ρ = ‖`b` − w*‖₂ (`model.rho`), and the table

```text
z* = argmin (1 − α)‖z − zhat0‖₁ + α‖z − zhat0‖₂²   subject to   A_k z = w*,   lb ≤ x ≤ ub
```

where zhat0 is the unbounded Step 1 estimate (`model.zhat0`) and `A`_k is `A` with the singular values below the cutoff `rcond` set to zero. All values of α attain the same residual ρ; α only selects among the equally well-fitting tables. α = 1 (Ridge, the default) gives the unique minimum-norm (bounded) least-squares table, α = 0 (Lasso) a table with the smallest L1 deviation from the anchor, and 0 < α < 1 the Elastic Net. Without bounds (and without signed slacks), Step 2 returns the Step 1 estimate for every α. α must be a single value in [0, 1]; it can be chosen by cross-validation over the model rows with `result.model.cv_alpha()` (pyclsp).

### Zero Diagonal and Symmetry

`zero_diagonal=True` adds the soft model rows x_kk = 0, k = 1, …, min(m, p), which CLSP fits in least squares like the totals: if the totals cannot be met with a zero diagonal, the diagonal cells become nonzero. A known value of a diagonal cell in `M` is kept together with its row x_kk = 0, so both are traded off (e.g. 2 × 2 totals of 2 and x₁₁ = 3 give x₁₁ = 1.2). `zero_diagonal="exact"` instead sets the bounds of the diagonal cells to (0, 0), which makes them exactly zero (the misfit, if any, is then carried by the totals; Example 3). The bounds of the diagonal cells must admit 0.

`symmetric=True` returns the symmetric table 0.5·(x + xᵀ) and requires m = p. The symmetrization is applied to `result.x` only (not to `result.model`) and after the fit: the totals of a symmetric table are reproduced if `b_row` = `b_col`, the bounds are preserved if they are symmetric (the bounds of the cells (k, l) and (l, k) are equal), and a known cell is preserved if its mirror cell is known with the same value. For symmetry in the model itself, add symmetry rows x_kl − x_lk = 0 to `M` (with zeros in `b_val`) in a full-model solve.

### Reduced (Block-Wise) Models

With `reduced=(r, c)`, the table is split into contiguous blocks of at most (r − 1) × (c − 1) cells, and the full problem is solved by block-coordinate (Gauss–Seidel) sweeps: in each sweep, every block is fitted in turn by CLSP as an AP (TM) problem with the other blocks fixed at their current values:

- the row (column) totals of a block are the entries of `b_row` (`b_col`) minus the current sums of the other blocks' cells in these rows (columns), so that the blocks are linked through the totals of the full table;
- the known values (`M` must consist of rows of the identity matrix; repeated rows on the same cell, e.g. a known diagonal cell and x_kk = 0, are passed to the block of the cell as separate rows with their own values, as in the full model, and their mean is used only for the start), the `bounds` and an exact zero diagonal are mapped to the blocks; with `zero_diagonal=True`, the zero diagonal is added as known zeros. The options of `CLSP.solve()` are passed to every block fit;
- the sweeps start from the minimum-norm table of the totals, with the known values inserted and clipped to the bounds, and stop when the largest change of a cell between two sweeps is below `tolerance` · max(1, max |x|), or after `sweep_limit` sweeps (default *100*) with a *RuntimeWarning*;
- `S` is ignored (with a *UserWarning*), and grouped totals (`i`, `j` > 1) are not supported.

Each block fit minimizes the full least-squares objective (totals and known values) over the cells of the block, so that the sweeps are block coordinate descent on a convex problem: they converge toward a least-squares fit of the full problem, which reproduces consistent totals within the bounds (Example 5). If the full problem has several least-squares solutions (the usual case), the sweeps reach one of them, not necessarily the minimum-norm (Step 2) solution of the full model. A sweep costs one CLSP fit per block, i.e. an SVD of a (r − 1 + c − 1 + k_block) × (r − 1)(c − 1) matrix instead of one of the full (m + p + k) × m·p matrix, and the number of sweeps (`result.sweeps`) depends on the problem (a few to several dozen in the examples). `result.model` holds the block fits of the last sweep; `symmetric=True` averages the table after the sweeps with its transpose: for a symmetric problem (`b_row` = `b_col`, symmetric known values and bounds), the average is again a least-squares fit, but, like the sweeps, not necessarily the one of the full model. With `r` > 1, every block fit builds its residual slacks `Q` from the residuals of its own model rows, given the current values of the other blocks; with one block, the result equals the full model with the same `r`, with several blocks, it is in general not equivalent to the full model with `r` > 1.

### `replace_value`

Since the bounds are enforced by CLSP, `replace_value` no longer replaces out-of-bound values of the fit. It applies only to

- the cells of a (full or block) model whose bounded Step 1 did not converge (in the reduced model, a block fit of the last sweep), i.e. `model.status["step1"]` starts with *'not converged'* (both the active-set method and scipy's BVLS failed, and CLSP kept the last feasible point, which is not a verified least-squares solution), and
- the cells that violate their bounds by more than `tolerance` after the symmetrization (`symmetric=True` with asymmetric bounds, Example 7).

The default *np.nan* marks these cells as missing; *None* keeps the estimate. The replaced cells are reported in the boolean mask `result.replaced` (all *False* for a converged fit), and `result.model` keeps the unreplaced CLSP estimate.

## Examples

Each example below is self-contained; the output shown was produced by running it with pytmpinv 5.0.0 and pyclsp 5.0.0 (numpy 64-bit floats; the last digits of solver-dependent values may differ across platforms and solver versions).

### 1. Quick Example: an Input-Output-Type Table with x ≥ 0 and Known Cells

```python
import numpy  as     np
from   tmpinv import tmpinv

# AP (TM), based on a symmetric input-output-type table with known cell pairs and x ≥ 0

seed   = 123456789
rng    = np.random.default_rng(seed)

# sample (dataset)
m, p   = 8, 8                                                # matrix dimensions (m x p)
X_true = np.abs(rng.normal(size=(m, p)))                     # non-negative X_true
X_true[rng.random((m, p)) < 0.4] = 0.0                       # empty cells
X_true = (X_true + X_true.T) / 2.0                           # symmetric X_true
idx    = rng.choice(m * p, size=int(0.15 * (m * p)),         # random numbers from
                    replace=False)                           # [0,(m * p) - 1], 15% of total
idx    = np.unique(np.r_[idx, (idx % p) * p + idx // p])     # with their mirror cells

# model
M      = np.eye(m * p)[idx,:]                                # unit matrix
b_row  = X_true.sum(axis=1)                                  # row sums (length m)
b_col  = X_true.sum(axis=0)                                  # column sums (length p)
b_val  = X_true.reshape(-1, 1)[idx]                          # column vector
result = tmpinv(
             M=M, b_row=b_row, b_col=b_col, b_val=b_val,
             bounds=(0, None),                               # x ≥ 0, enforced exactly
             symmetric=True,
             alpha=1.0                                       # unique, minimum-norm (α = 1)
         )

x_free = tmpinv(M=M, b_row=b_row, b_col=b_col, b_val=b_val,  # the same model without bounds
                symmetric=True).x

# results
print("true X:")
print(np.round(X_true,   4))
print("X_hat:")
print(np.round(result.x, 4))
print("min(X_hat)             :", result.x.min())
print("min(X_hat), unbounded  :", np.round(x_free.min(), 4))
print("max |row sums - b_row| :", f"{np.max(np.abs(result.x.sum(axis=1) - b_row)):.2e}")
print("max |col sums - b_col| :", f"{np.max(np.abs(result.x.sum(axis=0) - b_col)):.2e}")
print("max |known - b_val|    :", f"{np.max(np.abs(result.x.reshape(-1, 1)[idx] - b_val)):.2e}")
print("status:", result.model.status["step1"], "/", result.model.status["final"],
      "/", result.model.status["solver"])
print(f"rho = {result.model.rho:.2e}, residual = {result.model.residual:.2e}, "
      f"replaced = {int(result.replaced.sum())}")

x_true = X_true.flatten()
x_hat  = result.x.flatten()
ss_res = np.sum((x_true - x_hat) ** 2)
ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)
print(f"R2_user_defined: {1 - ss_res / ss_tot:.4f}")
```

Output:

```text
true X:
[[0.     0.1704 1.4032 0.5425 0.312  0.9686 0.8737 0.0616]
 [0.1704 0.     0.     0.6177 0.1679 0.4936 0.6805 0.    ]
 [1.4032 0.     1.2154 0.5329 0.     0.6259 0.0259 0.8683]
 [0.5425 0.6177 0.5329 0.     0.     1.3392 0.0596 0.4808]
 [0.312  0.1679 0.     0.     1.7209 0.1324 1.1566 0.1612]
 [0.9686 0.4936 0.6259 1.3392 0.1324 0.1372 0.395  0.0083]
 [0.8737 0.6805 0.0259 0.0596 1.1566 0.395  0.     0.1078]
 [0.0616 0.     0.8683 0.4808 0.1612 0.0083 0.1078 0.    ]]
X_hat:
[[0.3319 0.2447 1.4032 0.5135 0.312  0.5785 0.8737 0.0746]
 [0.2447 0.1575 0.3022 0.4263 0.2087 0.4913 0.2995 0.    ]
 [1.4032 0.3022 0.4468 0.5709 0.     0.636  0.4442 0.8683]
 [0.5135 0.4263 0.5709 0.     0.4775 0.7601 0.5683 0.2561]
 [0.312  0.2087 0.     0.4775 1.7209 0.5426 0.3507 0.0386]
 [0.5785 0.4913 0.636  0.7601 0.5426 0.1372 0.6333 0.3212]
 [0.8737 0.2995 0.4442 0.5683 0.3507 0.6333 0.     0.1294]
 [0.0746 0.     0.8683 0.2561 0.0386 0.3212 0.1294 0.    ]]
min(X_hat)             : 0.0
min(X_hat), unbounded  : -0.1302
max |row sums - b_row| : 2.31e-14
max |col sums - b_col| : 2.40e-14
max |known - b_val|    : 1.40e-14
status: converged / optimal / OSQP
rho = 3.56e-14, residual = 5.95e-14, replaced = 0
R2_user_defined: 0.5683
```

Without bounds, the minimum-norm estimate contains negative cells (−0.1302). With `bounds=(0, None)`, the estimate is non-negative exactly, and the totals and the known cells (chosen in mirror pairs, so that the symmetrization preserves them) are reproduced up to rounding, since they are consistent: ρ ≈ 0. With only the totals and 14 known cells for 64 unknowns, the estimate is the minimum-norm non-negative table consistent with this information, not the true table (R² = 0.5683).

### 2. Lower, Upper and Cell-Specific Bounds

```python
import numpy  as np
from   tmpinv import tmpinv

# AP (TM) 3×3 with known row and column sums: lower, upper and cell-specific bounds

b_row  = np.array([1, 10, 10])                               # row sums
b_col  = np.array([1, 10, 10])                               # column sums
lohi   = [(0, None)] * 9                                     # x ≥ 0 for all cells,
lohi[4] = (0, 4.5)                                           # x_22 ≤ 4.5,
lohi[8] = (5, 5)                                             # x_33 = 5 (a fixed cell)
for label, bounds in [("unbounded      ", None),
                      ("bounds=(0, None)", (0, None)),
                      ("cell-specific  ", lohi),
                      ("bounds=(0, 4.6)", (0, 4.6))]:      # infeasible with the totals
    result = tmpinv(b_row=b_row, b_col=b_col, bounds=bounds)
    model  = result.model
    print(f"{label}: step1 = {model.status['step1']}, rho = {model.rho:.4f}, "
          f"residual = {model.residual:.4f}")
    print(np.round(result.x, 4))
    print("  row sums:", np.round(result.x.sum(axis=1), 4),
          "column sums:", np.round(result.x.sum(axis=0), 4))
```

Output:

```text
unbounded      : step1 = converged, rho = 0.0000, residual = 0.0000
[[-1.6667  1.3333  1.3333]
 [ 1.3333  4.3333  4.3333]
 [ 1.3333  4.3333  4.3333]]
  row sums: [ 1. 10. 10.] column sums: [ 1. 10. 10.]
bounds=(0, None): step1 = converged, rho = 0.0000, residual = 0.0000
[[0.   0.5  0.5 ]
 [0.5  4.75 4.75]
 [0.5  4.75 4.75]]
  row sums: [ 1. 10. 10.] column sums: [ 1. 10. 10.]
cell-specific  : step1 = converged, rho = 0.0000, residual = 0.0000
[[0.   0.75 0.25]
 [0.75 4.5  4.75]
 [0.25 4.75 5.  ]]
  row sums: [ 1. 10. 10.] column sums: [ 1. 10. 10.]
bounds=(0, 4.6): step1 = converged, rho = 0.4899, residual = 0.4899
[[0.  0.6 0.6]
 [0.6 4.6 4.6]
 [0.6 4.6 4.6]]
  row sums: [1.2 9.8 9.8] column sums: [1.2 9.8 9.8]
```

The unbounded minimum-norm estimate has a negative cell. With `bounds=(0, None)`, the cell x_11 is fixed at 0 and the other cells are re-estimated, so that the totals are met exactly. The cell-specific bounds hold exactly as well (x_22 = 4.5 at its upper bound, x_33 = 5 fixed), again with the totals met. With `bounds=(0, 4.6)`, no table meets the totals (rows 2 and 3 must sum to 20, but column 1 contributes at most 1 and their other four cells at most 4 · 4.6 = 18.4): the bounds still hold exactly, and the totals are fitted in least squares with the smallest attainable residual ρ = 0.4899.

### 3. A Zero Diagonal: Soft Rows vs. Exact Zeros

```python
import numpy  as np
from   tmpinv import tmpinv

# AP (TM) 3×3 with x ≥ 0: a zero diagonal as soft model rows vs. exact zeros via bounds

b_row  = np.array([10, 1, 1])                                # row sums
b_col  = np.array([10, 1, 1])                                # column sums
for zero_diagonal in (True, "exact"):
    result = tmpinv(b_row=b_row, b_col=b_col, bounds=(0, None),
                    zero_diagonal=zero_diagonal)
    print(f"zero_diagonal={zero_diagonal!r}: rho = {result.model.rho:.4f}")
    print(np.round(result.x, 4))
    print("  row sums:", np.round(result.x.sum(axis=1), 4),
          "column sums:", np.round(result.x.sum(axis=0), 4))

# AP (TM) 5×5 trade matrix with a zero diagonal and feasible totals

rng    = np.random.default_rng(42)
X_true = rng.uniform(0, 2, size=(5, 5)) * (1 - np.eye(5))    # zero diagonal
result = tmpinv(b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0),
                bounds=(0, None), zero_diagonal="exact")
print("5×5, zero_diagonal='exact': diagonal =", np.diag(result.x),
      f"residual = {result.model.residual:.1e}")
```

Output:

```text
zero_diagonal=True: rho = 5.0596
[[3.2 2.6 2.6]
 [2.6 0.  0. ]
 [2.6 0.  0. ]]
  row sums: [8.4 2.6 2.6] column sums: [8.4 2.6 2.6]
zero_diagonal='exact': rho = 6.5320
[[0.     3.6667 3.6667]
 [3.6667 0.     0.    ]
 [3.6667 0.     0.    ]]
  row sums: [7.3333 3.6667 3.6667] column sums: [7.3333 3.6667 3.6667]
5×5, zero_diagonal='exact': diagonal = [0. 0. 0. 0. 0.] residual = 1.2e-14
```

Non-negative cells cannot meet these totals with a zero diagonal (row 1 must sum to 10 without x_11, but its other cells lie in columns 2 and 3, which sum to 1 each). `zero_diagonal=True` adds the rows x_kk = 0 as model rows, which are traded off against the totals: x_11 = 3.2. `zero_diagonal="exact"` fixes the diagonal at 0 through the bounds; the misfit is then carried by the totals alone (ρ = 6.5320 > 5.0596). If the totals are compatible with a zero diagonal, as in the 5 × 5 trade matrix, they are met exactly with an exactly zero diagonal.

### 4. Inconsistent Totals: ρ, Residual and Status

```python
import numpy  as np
from   tmpinv import tmpinv

# AP (TM) 3×3 with inconsistent totals (row sums: 21, column sums: 22)

b_row  = np.array([1, 10, 10])                               # row sums
b_col  = np.array([1, 10, 11])                               # column sums
for bounds, alpha in [(None, 1.0), ((0, None), 0.0), ((0, None), 0.5),
                      ((0, None), 1.0)]:
    result = tmpinv(b_row=b_row, b_col=b_col, bounds=bounds, alpha=alpha)
    model  = result.model
    print(f"bounds = {bounds}, alpha = {alpha}: step1 = {model.status['step1']}, "
          f"final = {model.status['final']}, solver = {model.status['solver']}")
    print(f"  rho = {model.rho:.4f}, residual = {model.residual:.4f}, "
          f"nrmse = {model.nrmse:.4f}")
    print(np.round(result.x, 4))
    print("  row sums:", np.round(result.x.sum(axis=1), 4),
          "column sums:", np.round(result.x.sum(axis=0), 4))
```

Output:

```text
bounds = None, alpha = 1.0: step1 = converged, final = optimal, solver = none (anchor feasible)
  rho = 0.4082, residual = 0.4082, nrmse = 0.0381
[[-1.7222  1.2778  1.6111]
 [ 1.2778  4.2778  4.6111]
 [ 1.2778  4.2778  4.6111]]
  row sums: [ 1.1667 10.1667 10.1667] column sums: [ 0.8333  9.8333 10.8333]
bounds = (0, None), alpha = 0.0: step1 = converged, final = optimal, solver = ECOS
  rho = 0.4082, residual = 0.4082, nrmse = 0.0381
[[0.     0.4789 0.6878]
 [0.4167 4.6772 5.0728]
 [0.4167 4.6772 5.0728]]
  row sums: [ 1.1667 10.1667 10.1667] column sums: [ 0.8333  9.8333 10.8333]
bounds = (0, None), alpha = 0.5: step1 = converged, final = optimal, solver = CLARABEL
  rho = 0.4082, residual = 0.4082, nrmse = 0.0381
[[0.     0.4167 0.75  ]
 [0.4167 4.7083 5.0417]
 [0.4167 4.7083 5.0417]]
  row sums: [ 1.1667 10.1667 10.1667] column sums: [ 0.8333  9.8333 10.8333]
bounds = (0, None), alpha = 1.0: step1 = converged, final = optimal, solver = OSQP
  rho = 0.4082, residual = 0.4082, nrmse = 0.0381
[[0.     0.4167 0.75  ]
 [0.4167 4.7083 5.0417]
 [0.4167 4.7083 5.0417]]
  row sums: [ 1.1667 10.1667 10.1667] column sums: [ 0.8333  9.8333 10.8333]
```

The row totals (21) and column totals (22) cannot both be met, so they are fitted in least squares: ρ = 0.4082 is the smallest attainable residual, with or without x ≥ 0 here, and every α attains it (`residual` = `rho`). Without bounds, Step 2 does not change the minimum-norm estimate. With x ≥ 0, α selects among the equally well-fitting non-negative tables: the Lasso solution (α = 0) differs from the Elastic Net and Ridge solutions. `status` reports the outcome of the bounded Step 1 (`"step1"`), of Step 2 (`"final"`) and the solver used.

### 5. Reduced (Block-Wise) Models

```python
import numpy  as np
from   tmpinv import tmpinv

# AP (TM), based on a 40×40 trade matrix with a zero diagonal and 20% of known values,
# estimated as reduced models of (20, 20), i.e. 3×3 blocks of at most 19×19 cells

seed   = 123456789
rng    = np.random.default_rng(seed)

# sample (dataset)
m, p   = 40, 40                                              # matrix dimensions (m x p)
X_true = np.abs(rng.normal(size=(m, p)))                     # non-negative X_true
X_true = X_true * (1 - np.eye(m, p))                         # zero diagonal
idx    = rng.choice(m * p, size=int(0.2 * (m * p)),          # random numbers from
                    replace=False)                           # [0,(m * p) - 1], 20% of total

# model
M      = np.eye(m * p)[idx,:]                                # unit matrix
b_row  = X_true.sum(axis=1)                                  # row sums (length m)
b_col  = X_true.sum(axis=0)                                  # column sums (length p)
b_val  = X_true.reshape(-1, 1)[idx]                          # column vector
result = tmpinv(
             M=M, b_row=b_row, b_col=b_col, b_val=b_val,
             bounds=(0, None),                               # x ≥ 0
             zero_diagonal="exact",                          # x_kk = 0 via bounds
             reduced=(20, 20)                                # reduced models of (20, 20)
         )

# results
print("blocks:", len(result.model), "shapes:", sorted({mod.x.shape for mod in result.model}))
print("sweeps:", result.sweeps)
print("step1 :", sorted({mod.status["step1"] for mod in result.model}),
      "final:", sorted({mod.status["final"] for mod in result.model}))
print("min(X_hat) =", result.x.min(), " max |diag(X_hat)| =", np.abs(np.diag(result.x)).max())
print("max |known - b_val|:", f"{np.max(np.abs(result.x.reshape(-1, 1)[idx] - b_val)):.1e}")
print("max relative error of the totals:",
      f"{max(np.max(np.abs(result.x.sum(axis=1) / b_row - 1)), np.max(np.abs(result.x.sum(axis=0) / b_col - 1))):.1e}")

x_true = X_true.flatten()
x_hat  = result.x.flatten()
ss_res = np.sum((x_true - x_hat) ** 2)
ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)
print(f"R2_user_defined: {1 - ss_res / ss_tot:.4f}")

full   = tmpinv(M=M, b_row=b_row, b_col=b_col, b_val=b_val, bounds=(0, None),
                zero_diagonal="exact")                       # the full model, for comparison
x_full = full.x.flatten()
print(f"full model: R2_user_defined = {1 - np.sum((x_true - x_full) ** 2) / ss_tot:.4f}, "
      f"max |X_hat - X_full| = {np.max(np.abs(result.x - full.x)):.4f}, "
      f"‖X_hat‖ / ‖X_full‖ = {np.linalg.norm(result.x) / np.linalg.norm(full.x):.4f}")
```

Output:

```text
blocks: 9 shapes: [(2, 2), (2, 19), (19, 2), (19, 19)]
sweeps: 31
step1 : ['converged'] final: ['optimal']
min(X_hat) = 0.0  max |diag(X_hat)| = 0.0
max |known - b_val|: 6.7e-08
max relative error of the totals: 2.1e-09
R2_user_defined: 0.2702
full model: R2_user_defined = 0.2784, max |X_hat - X_full| = 0.2109, ‖X_hat‖ / ‖X_full‖ = 1.0020
```

The 40 × 40 table is estimated as 9 blocks (19 + 19 + 2 rows and columns) in 31 sweeps. The bounds (x ≥ 0 and the exact zero diagonal) hold, and the totals and the known values are reproduced up to the convergence tolerance of the sweeps. The estimate is a least-squares fit of the full problem, but not the minimum-norm one selected by the full model: it is slightly larger in norm and differs from the full-model estimate by up to 0.21 per cell (R² = 0.2702 vs. 0.2784). Fitted independently, each with one slack per total (as in 4.0.0), the blocks' sums over a row were 1.9 to 2.7 times the row totals here (R² = −4.2279). A sweep costs one CLSP fit per block, so that the reduced model is slower than the full model for a table of this size (here, about 5 times); it pays off for larger tables (for an 80 × 80 table with 16 blocks, 13 sweeps took about a third of the time of the full model) and for tables that are too large for the full model (see [Limitations](#limitations)).

### 6. Reading the Results and Diagnostics

```python
import numpy  as np
from   tmpinv import tmpinv

# AP (TM) 5×5 with x ≥ 0 and 6 noisy known values (an inconsistent system), followed by the results

seed   = 7
rng    = np.random.default_rng(seed)

# sample (dataset)
m, p   = 5, 5                                                # matrix dimensions (m x p)
X_true = rng.uniform(0, 3, size=(m, p))
X_true[rng.random((m, p)) < 0.3] = 0.0                       # empty cells
idx    = rng.choice(m * p, size=6, replace=False)            # known cells

# model
result = tmpinv(
             M=np.eye(m * p)[idx], b_val=X_true.ravel()[idx] + 0.3 * rng.normal(size=6),
             b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0),
             bounds=(0, None)
         )
model  = result.model                                        # the CLSP object

# results
print("X_hat:")
print(np.round(result.x, 4))
print(f"rho = {model.rho:.4f}, residual = {model.residual:.4f}, rank = {model.rank} of {model.A.shape[1]}")
for key in ("step1", "rho", "final", "solver", "retry", "kkt_error", "bound_error", "eq_error"):
    val = model.status[key]
    print(f"  {key:<12}", f"{val:.1e}" if isinstance(val, float) else val)
result.summary(display=True)
print("  Bootstrap t-test (NRMSE_partial):")
for kw, val in model.ttest(sample_size=30, seed=seed, partial=True).items():
    print(f"    {kw}: {float(val):.6f}")
```

Output:

```text
X_hat:
[[1.3521 2.6717 2.5319 0.5873 1.3271]
 [1.2003 1.2133 0.     0.4356 1.1753]
 [0.0953 0.1083 0.     1.5354 0.4316]
 [2.2156 2.2286 2.1122 1.4509 2.1906]
 [0.2786 0.2916 1.8986 0.     0.2536]]
rho = 0.2027, residual = 0.2027, rank = 15 of 25
  step1        converged
  rho          bvls
  final        optimal
  solver       OSQP
  retry        None
  kkt_error    1.0e-13
  bound_error  0.0e+00
  eq_error     1.2e-15
Estimator Configuration:
  Generalized inverse:   Moore-Penrose
  Iterations (r):                    1
  Tolerance:                 1.490e-08
  Final correction:               True
  Regularization (α):           1.0000
  Bounds (x):                 [0, inf]
  Slack sign (S ≥ 0):            False
  Column scaling:                False
  Fixed variables:                   3

Numerical Stability:
  kappaC:                       1.4142
  kappaB:                       1.0698
  kappaA:                       6.7360
  rank:                             15

Goodness of Fit:
  r2_partial:                   0.9938
  nrmse:                        0.0182
  nrmse_partial:                0.0787
  z_lower:        min= 0.000e+00  max=    2.4847  mean=    1.0262  sd=    0.8311
  z_upper:        min= 0.000e+00  max=    2.8587  mean=    1.1807  sd=    0.9562
  x_lower:        min= 0.000e+00  max=    2.4847  mean=    1.0262  sd=    0.8311
  x_upper:        min= 0.000e+00  max=    2.8587  mean=    1.1807  sd=    0.9562
  y_lower:        min=       nan  max=       nan  mean=       nan  sd=       nan
  y_upper:        min=       nan  max=       nan  mean=       nan  sd=       nan
  Bootstrap t-test (NRMSE_partial):
    p_one_left: 0.998400
    p_one_right: 0.001600
    p_two_sided: 0.003199
    nrmse: 0.078688
    mean_null: 0.051109
    std_null: 0.046996
    failures: 0.000000
```

The noisy known values are inconsistent with the totals, so ρ = 0.2027 > 0. `result.model` is the CLSP object of the fit: its `status` reports a converged bounded Step 1 (`"step1"`, `"kkt_error"`) and a verified Step 2 (`"final"`, `"eq_error"`), and `summary()` prints the estimator configuration (3 cells at the bound 0 in Step 1), the condition numbers and the goodness of fit. The bootstrap t-test compares the observed NRMSE of the model rows with a resampled null sample; it and the sensitivity bands are heuristic diagnostics (see [Limitations](#limitations)).

| Attribute | Meaning |
|---|---|
| `result.x` | estimated m × p table (symmetrized if `symmetric=True`, with `replace_value` in the replaced cells) |
| `result.replaced` | boolean m × p mask of the cells set to `replace_value` |
| `result.full` | *True* for the full model, *False* for the reduced model |
| `result.model` | CLSP object (full model) or list of CLSP objects, one per block (reduced model) |
| `model.x` | CLSP estimate of the (block) table, before the symmetrization and the replacement |
| `model.rho` | smallest residual ‖`b` − w*‖₂ attainable under the bounds (*None* if `final=False`) |
| `model.residual` | ‖`b` − `A`z‖₂ of the returned estimate |
| `model.status["step1"]` | *'converged'*, *'bvls fallback'* (scipy's BVLS solved the bounded problem) or *'not converged: <reason>'* |
| `model.status["final"]` | *'optimal'* or *'optimal_inaccurate'* (verified), *'fallback: <reason>'* (z = zhat) or *'skipped'* |
| `model.zhat`, `model.zhat0` | Step 1 estimate (bounded) and the unbounded Step 1 estimate, the anchor of Step 2 |
| `model.fixed`, `model.fixed_at` | components of z at a bound at the end of Step 1, and the pass at which they were fixed |
| `model.A`, `model.b` | canonical system of the fit |

A fit is usable as intended if `status["step1"]` is *'converged'* or *'bvls fallback'* and `status["final"]` does not start with *'fallback'*.

### 7. `replace_value` and `replaced`

```python
import numpy  as np
from   tmpinv import tmpinv

# AP (TM) 3×3 symmetric table with an asymmetric cell-specific bound: replace_value

X_true = np.array([[0.0, 2.0, 1.0],
                   [2.0, 0.0, 3.0],
                   [1.0, 3.0, 0.0]])
lohi   = [(0, None)] * 9                                     # x ≥ 0
lohi[1] = (0, 0.1)                                           # x_12 ≤ 0.1, but not x_21
for replace_value in (np.nan, None):
    result = tmpinv(b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0),
                    bounds=lohi, symmetric=True, replace_value=replace_value)
    print(f"replace_value = {replace_value}: replaced cells =",
          np.argwhere(result.replaced).tolist())
    print(np.round(result.x, 4))
print("CLSP estimate before the symmetrization (x_12 ≤ 0.1 holds):")
print(np.round(result.model.x, 4))
```

Output:

```text
replace_value = nan: replaced cells = [[0, 1], [1, 0]]
[[1.2833    nan 1.1542]
 [   nan 2.6167 1.8208]
 [1.1542 1.8208 1.025 ]]
replace_value = None: replaced cells = []
[[1.2833 0.5625 1.1542]
 [0.5625 2.6167 1.8208]
 [1.1542 1.8208 1.025 ]]
CLSP estimate before the symmetrization (x_12 ≤ 0.1 holds):
[[1.2833 0.1    1.6167]
 [1.025  2.6167 1.3583]
 [0.6917 2.2833 1.025 ]]
```

The bound x_12 ≤ 0.1 holds in the CLSP estimate (`result.model.x`), but not after the symmetrization, since the mirror cell x_21 is not bounded: the average 0.5625 violates the bound. With the default `replace_value=np.nan`, the two cells are replaced and marked in `result.replaced`; with `replace_value=None`, the symmetric estimate is kept as it is. Symmetric bounds (equal bounds for x_12 and x_21) avoid the conflict.

## User Reference

For comprehensive information on the estimator's capabilities, advanced configuration options, and implementation details, please refer to the [pyclsp module](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming"), on which TMPinv is based, and to the docstrings in `tmpinv/tmpinv.py`.

To ensure cross-platform reproducibility, all CLSP implementations use a modified condition number function based on singular values, with a relative cutoff equal to `cond_tolerance * the largest singular value`.

```python
from tmpinv import tmpinv
result = tmpinv(bounds, replace_value, tolerance, iteration_limit, cond_tolerance, **kwargs)
```

All arguments are passed by keyword. `tmpinv()` normalizes `bounds`, calls `TMPinvSolve()` (`tmpinv.tmpinv.TMPinvSolve`), which builds the CLSP problem(s), and applies `replace_value`.

### TMPinv Parameters

`S` : *array_like* of shape *(m + p, m + p)*, optional<br>
A diagonal sign slack (surplus) matrix with entries in *{0, ±1}* (a zero-padded subset of ±I; zero columns may be omitted):<br>
-   *0* enforces equality (== `b_row` or `b_col`),<br>
-  *1* enforces a lower-than-or-equal (≤) condition,<br>
- *–1* enforces a greater-than-or-equal (≥) condition.

The first `m` diagonal entries correspond to row constraints, and the remaining `p` to column constraints; with grouped totals, `S` has `len(b_row)` + `len(b_col)` rows. The conditions are imposed through slacks y ≥ 0 (`slack_sign`, *True* by default when `S` is supplied): a total with S_kk = ±1 is free on its feasible side and remains a least-squares row on its infeasible side. Please note that, in the reduced model, `S` is ignored.

`M` : *array_like* of shape *(k, m * p)*, optional<br>
A model matrix with entries in *{0, 1}*. Each row defines a linear restriction on the flattened solution matrix (row-major). The corresponding right-hand side values must be provided in `b_val`. This block is used to encode known cell values (least-squares rows, like the totals). Please note that, in the reduced model, `M` must consist of rows of an identity matrix (known cells; a cell may repeat). Arbitrary or non-diagonal model matrices cannot be mapped to reduced blocks, making the model infeasible.

`b_row` : *array_like* of shape *(m,)*<br>
Right-hand side vector of row totals (of row groups if `i` > 1). Please note that both `b_row` and `b_col` must be provided, with at least 2 finite entries each.

`b_col` : *array_like* of shape *(p,)*<br>
Right-hand side vector of column totals (of column groups if `j` > 1). Please note that both `b_row` and `b_col` must be provided, with at least 2 finite entries each.

`b_val` : *array_like* of shape *(k,)*<br>
Right-hand side vector of known cell values (finite).

`i` : *int*, default = *1*<br>
Number of rows in a row group: each entry of `b_row` is the sum of `i` consecutive rows, and the table has m = `len(b_row)` · `i` rows. Not supported in the reduced model.

`j` : *int*, default = *1*<br>
Number of columns in a column group: each entry of `b_col` is the sum of `j` consecutive columns, and the table has p = `len(b_col)` · `j` columns. Not supported in the reduced model.

`zero_diagonal` : *bool* or *'exact'*, default = *False*<br>
If *True*, adds the soft model rows x_kk = 0 for the diagonal cells (least-squares rows, traded off against the totals in an inconsistent system). If *'exact'*, sets the bounds of the diagonal cells to (0, 0), which makes them exactly zero; their bounds in `bounds` must admit 0.

`reduced` : *tuple* of *(int, int)*, optional<br>
Dimensions of the reduced problem. If specified, the problem is estimated by block-coordinate sweeps over contiguous submatrices of the original table. For example, `reduced` = *(6, 6)* implies *5×5* data blocks (edge blocks may be smaller). Each dimension must be at least 3. See [Reduced Models](#reduced-block-wise-models).

`sweep_limit` : *int*, default = *100*<br>
Maximum number of block sweeps in the reduced model (ignored in the full model). The sweeps stop when the largest change of a cell between two sweeps is below `tolerance` · max(1, max |x|); if this is not reached within `sweep_limit` sweeps, a *RuntimeWarning* is issued and the last table is returned. The number of sweeps performed is `TMPinvResult.sweeps`.

`symmetric` : *bool*, default = *False*<br>
If True, enforces symmetry of the estimated solution matrix as: x = 0.5 * (x + x.T), which requires m = p.
Applies to TMPinvResult.x only (after the sweeps in the reduced model). For TMPinvResult.model symmetry, add explicit symmetry constraints to M in a full-model solve instead of using this flag.

`bounds` : *sequence* of *(low, high)*, optional<br>
Bounds on cell values. If a single tuple *(low, high)* is given, it is applied to all `m` * `p` cells. Example: *(0, None)*. A list of one pair or of `m` * `p` pairs (row-major) is also accepted, as is an array of shape (`m` * `p`, 2). *None*, *np.nan* and ±*np.inf* denote an unbounded side, low = high fixes a cell, and low ≤ high is required. The bounds are enforced exactly by CLSP (see [Bounds](#bounds)); in the reduced model, they are mapped to the blocks.

`replace_value` : *float* or *None*, default = *np.nan*<br>
Replacement value for the cells of a full or block model whose bounded first step did not converge (`model.status["step1"]` starting with *'not converged'*), and for the cells that violate their bounds by more than `tolerance` after the symmetrization. If *None*, no cell is replaced. The replaced cells are marked in `TMPinvResult.replaced`. See [`replace_value`](#replace_value).

`tolerance` : *float*, default = *square root of machine epsilon*<br>
Tolerance passed to CLSP (the bound and KKT tolerances of the bounded first step, the convergence of the refinement for `r` > 1, the verification tolerance of the second step sqrt(`tolerance`)·max(1, ‖`b`‖₂)), the tolerance of the bound check of `replace_value`, and the convergence tolerance of the sweeps of the reduced model.

`iteration_limit` : *int*, default = *50*<br>
Maximum number of passes of the bounded first step (after which scipy's BVLS is used) and of iterations of the refinement loop (`r` > 1) in CLSP.

`slack_sign` : *bool* or *None*, default = *None*<br>
If *True*, the slacks of `S` must be ≥ 0. If *None*, *True* if `S` is supplied, *False* otherwise. Ignored in the reduced model.

`cond_tolerance` : *float* or *None*, default = *None*<br>
Singular-value cutoff for the custom condition number function. If *None*, the implementation uses an internal relative cutoff of `1e-14`.

`m`, `p` : *int*, optional<br>
Accepted and ignored: the dimensions of the table are derived from `b_row`, `b_col`, `i` and `j`.

### CLSP Parameters

The following keyword arguments are passed to `CLSP.solve()` (of every block fit in the reduced model); see the pyclsp documentation for the details.

`r` : *int*, default = *1*<br>
Number of refinement iterations for the pseudoinverse-based estimator (used only with model rows `M`). For `r` > 1, the misfit of the model rows is moved into residual slacks, which changes the estimator.

`Z` : *np.ndarray* or *None*<br>
A symmetric idempotent matrix (projector) defining the subspace for Bott–Duffin pseudoinversion. If *None*, the identity matrix is used, reducing the Bott–Duffin inverse to the Moore–Penrose case. With `bounds`, `Z` must be diagonal (0/1). In the reduced model, a `Z` would have to conform to every block, so it is of use in the full model only.

`rcond` : *float*, *bool* or *None*, default = *None*<br>
Relative cutoff on the singular values of `A`·`Z`: the singular values σ ≤ `rcond`·σ_max are treated as zero in both steps (`model.rank`, `model.sv_discarded`). If *None* or *False*, the cutoff sqrt(max(`A`.shape)·eps) is applied, which is robust to the structural rank deficiency of AP (TM) problems (the row and the column totals both sum to the grand total); if *True*, a cutoff equal to `tolerance`. Please note that `rcond` is applied to the singular values of `A`·`Z` since pyclsp 5.0.0 (previously to those of `Z`·`A`ᵀ`A`·`Z`).

`final` : *bool* or *None*, default = *None* (*True*)<br>
If *True*, Step 2 (the convex correction) is solved; if *False*, `x` is the Step 1 estimate.

`alpha` : *float* or *None*, default = *None*<br>
Regularization parameter (weight) of Step 2, a single value in [0, 1]:<br>
- `α = 0`: Lasso (L1 norm), solved with ECOS<br>
- `α = 1`: Tikhonov Regularization/Ridge (L2 norm), solved with OSQP (retried with CLARABEL if needed)<br>
- `0 < α < 1`: Elastic Net, solved with CLARABEL<br>
If *None*, α = 1. A *list*, *tuple* or *np.ndarray* of candidates, NaN, inf and values outside [0, 1] raise a *CLSPError* (the selection of α by NRMSE was removed in pyclsp 5.0.0; use `result.model.cv_alpha()`).

`scale` : *bool* or *None*, default = *None* (*False*)<br>
If *True*, the columns of `A` are equilibrated in both steps (useful for totals of very different magnitudes); the minimum-norm selection is then made in the scaled variables.

`*args`, `**kwargs` : optional<br>
CVXPY arguments passed to the CVXPY solver of Step 2 (e.g., `solver="CLARABEL"` or solver tolerances).

### Returns

*TMPinvResult*

`TMPinvResult.full` : *bool*<br>
Indicates if this result comes from the full (non-reduced) model.

`TMPinvResult.model` : *CLSP* or *list* of *CLSP*<br>
A single CLSP object in the full model, or a list of CLSP objects for each reduced block in the reduced model (row-block by row-block; the fits of the last sweep). Each object exposes the CLSP results, e.g. `status`, `rho`, `residual`, `zhat`, `zhat0`, `rank`, `fixed`, the condition numbers, the goodness-of-fit statistics, and the methods `corr()`, `ttest()` and `cv_alpha()`.

`TMPinvResult.x` : *np.ndarray*<br>
Final estimated solution matrix of shape *(m, p)*.

`TMPinvResult.replaced` : *np.ndarray*<br>
Boolean mask of shape *(m, p)* of the cells set to `replace_value` (all *False* if none).

`TMPinvResult.sweeps` : *int* or *None*<br>
Number of block sweeps in the reduced model (*None* in the full model).

`TMPinvResult.summarize(i, display)`<br>
An alias of TMPinvResult.summary().

`TMPinvResult.summary(i, display)`<br>
Return or print a summary of the underlying CLSP result, where `i` : int, default = *None* is the index of a reduced-block model in TMPinvResult.model.

## Warnings and Troubleshooting

Invalid inputs raise a *TMPinvInputError* (`tmpinv.tmpinv.TMPinvInputError`, with the methods `as_dict()` and the attributes `message` and `code`) or, for the arguments passed to CLSP (e.g. `alpha`), a *CLSPError* (`clsp.errors.CLSPError`). A numerical step that does not reach its target issues a *RuntimeWarning* from pyclsp, and the result is still returned; the outcome is recorded in `model.status` (Example 6).

| Message | Cause and remedy |
|---|---|
| *TMPinvInputError*: Both b_row and b_col must be provided. / Minimum length for b_row and b_col is 2. / b_row and b_col must not contain inf or NaN. | Missing, too short or non-finite totals. |
| *TMPinvInputError*: Bounds length … does not match number of variables … / bounds must be None, a (low, high) pair, or a sequence of (low, high) pairs. / bounds must satisfy low ≤ high. | Malformed `bounds`: the number of pairs must be 1 or `m`·`p`. |
| *TMPinvInputError*: zero_diagonal must be True, False or 'exact'. / zero_diagonal='exact' requires bounds that admit 0 on the diagonal. | Invalid `zero_diagonal`, or bounds of a diagonal cell that exclude 0. |
| *TMPinvInputError*: S must have … rows. / S must be a zero-padded subset of ±I. | `S` must have one row per total (`len(b_row)` + `len(b_col)`) and entries in {0, ±1}, at most one nonzero per row and column. |
| *TMPinvInputError*: Both M and b_val must be defined. / M must have exactly … columns. / b_val must not contain inf or NaN. / M and b_val must have the same number of rows | Inconsistent known values. |
| *TMPinvInputError*: Each reduced block must be at least (3, 3) … / Grouped totals (i, j > 1) are not supported in the reduced model. / M must consist of rows of the identity matrix in the reduced model. / sweep_limit must be at least 1. | Unsupported reduced-model inputs. |
| *TMPinvInputError*: symmetric=True requires a square matrix (m == p). | `symmetric=True` for a non-square table. |
| *TMPinvInputError*: Reduced model: please supply the block index using i=#. / i must be in … | `summary()` of a reduced model without or with an invalid block index. |
| *CLSPError*: alpha must be a single value in [0, 1] … | A list, tuple or array of α, or an invalid α. |
| *UserWarning*: User-provided S is ignored in the reduced model. | `S` together with `reduced`. |
| *RuntimeWarning*: Reduced model: the block sweeps did not converge in … sweeps (…); increase sweep_limit. | The last table is returned; it may not reproduce the totals. Increase `sweep_limit` or use larger blocks. |
| *RuntimeWarning*: Bounded least-squares active set did not converge (…); using scipy BVLS | The result is a verified bounded least-squares solution (`status["step1"]` = *'bvls fallback'*); usually nothing to do. |
| *RuntimeWarning*: Bounded least-squares active set did not converge (…); the last feasible estimate is kept | Not a verified least-squares solution: the cells of this (block) model are set to `replace_value`. Try `scale=True`, a larger `iteration_limit` or a larger `rcond`. |
| *RuntimeWarning*: Step 2 failed (…); falling back to zhat | The Step 1 estimate (a bounded least-squares solution) is returned; try `scale=True` or another solver (`solver="CLARABEL"`). |

## Limitations

- **Soft totals and known values.** The totals, the known values and the rows of `zero_diagonal=True` are least-squares rows: in an inconsistent system, they are traded off against each other and are not met exactly (Examples 3, 4 and 6). Exact restrictions are available only as `bounds` (non-negativity, upper limits, fixed cells, `zero_diagonal="exact"`); `S` with signed slacks makes a total an inequality only on its feasible side.
- **Reduced model.** The sweeps of a reduced model converge toward a least-squares fit of the full problem, but not necessarily the minimum-norm (Step 2) solution of the full model, so that the estimate differs from the full-model estimate when the least-squares solution is not unique (Example 5). The convergence is linear and can be slow (several dozen sweeps of one CLSP fit per block); a sweep limit reached issues a *RuntimeWarning*. `M` must select cells, `S` and grouped totals are not supported, and `result.model` holds the block fits of the last sweep only.
- **Symmetrization after the fit.** `symmetric=True` averages the estimate with its transpose after the fit: asymmetric bounds can then be violated (the cells are set to `replace_value`, Example 7), and known cells are preserved only if their mirror cells are known with the same values.
- **Step 2 needs active bounds.** Without bounds (and signed slacks), Step 2 returns the Step 1 estimate for every α.
- **Heuristic diagnostics.** The t-tests, the sensitivity bands and the correlogram of pyclsp are heuristic diagnostics without statistical coverage (see the pyclsp documentation).
- **Dense matrices.** All matrices are dense NumPy arrays, and the number of unknowns is m·p: the canonical matrix of a full model has (m + p + k) × m·p entries, and pyclsp computes its SVD and stores a (m·p) × (m·p) projector `Z`. The pyclsp documentation reports about 0.3 s and 20 MB for a 40 × 40 table and 4 s and 328 MB for an 80 × 80 table (`bounds=(0, None)`); larger tables require the reduced model or another method.

## What's New in 5.0.0

Version 5.0.0 adapts TMPinv to pyclsp 5.0.0; the full list of changes is in [CHANGELOG.md](CHANGELOG.md).

- Bounds are passed to CLSP and enforced exactly (bounded-variable least squares in Step 1, constraints in Step 2); the iterative heuristic of 4.0.0 (freezing in-bound cells as known values and replacing out-of-bound cells) is removed.
- `replace_value` applies only to the cells of non-converged fits and to bound violations caused by the symmetrization; *None* keeps the estimate; new attribute `TMPinvResult.replaced`.
- `zero_diagonal="exact"` (exact zeros via bounds); `zero_diagonal=True` adds soft model rows as before, now without dropping the row x_kk = 0 of a known diagonal cell (the reduced model accepts repeated cells in `M`).
- `S` uses signed slacks (`slack_sign=True` by default), so that S_kk = ±1 makes a total a "≤"/"≥" condition on its feasible side.
- The reduced model solves the full problem by block-coordinate sweeps (new parameter `sweep_limit`, new attribute `TMPinvResult.sweeps`): the assembled table reproduces the totals, instead of fitting each block independently with one slack per total.
- All options of `CLSP.solve()` are passed through (`r`, `Z`, `rcond`, `final`, `alpha`, `scale`, `slack_sign`, solver arguments); α is a single value in [0, 1] (α = 1 by default).
- Bounds given as *np.nan* or ±*np.inf* denote unbounded sides, arrays of pairs are accepted, and malformed bounds raise a *TMPinvInputError*; a test suite and CI workflows; Python ≥ 3.11.

**Breaking changes:** pyclsp ≥ 5.0.0 and Python ≥ 3.11 are required; the estimates with `bounds` differ from 4.0.0 (exact bounded least squares instead of the heuristic); `replace_value` has a new meaning, and *None* now keeps the estimate; `alpha` lists and the NRMSE rule for `alpha=None` (pyclsp) are removed; the internal keyword arguments `b_lim` and `C_lim` of `TMPinvSolve()` are removed; `S` is validated against the number of totals and uses signed slacks; the reduced model is estimated by block sweeps (different estimates, no block slacks) and rejects grouped totals. See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use TMPinv in your work, please cite the method:

Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

and the version of the package you used (`tmpinv.__version__`):

pytmpinv, version 5.0.0: Tabular Matrix Problems via Pseudoinverse Estimation [Python package]. [https://github.com/econcz/pytmpinv](https://github.com/econcz/pytmpinv)

```bibtex
@article{bolotov2025clsp,
  author  = {Bolotov, I.},
  title   = {{CLSP}: Linear Algebra Foundations of a Modular Two-Step Convex
             Optimization-Based Estimator for Ill-Posed Problems},
  journal = {Mathematics},
  year    = {2025},
  volume  = {13},
  number  = {21},
  pages   = {3476},
  doi     = {10.3390/math13213476}
}

@software{pytmpinv,
  title   = {pytmpinv: Tabular Matrix Problems via Pseudoinverse Estimation},
  version = {5.0.0},
  url     = {https://github.com/econcz/pytmpinv}
}
```

## Acknowledgements

The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

## Bibliography
Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

## License

MIT License — see the [LICENSE](LICENSE) file.
