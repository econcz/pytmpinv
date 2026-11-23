"""
Regression and acceptance tests for pytmpinv 5.0.0.

The references are computed with numpy, scipy and pyclsp's CLSP only. Runs
under pytest or as a plain script: `python tests/test_tmpinv.py`. With the
environment variable TMPINV_TEST_INSTALLED=1, the installed package is tested
instead of the repo copy; CLSP_SRC=<path> prepends a pyclsp source tree to
sys.path.
"""
import os
import sys
import traceback
import warnings
from   functools      import wraps
from   pathlib        import Path
import numpy          as np
from   scipy.optimize import lsq_linear

ROOT      = Path(__file__).resolve().parents[1]
INSTALLED = os.environ.get("TMPINV_TEST_INSTALLED") == "1"
if os.environ.get("CLSP_SRC"):                       # pyclsp source tree
    sys.path.insert(0, os.environ["CLSP_SRC"])
if not INSTALLED:                                    # test the repo copy
    sys.path.insert(0, str(ROOT))
import clsp                                          # noqa: E402
import clsp.solver                                   # noqa: E402
import tmpinv                                        # noqa: E402
from   clsp           import CLSP                    # noqa: E402
from   clsp.errors    import CLSPError               # noqa: E402
from   tmpinv         import tmpinv as tmp           # noqa: E402
from   tmpinv.tmpinv  import TMPinvInputError        # noqa: E402

TOL      = 1e-8                                      # numerical tolerance
X_AP     = np.array([[.50, .25, .25],                # known AP case
                     [.25, 4.75, 5.0],
                     [.25, 5.0, 4.75]])
B_ROW    = X_AP.sum(1)                               # [1, 10, 10]
B_COL    = X_AP.sum(0)                               # [1, 10, 10]
X_AP_NN  = np.array([[0.0, .50, .50],                # min-norm, x ≥ 0
                     [.50, 4.75, 4.75],
                     [.50, 4.75, 4.75]])

# Ancillary functions (independent references)
def _ap_matrix(m, p):
    """Row sums, then column sums of an m×p matrix (row-major vec)."""
    return np.vstack([np.kron(np.eye(m), np.ones((1, p))),
                      np.kron(np.ones((1, m)), np.eye(p))])

def _min_norm(A, b):
    """Moore–Penrose minimum-norm least-squares solution."""
    return np.linalg.lstsq(A, b, rcond=None)[0]

def _bounded_ls(A, b, lb, ub):
    """Bounded least squares (scipy BVLS): z and ‖b - Az‖."""
    eq     = lb == ub                                # BVLS needs lb < ub
    z      = np.where(eq, lb, 0.0)
    res    = lsq_linear(A[:, ~eq], b - A[:, eq] @ lb[eq],
                        bounds=(lb[~eq], ub[~eq]), method="bvls",
                        tol=1e-14, max_iter=10_000)
    z[~eq] = res.x
    return z, np.linalg.norm(b - A @ z)

def _table(seed, m, p, empty=0.3):
    """Random non-negative m×p table with empty cells."""
    rng                          = np.random.default_rng(seed)
    X                            = rng.uniform(0, 3, size=(m, p))
    X[rng.random((m, p)) < empty] = 0.0
    return X

def _close(a, b, tol=TOL):
    return np.allclose(np.ravel(a), np.ravel(b), rtol=0.0, atol=tol)

def _raises(func, error=TMPinvInputError, text=""):
    try:
        func()
    except error as e:
        assert text in str(e), str(e)
        return
    raise AssertionError(f"no {error.__name__} raised")

def _converged(model):
    return (model.status["step1"] in {"converged", "bvls fallback"} and
            not model.status["final"].startswith("fallback"))

def _patched(obj, name, value, func):
    """func() with obj.name temporarily replaced by value."""
    orig = getattr(obj, name)
    setattr(obj, name, value)
    try:
        return func()
    finally:
        setattr(obj, name, orig)

def _step1_fails(fail):
    """A wrapper of clsp.solver._bvls reporting non-convergence if fail()."""
    bvls = clsp.solver._bvls
    def patched(*args, **kwargs):
        out = list(bvls(*args, **kwargs))
        if fail():
            out[3] = "not converged: test"
        return tuple(out)
    return patched

def _bvls_fallback_ok(func):
    """func() without the RuntimeWarning of a scipy BVLS fallback."""
    @wraps(func)
    def quiet():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Bounded least-"
                                    r"squares .*; using scipy BVLS$",
                                    category=RuntimeWarning)
            return func()
    return quiet

# 0. Package under test
def test_imports_package_under_test():
    path = Path(tmpinv.__file__).resolve().parent
    assert (path != ROOT / "tmpinv") if INSTALLED else (path == ROOT /
                                                        "tmpinv")
    assert tmpinv.__version__ == "5.0.0"
    assert clsp.__version__.split(".")[0] == "5"

# 1. Full model with known totals
def test_full_unbounded_min_norm():
    res = tmp(b_row=B_ROW, b_col=B_COL)
    A   = _ap_matrix(3, 3)
    assert res.full and isinstance(res.model, CLSP)
    assert res.x.shape == (3, 3) and not res.replaced.any()
    assert _close(res.x, _min_norm(A, np.r_[B_ROW, B_COL]))
    assert abs(res.x[0, 0] + 5 / 3) < TOL            # negative cell
    assert _close(res.x.sum(1), B_ROW) and _close(res.x.sum(0), B_COL)
    assert res.model.status["step1"] == "converged"
    assert res.model.status["rho"]   == "pinv"

def test_full_nonnegative_exact():
    res = tmp(b_row=B_ROW, b_col=B_COL, bounds=(0, None))
    assert _close(res.x, X_AP_NN)
    assert np.min(res.x) >= 0.0                      # exact, no tolerance
    assert _close(res.x.sum(1), B_ROW) and _close(res.x.sum(0), B_COL)
    assert _converged(res.model) and res.model.status["rho"] == "bvls"
    assert list(np.flatnonzero(res.model.fixed)) == [0]
    assert res.model.residual < TOL and not res.replaced.any()

def test_full_nonnegative_random_tables():
    for seed in range(5):
        X   = _table(seed, 6, 7)
        res = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None))
        assert np.min(res.x) >= 0.0 and _converged(res.model)
        assert _close(res.x.sum(1), X.sum(1), 1e-7)  # feasible: totals
        assert _close(res.x.sum(0), X.sum(0), 1e-7)
        assert res.model.residual < 1e-7

def test_upper_bounds_feasible_and_infeasible():
    X     = _table(11, 5, 5)
    hi    = float(np.max(X))
    res   = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, hi))
    assert np.min(res.x) >= 0.0 and np.max(res.x) <= hi
    assert _close(res.x.sum(1), X.sum(1), 1e-7)
    res   = tmp(b_row=B_ROW, b_col=B_COL,
                bounds=(0, 4.6))                     # infeasible
    lb    = np.zeros(9)
    ub    = np.full(9, 4.6)
    _, r  = _bounded_ls(_ap_matrix(3, 3), np.r_[B_ROW, B_COL], lb, ub)
    assert np.min(res.x) >= 0.0 and np.max(res.x) <= 4.6
    assert r > 0.1 and abs(res.model.rho - r) < 1e-8
    assert abs(res.model.residual - r) < 1e-7 and _converged(res.model)

def test_per_cell_bounds():
    X     = _table(12, 4, 5)
    lohi  = [(0, v + 0.1) for v in X.ravel()]        # active upper bounds
    lohi[3] = (X.ravel()[3], X.ravel()[3])           # a fixed cell
    res   = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=lohi)
    lb    = np.array([l for l, _ in lohi])
    ub    = np.array([h for _, h in lohi])
    assert np.all(res.x.ravel() >= lb) and np.all(res.x.ravel() <= ub)
    assert res.x.ravel()[3] == X.ravel()[3]
    assert _close(res.x.sum(1), X.sum(1), 1e-7) and _converged(res.model)

def test_bvls_equivalence_canonical_system():
    for seed in range(6):
        rng   = np.random.default_rng(100 + seed)
        m, p  = 4, 5
        idx   = rng.choice(m * p, size=3, replace=False)
        M     = np.eye(m * p)[idx]
        b_row = rng.uniform(1, 4, m)
        b_col = rng.uniform(1, 4, p)                 # inconsistent totals
        b_val = rng.normal(size=3)
        lb    = np.where(rng.random(m * p) < 0.8, rng.uniform(-1, 0, m * p),
                         -np.inf)
        ub    = np.where(rng.random(m * p) < 0.5, rng.uniform(0.2, 1, m * p),
                         np.inf)
        lohi  = [(None if np.isinf(l) else l, None if np.isinf(h) else h)
                 for l, h in zip(lb, ub)]
        A     = np.vstack([_ap_matrix(m, p), M])
        b     = np.r_[b_row, b_col, b_val]
        z, r  = _bounded_ls(A, b, lb, ub)
        for final in (False, True):
            res = tmp(b_row=b_row, b_col=b_col, M=M, b_val=b_val,
                      bounds=lohi, final=final)
            x   = res.x.ravel()
            assert np.all(x >= lb) and np.all(x <= ub)
            assert _close(res.model.A, A, 0.0)       # canonical system
            assert abs(np.linalg.norm(b - A @ x) - r) < 1e-8, seed
            assert _close(A @ x, A @ z, 1e-6), seed  # unique fitted vector
            if final:
                assert abs(res.model.rho - r) < 1e-8 and _converged(res.model)

# 2. Known values, symmetry, zero diagonal and grouped totals
def test_known_values():
    X     = _table(21, 5, 6)
    rng   = np.random.default_rng(21)
    idx   = rng.choice(30, size=8, replace=False)
    M     = np.eye(30)[idx]
    b_val = X.ravel()[idx]
    for bounds in (None, (0, None)):
        res = tmp(b_row=X.sum(1), b_col=X.sum(0), M=M, b_val=b_val,
                  bounds=bounds)
        assert _close(res.x.ravel()[idx], b_val, 1e-7)
        assert _close(res.x.sum(1), X.sum(1), 1e-7)
        assert res.model.A.shape == (11 + 8, 30)
    assert np.min(res.x) >= 0.0
    _raises(lambda: tmp(b_row=X.sum(1), b_col=X.sum(0), M=M), text="b_val")
    _raises(lambda: tmp(b_row=X.sum(1), b_col=X.sum(0), M=M[:, :29],
                        b_val=b_val), text="columns")

def test_symmetric():
    X   = _table(31, 5, 5)
    X   = (X + X.T) / 2
    res = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
              symmetric=True)
    assert _close(res.x, res.x.T, 0.0) and np.min(res.x) >= 0.0
    assert _close(res.x.sum(1), X.sum(1), 1e-7) and not res.replaced.any()
    _raises(lambda: tmp(b_row=[1, 2], b_col=[1, 1, 1], symmetric=True),
            text="square")

def test_zero_diagonal_soft_and_exact():
    kw    = dict(b_row=[10, 1, 1], b_col=[10, 1, 1], bounds=(0, None))
    soft  = tmp(zero_diagonal=True,    **kw)         # soft rows x_kk = 0
    exact = tmp(zero_diagonal="exact", **kw)         # bounds lo = hi = 0
    assert abs(soft.x[0, 0] - 3.2) < 1e-7            # traded off
    assert soft.model.A.shape[0] == 6 + 3
    assert np.all(np.diag(exact.x) == 0.0) and np.min(exact.x) >= 0.0
    assert exact.model.A.shape[0] == 6               # no model rows
    assert _close(exact.x, [[0, 11 / 3, 11 / 3], [11 / 3, 0, 0],
                            [11 / 3, 0, 0]], 1e-7)
    X     = _table(32, 4, 4) * (1 - np.eye(4))
    res   = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
                zero_diagonal="exact")
    assert np.all(np.diag(res.x) == 0.0)
    assert _close(res.x.sum(1), X.sum(1), 1e-7)      # feasible: totals
    _raises(lambda: tmp(zero_diagonal="exact", **(kw | dict(bounds=(1, 2)))),
            text="admit 0")
    _raises(lambda: tmp(zero_diagonal="hard", **kw), text="zero_diagonal")

def test_zero_diagonal_known_cell():
    # x11 = 3 (M) and x11 = 0 (zero diagonal) are both least-squares rows:
    # the least-squares fit of all 7 rows, ‖r‖² = 5.4
    kw  = dict(b_row=[2.0, 2.0], b_col=[2.0, 2.0], M=np.eye(4)[[0]],
               b_val=[3.0], zero_diagonal=True)
    for extra in ({}, {"reduced": (3, 3)}):
        assert _close(tmp(**kw, **extra).x, [[1.2, 1.1], [1.1, 0.6]], 1e-8)
    # repeated cells in the reduced model: rows kept, not averaged
    X   = _table(33, 6, 5)
    idx = np.r_[np.random.default_rng(33).choice(30, size=8), 0, 0, 7]
    kw  = dict(b_row=X.sum(1) + 0.1, b_col=X.sum(0),
               M=np.eye(30)[idx], b_val=X.ravel()[idx] + 0.2 * (idx == 0))
    rho = _ls_residual(tmp(**kw).x, **kw)
    res = tmp(reduced=(4, 4), sweep_limit=1000, **kw)
    assert rho > 0.1 and abs(_ls_residual(res.x, **kw) - rho) < 1e-7

def test_reduced_repeated_rows_one_block():
    # one block = the full model, also for r > 1 and for rho
    kw = dict(b_row=[2.0, 2.0], b_col=[2.0, 2.0], M=np.eye(4)[[0, 0]],
              b_val=[3.0, 0.0])
    for r, x in ((1, [[1.5, 0.5], [0.5, 1.5]]),
                 (3, [[7 / 6, 5 / 6], [5 / 6, 7 / 6]])):
        full = tmp(r=r, **kw)
        res  = tmp(r=r, reduced=(3, 3), **kw)
        assert _close(full.x, x, 1e-8) and _close(res.x, x, 1e-8), f"{r}"
        assert abs(res.model[0].rho - full.model.rho) < 1e-10
    assert abs(tmp(**kw).model.rho - np.sqrt(4.5)) < 1e-10

def test_grouped_totals():
    X   = _table(41, 4, 6)
    res = tmp(b_row=X.reshape(2, 12).sum(1),
              b_col=X.reshape(4, 3, 2).sum(axis=(0, 2)),
              i=2, j=2, bounds=(0, None))
    assert res.x.shape == (4, 6) and np.min(res.x) >= 0.0
    assert _close(res.x.reshape(2, 12).sum(1), X.reshape(2, 12).sum(1),
                  1e-7)
    S   = np.zeros((5, 1))
    S[0, 0] = 1.0                                    # rows m/i + p/j = 5
    res = tmp(b_row=X.reshape(2, 12).sum(1),
              b_col=X.reshape(4, 3, 2).sum(axis=(0, 2)), i=2, j=2, S=S)
    assert res.model.slack_sign and res.model.A.shape == (5, 25)

def test_S_and_slack_sign():
    S   = np.zeros((6, 1))
    S[0, 0] = 1.0                                    # row 1 ≤ 2
    kw  = dict(b_row=[2, 10, 10], b_col=[1, 10, 10], bounds=(0, None), S=S)
    res = tmp(**kw)
    assert res.model.slack_sign and np.min(res.model.y) >= 0.0
    assert res.x[0].sum() <= 2.0 + 1e-8 and res.model.residual < 1e-7
    assert not tmp(slack_sign=False, **kw).model.slack_sign
    assert not tmp(b_row=B_ROW, b_col=B_COL).model.slack_sign
    _raises(lambda: tmp(**(kw | dict(S=np.zeros((5, 1))))), text="rows")
    _raises(lambda: tmp(**(kw | dict(S=2 * S))), text="±I")

# 3. Reduced (block-wise) models
def _ls_residual(x, b_row, b_col, M=None, b_val=None):
    """‖b - Ax‖ of the full problem (totals and known values)."""
    m, p = np.shape(x)
    A    = _ap_matrix(m, p) if M is None else np.vstack([_ap_matrix(m, p), M])
    b    = np.r_[b_row, b_col, [] if b_val is None else np.ravel(b_val)]
    return np.linalg.norm(b - A @ np.ravel(x))

def test_reduced_shapes_and_sweeps():
    X      = _table(51, 14, 11)
    res    = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
                 reduced=(7, 7))
    shapes = [mod.x.shape for mod in res.model]
    assert not res.full and res.x.shape == (14, 11)
    assert shapes == [(6, 6), (6, 5), (6, 6), (6, 5), (2, 6), (2, 5)]
    assert np.min(res.x) >= 0.0 and all(_converged(v) for v in res.model)
    assert 1 <= res.sweeps < 100 and tmp(b_row=B_ROW,
                                         b_col=B_COL).sweeps is None
    k      = 0
    for r0 in range(0, 14, 6):
        for c0 in range(0, 11, 6):
            blk  = res.x[r0:r0 + 6, c0:c0 + 6]
            rest = res.x.copy()
            rest[r0:r0 + 6, c0:c0 + 6] = 0.0         # other blocks
            n    = blk.shape[0]
            assert _close(blk, res.model[k].x, 0.0)  # last sweep
            assert _close(res.model[k].b[:n, 0],     # remaining totals
                          (X.sum(1) - rest.sum(1))[r0:r0 + 6], 1e-6)
            assert _close(res.model[k].b[n:, 0],
                          (X.sum(0) - rest.sum(0))[c0:c0 + 6], 1e-6)
            assert res.model[k].A.shape[1] == (      # no slacks
                   blk.size)
            k   += 1
    assert isinstance(res.summary(i=0), dict)
    _raises(lambda: res.summary(), text="block")
    _raises(lambda: res.summary(i=6), text="0..")

@_bvls_fallback_ok
def test_reduced_converges_to_full():
    X    = _table(52, 12, 12) * (1 - np.eye(12))
    idx  = np.random.default_rng(52).choice(144, size=30, replace=False)
    kw   = dict(b_row=X.sum(1), b_col=X.sum(0), M=np.eye(144)[idx],
                b_val=X.ravel()[idx])
    full = tmp(bounds=(0, None), zero_diagonal="exact", **kw)
    res  = tmp(bounds=(0, None), zero_diagonal="exact", reduced=(13, 13),
               **kw)                                 # one block = full
    assert len(res.model) == 1 and res.sweeps <= 2
    assert _close(res.x, full.x, 1e-10)
    for reduced in ((7, 7), (4, 5)):                 # 2 × 2, 4 × 3 blocks
        res = tmp(bounds=(0, None), zero_diagonal="exact", reduced=reduced,
                  **kw)
        assert np.min(res.x) >= 0.0 and np.all(np.diag(res.x) == 0.0)
        assert 1 < res.sweeps < 100
        assert _close(res.x.sum(1) / X.sum(1),       # totals
                      1.0, 1e-7)
        assert _close(res.x.sum(0) / X.sum(0), 1.0, 1e-7)
        assert _ls_residual(res.x, **kw) < 1e-6      # full: ≈ 0
    X    = np.random.default_rng(52).normal(size=(9, 8))
    kw   = dict(b_row=X.sum(1),
                b_col=X.sum(0) + 0.1)                # inconsistent
    res  = tmp(reduced=(4, 4), **kw)                 # min-norm start
    assert res.sweeps == 1 and _close(res.x, tmp(**kw).x, 1e-10)
    idx  = [3, 17, 40, 41, 66]
    kw  |= dict(M=np.eye(72)[idx], b_val=X.ravel()[idx] + 0.2)
    rho  = _ls_residual(tmp(**kw).x, **kw)
    res  = tmp(reduced=(4, 4), **kw)
    assert rho > 0.1 and abs(_ls_residual(res.x, **kw) - rho) < 1e-7

def test_reduced_not_converged_warning():
    X  = _table(55, 10, 10)
    kw = dict(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
              reduced=(6, 6))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = tmp(sweep_limit=1, **kw)
    assert res.sweeps == 1 and len(res.model) == 4
    assert any(issubclass(v.category, RuntimeWarning) and
               "sweep_limit" in str(v.message) for v in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = tmp(**kw)
    assert res.sweeps > 1 and not any("sweep_limit" in str(v.message)
                                      for v in w)
    _raises(lambda: tmp(sweep_limit=0, **kw), text="sweep_limit")

@_bvls_fallback_ok
def test_reduced_known_values_and_per_cell_bounds():
    X     = _table(53, 10, 10)
    rng   = np.random.default_rng(53)
    idx   = np.sort(rng.choice(100, size=20, replace=False))
    kw    = dict(b_row=X.sum(1), b_col=X.sum(0), M=np.eye(100)[idx],
                 b_val=np.minimum(X.ravel()[idx], 1.5))
    lohi  = [(0, 1.5)] * 100
    full  = tmp(bounds=lohi, **kw)
    res   = tmp(bounds=lohi, reduced=(6, 6), **kw)
    assert np.min(res.x) >= 0.0 and np.max(res.x) <= 1.5
    assert abs(_ls_residual(res.x, **kw) -           # traded off as in
               _ls_residual(full.x, **kw)) < 1e-6    # the full model
    assert [v.A.shape[0] - 10 for v in res.model] == [
        int(np.sum((idx // 10 // 5 == r) & (idx % 10 // 5 == c)))
        for r in range(2) for c in range(2)]         # M rows per block
    kw    = dict(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
                 reduced=(6, 6))
    res   = tmp(zero_diagonal="exact", **kw)
    assert np.all(np.diag(res.x) == 0.0)
    res   = tmp(zero_diagonal=True, **kw)            # soft, as M rows
    assert [v.A.shape[0] for v in res.model] == [15, 10, 10, 15]

def test_reduced_symmetric():
    X     = _table(56, 9, 9) * (1 - np.eye(9))
    X     = (X + X.T) / 2
    kw    = dict(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
                 zero_diagonal="exact", symmetric=True)
    res   = tmp(reduced=(5, 5), **kw)                # after the sweeps
    assert _close(res.x, res.x.T, 0.0) and np.min(res.x) >= 0.0
    assert _close(res.x.sum(1), X.sum(1), 1e-7) and res.sweeps > 1
    assert np.all(np.diag(res.x) == 0.0) and not res.replaced.any()

def test_reduced_input_errors():
    X  = _table(54, 6, 6)
    kw = dict(b_row=X.sum(1), b_col=X.sum(0))
    _raises(lambda: tmp(reduced=(2, 4), **kw), text="(3, 3)")
    _raises(lambda: tmp(b_row=X.reshape(3, 12).sum(1), b_col=X.sum(0), i=2,
                        reduced=(4, 4)), text="Grouped")
    _raises(lambda: tmp(M=np.ones((1, 36)), b_val=[1.0], reduced=(4, 4),
                        **kw), text="identity")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tmp(S=np.eye(12), reduced=(4, 4), **kw)
    assert any("ignored" in str(v.message) for v in w)

# 4. Inconsistent totals, Step 2 and the passed options
def test_inconsistent_totals():
    b_row, b_col = [1, 10, 10], [1, 10, 11]          # 21 vs 22
    A            = _ap_matrix(3, 3)
    b            = np.r_[b_row, b_col]
    res          = tmp(b_row=b_row, b_col=b_col)
    r            = np.linalg.norm(b - A @ _min_norm(A, b))
    assert abs(res.model.rho - r) < 1e-8 and abs(res.model.residual - r) < 1e-8
    res          = tmp(b_row=b_row, b_col=b_col, bounds=(0, None))
    _, r         = _bounded_ls(A, b, np.zeros(9), np.full(9, np.inf))
    assert abs(res.model.rho - r) < 1e-8 and abs(r - 0.408248) < 1e-6
    assert abs(res.model.residual - res.model.rho) < 1e-7
    assert res.model.status["step1"] == "converged"
    assert res.model.status["final"] == "optimal" and np.min(res.x) >= 0.0
    assert not _close(res.x.sum(0), b_col, 1e-3)     # soft totals
    for alpha in (0.0, 0.5):                         # equal residual
        fit = tmp(b_row=b_row, b_col=b_col, bounds=(0, None), alpha=alpha)
        assert abs(fit.model.residual - r) < 1e-6 and _converged(fit.model)

def test_alpha_validation():
    kw  = dict(b_row=B_ROW, b_col=B_COL, bounds=(0, None))
    _raises(lambda: tmp(alpha=[0.0, 1.0], **kw), CLSPError, "single value")
    _raises(lambda: tmp(alpha=(0.0, 1.0), reduced=(3, 3), **kw), CLSPError)
    _raises(lambda: tmp(alpha=1.5, **kw), CLSPError, "single value")
    assert tmp(**kw).model.alpha == 1.0              # None means 1
    res = tmp(alpha=0.0, **kw)
    assert res.model.alpha == 0.0 and res.model.status["solver"] == "ECOS"
    res = tmp(alpha=0.5, reduced=(3, 3), **kw)
    assert all(v.alpha == 0.5 for v in res.model)

def test_passthrough_options():
    X   = _table(61, 4, 4)
    idx = [1, 6, 11]
    kw  = dict(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None),
               M=np.eye(16)[idx], b_val=X.ravel()[idx] + 0.1)
    res = tmp(tolerance=1e-10, iteration_limit=20, rcond=1e-6, scale=True,
              r=3, cond_tolerance=1e-12, **kw)
    mod = res.model
    assert mod.tolerance == 1e-10 and mod.iteration_limit == 20
    assert mod.rcond == 1e-6 and mod.status["scale"] and mod.r_max == 3
    res = tmp(final=False, **kw)
    assert res.model.status["final"] == "skipped" and res.model.rho is None
    assert _close(res.x, res.model.zhat[:16], 0.0)
    res = tmp(solver="CLARABEL", **kw)
    assert res.model.status["solver"] == "CLARABEL" and _converged(res.model)

# 5. Bounds input and replace_value
def test_bounds_input_forms():
    kw   = dict(b_row=B_ROW, b_col=B_COL)
    ref  = tmp(bounds=(0, None), **kw).x
    for bounds in ([(0, None)], [(0, None)] * 9, tuple([(0, None)] * 9),
                   [(0, np.inf)] * 9,
                   np.array([[0, np.nan]] * 9), [[0.0, np.nan]] * 9):
        assert _close(tmp(bounds=bounds, **kw).x, ref, 1e-10)
    for bounds in ((None, None), (np.nan, np.nan), (-np.inf, np.inf)):
        res = tmp(bounds=bounds, **kw)
        assert res.model.bounds is None and res.x[0, 0] < 0
    _raises(lambda: tmp(bounds=[(0, None)] * 8, **kw), text="length")
    _raises(lambda: tmp(bounds=(2, 1), **kw), text="low ≤ high")
    _raises(lambda: tmp(bounds=(np.inf, None), **kw), text="low ≤ high")
    _raises(lambda: tmp(bounds=[(0, 1, 2)] * 9, **kw), text="pair")
    _raises(lambda: tmp(bounds="x", **kw), text="pair")
    _raises(lambda: tmp(b_row=[1.0], b_col=[1.0, 2.0]), text="Minimum")
    _raises(lambda: tmp(b_row=[1.0, 2.0]), text="Both")

def test_replace_value_not_converged():
    kw   = dict(b_row=B_ROW, b_col=B_COL, bounds=(0, None))
    fail = _step1_fails(lambda: True)
    res  = _patched(clsp.solver, "_bvls", fail, lambda: tmp(**kw))
    assert res.model.status["step1"] == "not converged: test"
    assert np.all(np.isnan(res.x)) and res.replaced.all()
    assert np.all(np.isfinite(res.model.x))          # model kept intact
    res  = _patched(clsp.solver, "_bvls", fail,
                    lambda: tmp(replace_value=None, **kw))
    assert not res.replaced.any() and np.min(res.x) >= 0.0
    res  = _patched(clsp.solver, "_bvls", fail,
                    lambda: tmp(replace_value=-1.0, **kw))
    assert np.all(res.x == -1.0)
    calls = []
    fail  = _step1_fails(lambda: calls.append(1) or len(calls) % 4 == 2)
    X     = _table(71, 8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = _patched(clsp.solver, "_bvls", fail,
                       lambda: tmp(b_row=X.sum(1), b_col=X.sum(0),
                                   bounds=(0, None), reduced=(5, 5)))
    mask  = np.zeros((8, 8), dtype=bool)
    mask[:4, 4:] = True                              # second block only
    assert len(calls) == 4 * res.sweeps              # one Step 1 per block
    assert np.array_equal(res.replaced, mask)
    assert np.all(np.isnan(res.x[mask])) and np.all(res.x[~mask] >= 0.0)

def test_replace_value_symmetric_asymmetric_bounds():
    X       = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    lohi    = [(0, None)] * 9
    lohi[1] = (0, 0.1)                               # x_12 ≤ 0.1 only
    res     = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=lohi,
                  symmetric=True)
    mask    = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    assert np.array_equal(res.replaced, mask)
    assert np.all(np.isnan(res.x[mask])) and np.all(res.x[~mask] >= 0.0)
    assert res.model.x[0, 1] <= 0.1                  # the CLSP fit
    res     = tmp(b_row=X.sum(1), b_col=X.sum(0), bounds=lohi,
                  symmetric=True, replace_value=None)
    assert not res.replaced.any() and res.x[0, 1] > 0.1
    assert _close(res.x, res.x.T, 0.0)

# Plain-script runner
if __name__ == "__main__":
    tests  = [(k, f) for k, f in list(globals().items())
              if k.startswith("test_") and callable(f)]
    failed = []
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=-2)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
