import warnings
import numbers
import numpy                           as np
from   typing          import Sequence
from   collections.abc import Sequence as sq
from   clsp            import CLSP

class TMPinvInputError(Exception):
    """
    Exception class for TMPinv-related input errors.

    Represents internal failures in Tabular Matrix Problems via Pseudoinverse
    Estimation routines due to malformed or missing input. Supports structured
    messaging and optional diagnostic augmentation.

    Parameters
    ----------
    message : str, optional
        Description of the error. Defaults to a generic TMPinv message.

    code : int or str, optional
        Optional error code or identifier for downstream handling.

    Attributes
    ----------
    message : str
        Human-readable error message.

    code : int or str
        Optional error code for custom handling or debugging.

    Usage
    -----
    raise TMPinvInputError("Both b_row and b_col must be provided", code=201)
    """
    def __init__(self, message: str = "An error occurred in TMPinv",
                 code: int | str | None = None):
        self.message = message
        self.code    = code
        full_message = f"{message} (Code: {code})" if code is not None         \
                                                   else message
        super().__init__(full_message)

    def __str__(self) -> str:
        return self.message if self.code is None                               \
                            else f"{self.message} [Code: {self.code}]"

    def as_dict(self) -> dict:
        """
        Return the error as a dictionary for structured logging or JSON output.
        """
        return {"error": self.message, "code": self.code}

class TMPinvResult:
    """
    Result container for TMPinv estimation.

    Attributes
    ----------
    full : bool
        Indicates if this result comes from the full (non-reduced) model.

    model : CLSP or list of CLSP
        A single CLSP object in the full model, or a list of CLSP objects
        for each reduced block in the reduced model.

    x : np.ndarray
        Final estimated solution matrix of shape (m, p).

    replaced : np.ndarray
        Boolean mask of shape (m, p) of the cells of `x` set to
        `replace_value` in `tmpinv()` (all False if none).

    sweeps : int or None
        Number of block sweeps in the reduced model (None in the full
        model).
    """
    def __init__(self, full: bool, model, x: np.ndarray,
                 replaced: np.ndarray | None = None):
        self.full     = full
        self.model    = model
        self.x        = x
        self.replaced = (np.zeros(np.shape(x), dtype=bool) if replaced is None
                         else replaced)
        self.sweeps   = None

    def summarize(self, i: int | None = None, display : bool = False):
        return self.summary(i=i, display=display)

    def summary(self, i: int | None = None, display: bool = False):
        """
        Return or print a summary for the TMPinv estimator.

        Parameters
        ----------
        display : bool, default = False
            If True, prints the summary instead of returning a dictionary.
        """
        if not isinstance(self.model, list):
            return self.model.summarize(display=display)
        if i is None:
            raise TMPinvInputError("Reduced model: please supply the block " +
                                   "index using i=#.")
        idx = int(i)
        if idx < 0 or idx > len(self.model) - 1:
            raise TMPinvInputError(f"i must be in 0..{len(self.model)}-1 "   +
                                   f"for reduced model.")
        return self.model[idx].summarize(display=display)

def TMPinvSolve(
    S:              Sequence[Sequence[float]] | None = None,
    M:              Sequence[Sequence[float]] | None = None,
    b_row:          Sequence[float]           | None = None,
    b_col:          Sequence[float]           | None = None,
    b_val:          Sequence[float]           | None = None,
    i:              int                              = 1,
    j:              int                              = 1,
    zero_diagonal:  bool                             = False,
    reduced:        tuple[int, int]           | None = None,
    symmetric:      bool                             = False,
    bounds:         list[tuple[float | None,
                               float | None]] | None = None,
    slack_sign:     bool                      | None = None,
    cond_tolerance: float                     | None = None,
    sweep_limit:    int                              = 100,
    *args, **kwargs
) -> TMPinvResult:
    """
    Solve a tabular matrix estimation problem via Convex Least Squares
    Programming (CLSP).

    Parameters
    ----------
    S : array_like of shape (m + p, m + p), optional
        A diagonal sign slack (surplus) matrix with entries in {0, ±1}.
        -  0 enforces equality (== b_row or b_col),
        -  1 enforces a lower-than-or-equal (≤) condition,
        - –1 enforces a greater-than-or-equal (≥) condition.
        The first `m` diagonal entries correspond to row constraints,
        and the remaining `p` to column constraints. In the grouped case,
        `m` and `p` are the lengths of `b_row` and `b_col`. The conditions
        require signed slacks (`slack_sign`); the rows remain least-squares
        rows on their infeasible side.
    M : array_like of shape (k, m * p), optional
        A model matrix, for example, with entries in {0, 1}. Each row defines a
        linear restriction on the flattened solution matrix. The corresponding
        right-hand side values must be provided in `b_val`. This block is
        used to encode known cell values (least-squares rows, like the
        totals).
    b_row : array_like of shape (m,)
        Right-hand side vector of row totals.
    b_col : array_like of shape (p,)
        Right-hand side vector of column totals.
    b_val : array_like of shape (k,)
        Right-hand side vector of known cell values.
    i : int, default = 1
        Number of row groups.
    j : int, default = 1
        Number of column groups.
    zero_diagonal : bool, default = False
        If True, adds the soft model rows x_kk = 0 (least-squares rows, see
        `tmpinv()` for an exact zero diagonal).
    reduced : tuple of (int, int), optional
        Dimensions of the reduced problem.
        If specified, the problem is estimated by block-coordinate sweeps
        over contiguous submatrices of the original table. For example,
        reduced = (6, 6) implies 5×5 data blocks (edge blocks may be
        smaller).
    symmetric : bool, default = False
        If True, enforces symmetry of the estimated solution matrix as:
        x = 0.5 * (x + x.T)
        Applies to TMPinvResult.x only (after the sweeps in the reduced
        model). For TMPinvResult.model symmetry, add explicit symmetry
        constraints to M in a full-model solve instead of using this flag.
    bounds : list of (low, high) or None, default = None
        Bounds on the m * p cells, one pair per cell (None = unbounded
        side), as normalized by `tmpinv()`. Enforced exactly by CLSP.
    slack_sign : bool or None, default = None
        If True, the slack variables of `S` must be ≥ 0. If None, True if
        `S` is supplied, False otherwise. Ignored in the reduced model.
    cond_tolerance : float or None, default = None
        Singular-value cutoff for the custom condition number function.
        If None, the implementation uses an internal relative cutoff of 1e-14.
    sweep_limit : int, default = 100
        Maximum number of block sweeps in the reduced model. The sweeps
        stop when the largest change of a cell is below `tolerance` (of
        CLSP) times max(1, max |x|); otherwise, a RuntimeWarning is issued.

    Returns
    -------
    TMPinvResult
        An object containing the fitted CLSP model(s) and the solution
        matrix `x`.

    Notes
    ----
    - In the reduced model, the full problem is solved by block-coordinate
    (Gauss–Seidel) sweeps: each block is fitted by CLSP with the other
    blocks fixed (the totals minus the sums of the other blocks, its known
    values and its bounds), starting from the minimum-norm table of the
    totals clipped to the bounds. The sweeps converge toward a least-squares
    fit of the full problem, not necessarily the minimum-norm one.
    - In the reduced model, `S` is ignored, and `M` must consist of rows
    of an identity matrix (known cells). Repeated cells, e.g. a known
    diagonal cell with zero_diagonal=True, are kept as separate rows of
    their block (their mean is used only for the start). Arbitrary or
    non-diagonal model matrices cannot be mapped to reduced blocks.
    Grouped totals (i, j > 1) are not supported in the reduced model.
    - `bounds` are sliced block-wise in the reduced model.
    """
    # (m), (p) Process the parameters, assert conformity, and get dimensions
    if  b_row is None or b_col is None:
        raise TMPinvInputError("Both b_row and b_col must be provided.")
    if  len(b_row) < 2 or len(b_col) < 2:
        raise TMPinvInputError("Minimum length for b_row and b_col is 2.")
    b_row = np.asarray(b_row, dtype=np.float64).reshape(-1, 1)
    b_col = np.asarray(b_col, dtype=np.float64).reshape(-1, 1)
    if  not np.isfinite(b_row).all() or not np.isfinite(b_col).all():
        raise TMPinvInputError("b_row and b_col must not contain inf or NaN.")
    m     = b_row.shape[0] * i
    p     = b_col.shape[0] * j
    if  S     is not None:
        n_rows = b_row.shape[0] + b_col.shape[0]
        S      = np.asarray(S, dtype=np.float64)
        if S.shape[0] != n_rows:
            raise TMPinvInputError(f"S must have {n_rows} rows.")
        if (not np.all((S == -1) | (S == 0) | (S == 1)) or
                np.abs(S).sum(axis=0).max() > 1         or
                np.abs(S).sum(axis=1).max() > 1):
            raise TMPinvInputError("S must be a zero-padded subset of ±I.")
    if  M     is not None:
        if  b_val is None:
            raise TMPinvInputError("Both M and b_val must be defined.")
        M = np.asarray(M, dtype=np.float64)
        if  M.ndim == 1:
            M = M.reshape(1, -1)
        if  M.shape[1] != m * p:
            raise TMPinvInputError(f"M must have exactly {m * p} columns.")
    if  b_val is not None:
        if  M     is None:
            raise TMPinvInputError("Both M and b_val must be defined.")
        if not np.isfinite(b_val).all():
            raise TMPinvInputError("b_val must not contain inf or NaN.")
        b_val = np.asarray(b_val, dtype=np.float64).reshape(-1, 1)
    if  M is not None and b_val is not None:
        if  M.shape[0] != b_val.shape[0]:
            raise TMPinvInputError(f"M and b_val must have the same number "
                                   f"of rows: "
                                   f"{M.shape[0]} vs {b_val.shape[0]}")
    if  bounds is not None and len(bounds) != m * p:
        raise TMPinvInputError(f"bounds must have exactly {m * p} pairs.")
    kw = {k: v for k, v in kwargs.items() if k not in {"b",
                                                       "S", "M",
                                                       "m", "p",
                                                       "i", "j",
                                                       "zero_diagonal",
                                                       "symmetric", "bounds",
                                                       "slack_sign"}}

    # perform full estimation and return the result
    if  reduced is None:
        result   = TMPinvResult(True,  None, np.empty((m, p), dtype=float))
        b_blocks = [b_row, b_col]
        if b_val is not None:
            b_blocks.append(b_val)
        b = np.vstack(b_blocks)
        result.model = CLSP().solve(*args,
                                    problem='ap', b=b,
                                                  S=S,
                                                  M=M,
                                                  m=m,
                                                  p=p,
                                                  i=i,
                                                  j=j,
                                      zero_diagonal=zero_diagonal,
                                             bounds=bounds,
                                         slack_sign=(S is not None
                                                     if slack_sign is None
                                                     else slack_sign),
                                     cond_tolerance=cond_tolerance,
                                    **kw)
        result.x = result.model.x
        result.replaced[:, :] = result.model.status["step1"].startswith(
                                    "not converged")

    # perform reduced estimation by block sweeps and return the result
    else:
        reduced  = tuple(map(int, reduced))
        if  reduced[0] < 3 or reduced[1] < 3:
            raise TMPinvInputError("Each reduced block must be at least "
                                   "(3, 3), i.e. 2×2 data blocks.")
        if  i != 1 or j != 1:
            raise TMPinvInputError("Grouped totals (i, j > 1) are not "
                                   "supported in the reduced model.")
        if  int(sweep_limit) < 1:
            raise TMPinvInputError("sweep_limit must be at least 1.")
        result   = TMPinvResult(False, [], np.empty((m, p), dtype=float))
        m_subset = reduced[0] - 1
        p_subset = reduced[1] - 1
        tol      = (np.sqrt(np.finfo(float).eps)
                    if kwargs.get("tolerance") is None
                    else float(kwargs["tolerance"]))

        if  zero_diagonal:                             # not processed by CLSP
            M_diag = np.zeros((min(m, p), m * p))
            b_diag = np.zeros((min(m, p),     1))
            for k in range(min(m, p)):
                M_diag[k, k * p + k] = 1
            M      = (M_diag.copy()          if M     is None or M.size     == 0
                                             else np.vstack([M,     M_diag]))
            b_val  = (b_diag.copy()          if b_val is None or b_val.size == 0
                                             else np.vstack([b_val, b_diag]))
            del M_diag, b_diag
        # (cells), (vals) Cell and value of every model row, repeated cells
        # kept as separate rows; (X_true) their mean, used only as a start
        cells  = np.empty(0, dtype=int)
        vals   = np.empty(0, dtype=np.float64)
        if  M is not None:
            if  not ((np.isclose(M, 0) | np.isclose(M, 1)).all() and
                     (np.isclose(M, 1).sum(axis=1) == 1).all()):
                raise TMPinvInputError("M must consist of rows of the "
                                       "identity matrix in the reduced model.")
            cells  = np.argmax(M, axis=1)              # M has m * p columns
            vals   = b_val.ravel(order='C')
        n_obs  = np.bincount(cells, minlength=m * p).reshape(m, p)
        X_true = np.where(n_obs > 0,
                          np.bincount(cells, vals, m * p).reshape(m, p) /
                          np.maximum(n_obs, 1),
                          np.nan)
        r_obs, c_obs = np.divmod(cells, p)

        if  S is not None and np.size(S) > 0:
            warnings.warn("User-provided S is ignored in the reduced model.",
                          UserWarning)

        # (x_lb), (x_ub) Bounds on the cells
        x_lb   = np.full((m, p), -np.inf)
        x_ub   = np.full((m, p),  np.inf)
        if  bounds is not None:
            x_lb = np.array([-np.inf if l is None else l
                             for l, _ in bounds]).reshape(m, p)
            x_ub = np.array([ np.inf if h is None else h
                             for _, h in bounds]).reshape(m, p)

        # (x) Start: min-norm fit of the totals, known values, bounds
        shift  = (b_col.sum() - b_row.sum()) / (m + p) # LS shift of totals
        x      = ((b_row + shift) / p + (b_col.T - shift) / m -
                  (b_row.sum() + m * shift) / (m * p))
        x      = np.clip(np.where(np.isnan(X_true), x, X_true), x_lb, x_ub)
        blocks = [(m_start, min(m_start + m_subset, m),
                   p_start, min(p_start + p_subset, p))
                  for m_start in range(0, m, m_subset)
                  for p_start in range(0, p, p_subset)]

        # (x) Block-coordinate (Gauss–Seidel) sweeps on the full problem
        for n_sweep in range(1, 1 + int(sweep_limit)):
            x_prev       = x.copy()
            result.model = []
            for m_start, m_end, p_start, p_end in blocks:
                x_rest   = x.copy()                    # other blocks
                x_rest[m_start:m_end, p_start:p_end] = 0.0
                B_subset = None
                b_subset = [(b_row - x_rest.sum(axis=1, keepdims=True)
                            )[m_start:m_end],
                            (b_col - x_rest.sum(axis=0).reshape(-1, 1)
                            )[p_start:p_end]]
                M_subset = None
                if bounds              is not None:
                    B_subset     = [bounds[r * p + c]
                                    for r in range(m_start, m_end)
                                    for c in range(p_start, p_end)]
                if M                   is not None:
                    in_block     = ((r_obs >= m_start) & (r_obs < m_end) &
                                    (c_obs >= p_start) & (c_obs < p_end))
                    if in_block.any():                 # the original rows
                        local    = ((r_obs - m_start) * (p_end - p_start) +
                                    (c_obs - p_start))[in_block]
                        b_subset.append(vals[in_block].reshape(-1, 1))
                        M_subset = np.eye((m_end - m_start) *
                                          (p_end - p_start))[local, :]
                tmp      = CLSP().solve(*args,
                                        problem='ap', b=np.vstack(b_subset),
                                                      M=M_subset,
                                                      m=m_end - m_start,
                                                      p=p_end - p_start,
                                                      i=i,
                                                      j=j,
                                          zero_diagonal=False,
                                                 bounds=B_subset,
                                         cond_tolerance=cond_tolerance,
                                        **kw)
                result.model.append(tmp)
                x[m_start:m_end, p_start:p_end] = tmp.x
                result.replaced[m_start:m_end, p_start:p_end] = (
                    tmp.status["step1"].startswith("not converged"))
            change = np.max(np.abs(x - x_prev))
            if  change < tol * max(1.0, np.max(np.abs(x))):
                break
        else:
            warnings.warn(f"Reduced model: the block sweeps did not converge "
                          f"in {n_sweep} sweeps (max change {change:.3e}); "
                          f"increase sweep_limit.", RuntimeWarning)
        result.x      = x
        result.sweeps = n_sweep

    # enforce symmetry
    if  symmetric:
        if  result.x.shape[0] != result.x.shape[1]:
            raise TMPinvInputError("symmetric=True requires a square matrix "
                                   "(m == p).")
        result.x = float(0.5) * (result.x + result.x.T)

    return result

def tmpinv(
    bounds:          list[tuple[float | None, float | None]] | \
                          tuple[float | None, float | None]  | None = None,
    replace_value:   float                                   | None = np.nan,
    tolerance:       float = np.sqrt(np.finfo(float).eps),
    iteration_limit: int   = 50,
    cond_tolerance:  float                                   | None = None,
    *args, **kwargs
) -> TMPinvResult:
    """
    Solve a tabular matrix estimation problem via Convex Least Squares
    Programming (CLSP) with exactly enforced cell-value bounds.

    Parameters
    ----------
    bounds : sequence of (low, high), optional
        Bounds on cell values. If a single tuple (low, high) is given, it
        is applied to all m * p cells. Example: (0, None). None, np.nan and
        ±np.inf denote an unbounded side, low = high fixes a cell. The
        bounds are enforced exactly by CLSP (block-wise in the reduced
        model).
    replace_value : float or None, default = np.nan
        Replacement value for the cells of a full or block model whose
        bounded first step did not converge (status "step1" starting with
        'not converged') and for the cells that violate their bounds by
        more than `tolerance` after symmetrization. If None, no cell is
        replaced. The replaced cells are marked in TMPinvResult.replaced.
    tolerance : float, optional
        Tolerance of CLSP, of the bound check of `replace_value` and of
        the sweeps of the reduced model. Default is the square root of
        machine epsilon.
    iteration_limit : int, default = 50
        Maximum number of iterations of the refinement loop and of the
        passes of the bounded first step in CLSP.
    cond_tolerance : float or None, default = None
        Singular-value cutoff for the custom condition number function.
        If None, the implementation uses an internal relative cutoff of 1e-14.
    *args, **kwargs : additional arguments
        Passed directly to TMPinvSolve() (e.g., reduced, sweep_limit) and
        CLSP (e.g., r, Z, rcond, final, alpha, scale, slack_sign, and CVXPY
        solver arguments). In addition, zero_diagonal='exact' sets the
        bounds of the diagonal cells to (0, 0) instead of adding soft model
        rows (zero_diagonal=True).

    Returns
    -------
    TMPinvResult
        An object containing the fitted CLSP model(s), the solution
        matrix `x`, and the mask `replaced`.
    """
    # (n_cells) Get cell count from keyword arguments
    if  kwargs.get("b_row") is None or kwargs.get("b_col") is None:
        raise TMPinvInputError("Both b_row and b_col must be provided.")
    if  len(kwargs.get("b_row")) < 2 or len(kwargs.get("b_col")) < 2:
        raise TMPinvInputError("Minimum length for b_row and b_col is 2.")
    m       = len(kwargs.get("b_row")) * kwargs.get("i", 1)
    p       = len(kwargs.get("b_col")) * kwargs.get("j", 1)
    n_cells = m * p
    if  bounds is None:
        bounds = (None, None)
    if  isinstance(bounds, np.ndarray):
        bounds = bounds.tolist()
    if  (isinstance(bounds, sq) and len(bounds) == 2 and
         all(v is None or isinstance(v, numbers.Real) for v in bounds)):
        bounds = [tuple(bounds)] * n_cells             # replicate (low, high)
    elif   isinstance(bounds, sq):                     # normalize bounds
        if len(bounds) > 1 and len(bounds) != n_cells:
            raise TMPinvInputError(f"Bounds length {len(bounds)} does not "
                                   f"match number of variables {n_cells}.")
        elif len(bounds) == 1:
            bounds = list(bounds) * n_cells            # replicate (low, high)
    else:
        raise TMPinvInputError("bounds must be None, a (low, high) pair, or "
                               "a sequence of (low, high) pairs.")
    try:                                               # unbounded: None
        bounds = [(None if l is None or np.isnan(l) or l == -np.inf
                        else float(l),
                   None if h is None or np.isnan(h) or h ==  np.inf
                        else float(h)) for l, h in bounds]
    except (TypeError, ValueError):
        raise TMPinvInputError("bounds must be None, a (low, high) pair, or "
                               "a sequence of (low, high) pairs.")
    if  any(l is not None and h is not None and l > h or
            l is not None and np.isinf(l) or h is not None and np.isinf(h)
            for l, h in bounds):
        raise TMPinvInputError("bounds must satisfy low ≤ high.")

    # (zero_diagonal) Set the bounds of the diagonal cells to (0, 0)
    zero_diagonal = kwargs.get("zero_diagonal", False)
    if  isinstance(zero_diagonal, str):
        if  zero_diagonal != "exact":
            raise TMPinvInputError("zero_diagonal must be True, False or "
                                   "'exact'.")
        for k in range(min(m, p)):
            l, h = bounds[k * p + k]
            if (l is not None and l > 0) or (h is not None and h < 0):
                raise TMPinvInputError("zero_diagonal='exact' requires "
                                       "bounds that admit 0 on the "
                                       "diagonal.")
            bounds[k * p + k] = (0.0, 0.0)

    # (result) Perform estimation with bounds enforced by CLSP
    kw     = {k: v for k, v in kwargs.items() if k not in {"zero_diagonal"}}
    result = TMPinvSolve(*args, bounds=(bounds if any(l is not None or
                                                      h is not None
                                                      for l, h in bounds)
                                        else None),
                                zero_diagonal=(False
                                               if isinstance(zero_diagonal,
                                                             str)
                                               else bool(zero_diagonal)),
                                tolerance=tolerance,
                                iteration_limit=iteration_limit,
                                cond_tolerance=cond_tolerance,
                                **kw)

    # (result) Replace non-converged and out-of-bound values
    x        = result.x.reshape(-1, 1)
    x_lb     = np.array([l if l is not None else -np.inf
                         for  l, _ in bounds]).reshape(-1, 1)
    x_ub     = np.array([h if h is not None else  np.inf
                         for  _, h in bounds]).reshape(-1, 1)
    replaced = (result.replaced.reshape(-1, 1)                     |
                (x < x_lb - tolerance) | (x > x_ub + tolerance)).reshape(
                result.x.shape)
    if  kwargs.get("symmetric", False):
        replaced = replaced | replaced.T               # (final) symmetry
    if  replace_value is None:
        replaced = np.zeros_like(replaced)
    else:
        result.x = result.x.copy()
        result.x[replaced] = replace_value
    result.replaced = replaced

    return result
