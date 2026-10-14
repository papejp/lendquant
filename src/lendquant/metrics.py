"""Yield & valuation metrics on cashflow arrays.

**Time convention (uniform across this module):** ``cashflows[i]`` is the
cashflow at time index ``i`` months from today (0-indexed). For a loan
with initial outflow at t=0 and receipts at months 1..N, pass
``[-P, c_1, c_2, ..., c_N]`` (length N+1). For an amortization schedule
whose first row is month 1, prepend a 0 to measure WAL / duration from
t=0 (otherwise the result is shifted one month early).

References
----------
- Brealey, Myers, Allen, *Principles of Corporate Finance* §5
  (IRR, NPV).
- Hull, *Options, Futures, and Other Derivatives* §4.2, §4.11
  (compounding conventions, Macaulay / modified duration).
- Fabozzi, *The Handbook of Mortgage-Backed Securities* §9
  (weighted average life).
- Excel ``XIRR`` convention (365.25-day year fraction).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from lendquant.rates import annualize_monthly, monthly_from_annualized

__all__ = [
    "irr",
    "loss_adjusted_yield",
    "macaulay_duration",
    "modified_duration",
    "npv",
    "price_from_yield",
    "wal",
    "xirr",
    "yield_from_price",
]

# Rate search bracket for Brent's method. Covers everything realistic for
# monthly IRR / annualized YTM; extreme distressed rates (<-99%) are out
# of scope.
_RATE_LO = -0.999999
_RATE_HI = 10.0
_SCAN_POINTS = 500


def _scan_bracket(
    f: Callable[[float], float], lo: float, hi: float, n: int
) -> tuple[float, float] | None:
    """Scan [lo, hi] for the first consecutive-pair sign change in ``f``."""
    xs = np.linspace(lo, hi, n)
    prev_x = xs[0]
    try:
        prev_y = f(prev_x)
    except (OverflowError, ValueError, ZeroDivisionError):
        prev_y = math.nan
    for x in xs[1:]:
        try:
            y = f(x)
        except (OverflowError, ValueError, ZeroDivisionError):
            y = math.nan
        if (
            math.isfinite(prev_y)
            and math.isfinite(y)
            and prev_y * y < 0
        ):
            return (prev_x, x)
        prev_x, prev_y = x, y
    return None


def _solve_rate(f: Callable[[float], float]) -> float:
    """Find a root of ``f`` on the standard rate bracket; return nan if none.

    Numpy floating-point warnings (divide-by-zero, overflow) are silenced
    here — the scan deliberately probes extreme rates to find a sign
    change, and the resulting infinities are expected signal, not an
    error worth surfacing to the caller.
    """
    with np.errstate(all="ignore"):
        try:
            f_lo = f(_RATE_LO)
            f_hi = f(_RATE_HI)
        except (OverflowError, ValueError, ZeroDivisionError):
            f_lo = f_hi = math.nan
        if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi < 0:
            try:
                return float(brentq(f, _RATE_LO, _RATE_HI, xtol=1e-12, maxiter=200))
            except (RuntimeError, ValueError):
                pass
        bracket = _scan_bracket(f, _RATE_LO, _RATE_HI, _SCAN_POINTS)
        if bracket is None:
            return math.nan
        try:
            return float(brentq(f, bracket[0], bracket[1], xtol=1e-12, maxiter=200))
        except (RuntimeError, ValueError):
            return math.nan


def npv(cashflows: ArrayLike, rate_m: float) -> float:
    """Net present value at a monthly rate.

    ``NPV = sum(cf_t / (1 + r_m)^t)`` for ``t = 0, 1, ..., len(cashflows)-1``.
    """
    cfs = np.asarray(cashflows, dtype=np.float64)
    if cfs.size == 0:
        return 0.0
    t = np.arange(cfs.size, dtype=np.float64)
    return float(np.sum(cfs / (1.0 + rate_m) ** t))


def irr(cashflows: ArrayLike) -> float:
    """Annualized IRR of uniformly-spaced monthly cashflows (Brealey §5).

    Solves ``NPV(r_m) = 0`` via Brent's method, then annualizes with
    :func:`lendquant.rates.annualize_monthly`. Returns ``nan`` if the
    cashflow array has no sign change (no IRR is defined).
    """
    cfs = np.asarray(cashflows, dtype=np.float64)
    if cfs.size < 2:
        return math.nan
    if not (np.any(cfs > 0) and np.any(cfs < 0)):
        return math.nan

    t = np.arange(cfs.size, dtype=np.float64)

    def residual(r_m: float) -> float:
        return float(np.sum(cfs / (1.0 + r_m) ** t))

    r_m = _solve_rate(residual)
    if not math.isfinite(r_m):
        return math.nan
    return float(annualize_monthly(r_m))


def xirr(cashflows: ArrayLike, dates: ArrayLike) -> float:
    """Date-aware IRR for irregularly-spaced cashflows (Excel XIRR convention).

    Year fraction is ``(d - d0) / 365.25``. The returned rate is
    already annualized (effective). Returns ``nan`` when the cashflow
    array has no sign change.
    """
    cfs = np.asarray(cashflows, dtype=np.float64)
    ds = np.asarray(dates, dtype="datetime64[D]")
    if cfs.size != ds.size:
        raise ValueError("cashflows and dates must have the same length")
    if cfs.size < 2:
        return math.nan
    if not (np.any(cfs > 0) and np.any(cfs < 0)):
        return math.nan

    year_frac = (ds - ds[0]).astype("timedelta64[D]").astype(np.float64) / 365.25

    def residual(r_annual: float) -> float:
        return float(np.sum(cfs / (1.0 + r_annual) ** year_frac))

    return _solve_rate(residual)


def price_from_yield(cashflows: ArrayLike, ytm_annualized: float) -> float:
    """PV of cashflows at the monthly rate implied by an effective YTM."""
    r_m = monthly_from_annualized(ytm_annualized)
    return npv(cashflows, r_m)


def yield_from_price(cashflows: ArrayLike, price: float) -> float:
    """Effective YTM for which the PV of cashflows equals ``price``."""
    cfs = np.asarray(cashflows, dtype=np.float64)
    if cfs.size < 1:
        return math.nan

    def residual(ytm: float) -> float:
        return price_from_yield(cfs, ytm) - price

    return _solve_rate(residual)


def wal(principal_cashflows: ArrayLike) -> float:
    """Weighted average life in years (Fabozzi §9).

    ``WAL = sum(t * prin_t) / sum(prin_t) / 12`` with
    ``t = 0, 1, ..., len-1``.
    """
    arr = np.asarray(principal_cashflows, dtype=np.float64)
    total = float(np.sum(arr))
    if total == 0.0:
        return math.nan
    t = np.arange(arr.size, dtype=np.float64)
    return float(np.sum(t * arr) / total / 12.0)


def macaulay_duration(cashflows: ArrayLike, ytm_annualized: float) -> float:
    """Macaulay duration in years (Hull §4.11).

    ``D = sum(t * cf_t * (1 + r_m)^-t) / sum(cf_t * (1 + r_m)^-t) / 12``
    with ``t = 0, 1, ..., len-1`` and
    ``r_m = monthly_from_annualized(ytm_annualized)``.
    """
    cfs = np.asarray(cashflows, dtype=np.float64)
    if cfs.size == 0:
        return math.nan
    r_m = monthly_from_annualized(ytm_annualized)
    t = np.arange(cfs.size, dtype=np.float64)
    discount = (1.0 + r_m) ** -t
    pv = cfs * discount
    total_pv = float(np.sum(pv))
    if total_pv == 0.0:
        return math.nan
    return float(np.sum(t * pv) / total_pv / 12.0)


def modified_duration(cashflows: ArrayLike, ytm_annualized: float) -> float:
    """Modified duration = Macaulay / (1 + YTM) (Hull §4.11)."""
    mac = macaulay_duration(cashflows, ytm_annualized)
    return mac / (1.0 + ytm_annualized)


def loss_adjusted_yield(
    cashflows: ArrayLike,
    ytm_annualized: float,
    loss_cashflows: ArrayLike,
) -> float:
    """Yield of ``cashflows - loss_cashflows`` at the ``cashflows`` price.

    Prices the clean cashflows at ``ytm_annualized`` to get a benchmark
    price, then solves for the yield of the loss-adjusted stream at
    that same price — i.e., the YTM an investor realizes once modeled
    losses are netted against receipts.
    """
    cfs = np.asarray(cashflows, dtype=np.float64)
    losses = np.asarray(loss_cashflows, dtype=np.float64)
    if cfs.size != losses.size:
        raise ValueError("cashflows and loss_cashflows must have the same length")
    price = price_from_yield(cfs, ytm_annualized)
    adjusted: NDArray[np.float64] = cfs - losses
    return yield_from_price(adjusted, price)
