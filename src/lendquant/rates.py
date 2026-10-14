"""Rate conversions per SIFMA Standard Formulas.

All functions are scalar/array-polymorphic: pass a Python float to get a
float back, pass a numpy array to get an array of the same shape back.

References
----------
- SIFMA, *Standard Formulas for the Analysis of Mortgage-Backed Securities
  and Other Related Securities* (CPR/SMM, CDR/MDR conventions).
- Brealey, Myers, Allen, *Principles of Corporate Finance* §3 (nominal
  APR convention on monthly-pay US loans).
- Hull, *Options, Futures, and Other Derivatives* §4.2 (effective-rate
  compounding).
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "annualize_monthly",
    "apr_from_monthly",
    "cdr_from_mdr",
    "cpr_from_smm",
    "mdr_from_cdr",
    "monthly_from_annualized",
    "monthly_from_apr",
    "smm_from_cpr",
]

_FloatArr = NDArray[np.floating]


@overload
def smm_from_cpr(cpr: float) -> float: ...
@overload
def smm_from_cpr(cpr: _FloatArr) -> _FloatArr: ...
def smm_from_cpr(cpr: float | _FloatArr) -> float | _FloatArr:
    """Single Monthly Mortality from Conditional Prepayment Rate.

    SIFMA Standard Formulas: ``smm = 1 - (1 - cpr) ** (1/12)``.
    """
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


@overload
def cpr_from_smm(smm: float) -> float: ...
@overload
def cpr_from_smm(smm: _FloatArr) -> _FloatArr: ...
def cpr_from_smm(smm: float | _FloatArr) -> float | _FloatArr:
    """Conditional Prepayment Rate from Single Monthly Mortality.

    SIFMA Standard Formulas: ``cpr = 1 - (1 - smm) ** 12``.
    """
    return 1.0 - (1.0 - smm) ** 12


@overload
def mdr_from_cdr(cdr: float) -> float: ...
@overload
def mdr_from_cdr(cdr: _FloatArr) -> _FloatArr: ...
def mdr_from_cdr(cdr: float | _FloatArr) -> float | _FloatArr:
    """Monthly Default Rate from Conditional Default Rate.

    SIFMA Standard Formulas (same shape as SMM/CPR):
    ``mdr = 1 - (1 - cdr) ** (1/12)``.
    """
    return 1.0 - (1.0 - cdr) ** (1.0 / 12.0)


@overload
def cdr_from_mdr(mdr: float) -> float: ...
@overload
def cdr_from_mdr(mdr: _FloatArr) -> _FloatArr: ...
def cdr_from_mdr(mdr: float | _FloatArr) -> float | _FloatArr:
    """Conditional Default Rate from Monthly Default Rate.

    SIFMA Standard Formulas: ``cdr = 1 - (1 - mdr) ** 12``.
    """
    return 1.0 - (1.0 - mdr) ** 12


@overload
def monthly_from_apr(apr: float) -> float: ...
@overload
def monthly_from_apr(apr: _FloatArr) -> _FloatArr: ...
def monthly_from_apr(apr: float | _FloatArr) -> float | _FloatArr:
    """Monthly rate from nominal APR (Brealey §3): ``r_m = apr / 12``."""
    return apr / 12.0


@overload
def apr_from_monthly(r_m: float) -> float: ...
@overload
def apr_from_monthly(r_m: _FloatArr) -> _FloatArr: ...
def apr_from_monthly(r_m: float | _FloatArr) -> float | _FloatArr:
    """Nominal APR from monthly rate: ``apr = 12 * r_m``."""
    return 12.0 * r_m


@overload
def monthly_from_annualized(ann: float) -> float: ...
@overload
def monthly_from_annualized(ann: _FloatArr) -> _FloatArr: ...
def monthly_from_annualized(ann: float | _FloatArr) -> float | _FloatArr:
    """Monthly rate from effective annualized rate (Hull §4.2):
    ``r_m = (1 + ann) ** (1/12) - 1``.
    """
    return (1.0 + ann) ** (1.0 / 12.0) - 1.0


@overload
def annualize_monthly(r_m: float) -> float: ...
@overload
def annualize_monthly(r_m: _FloatArr) -> _FloatArr: ...
def annualize_monthly(r_m: float | _FloatArr) -> float | _FloatArr:
    """Effective annualized rate from monthly rate (Hull §4.2):
    ``ann = (1 + r_m) ** 12 - 1``.
    """
    return (1.0 + r_m) ** 12 - 1.0
