"""Level-pay amortization and pool-level cashflows.

All inputs are user-supplied; nothing is forecast here. The pool model
applies a monthly default rate (MDR, from CDR) and a single-month-mortality
(SMM, from CPR) to the beginning balance, then amortizes the surviving
performing balance over the remaining term — the account-fraction
convention of Fabozzi §11–§12. Severity is loss-given-default; recovery
is ``(1 - severity)`` of the defaulted amount, lagged.

References
----------
- Fabozzi, *The Handbook of Mortgage-Backed Securities* §11–§12
  (static-pool cashflow model, CDR/CPR/severity vectors).
- Brealey, Myers, Allen, *Principles of Corporate Finance* §3
  (level-pay annuity formula).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = ["amortization_schedule", "level_payment", "pool_cashflows"]


def level_payment(principal: float, rate_m: float, nper: int) -> float:
    """Level amortization payment (Brealey §3).

    ``PMT = P * r_m / (1 - (1 + r_m) ** -n)``. The ``r_m == 0`` edge
    collapses to ``P / n``.
    """
    if nper <= 0:
        raise ValueError("nper must be positive")
    if rate_m == 0.0:
        return principal / nper
    return principal * rate_m / (1.0 - (1.0 + rate_m) ** -nper)


def amortization_schedule(
    principal: float, rate_m: float, nper: int
) -> pd.DataFrame:
    """Monthly schedule for a performing level-pay loan.

    Columns: ``mob`` (1..nper), ``pay`` (level payment, constant),
    ``interest``, ``principal``, ``balance`` (ending balance after
    month-``mob`` payment).
    """
    pay = level_payment(principal, rate_m, nper)
    balances = np.empty(nper, dtype=np.float64)
    interests = np.empty(nper, dtype=np.float64)
    principals = np.empty(nper, dtype=np.float64)
    balance = float(principal)
    for i in range(nper):
        interest = balance * rate_m
        prin = pay - interest
        balance -= prin
        interests[i] = interest
        principals[i] = prin
        balances[i] = balance
    return pd.DataFrame(
        {
            "mob": np.arange(1, nper + 1),
            "pay": pay,
            "interest": interests,
            "principal": principals,
            "balance": balances,
        }
    )


def pool_cashflows(
    balance: float,
    rate_m: float,
    nper: int,
    cdr: ArrayLike,
    cpr: ArrayLike,
    severity: ArrayLike,
    recovery_lag_months: int = 0,
) -> pd.DataFrame:
    """Vector-driven pool cashflow (Fabozzi §11–§12 account-fraction convention).

    At each MOB:

    1. Default: ``MDR_t`` of the beginning balance defaults
       (``MDR = 1 - (1 - CDR) ** (1/12)``).
    2. Prepay: ``SMM_t`` of the post-default balance prepays in full
       (``SMM = 1 - (1 - CPR) ** (1/12)``).
    3. Scheduled amortization: survivors pay interest on the current
       balance plus the level-pay principal computed over the remaining
       term.
    4. Recovery: ``(1 - severity_t)`` of the defaulted amount lands
       ``recovery_lag_months`` months later.

    The returned DataFrame has ``nper + recovery_lag_months`` rows (so
    recoveries from the last defaults always land inside the frame).

    Columns: ``mob, scheduled_prin, scheduled_int, prepay_prin,
    default_prin, recovery, net_cashflow, surviving_fraction``.
    ``net_cashflow`` is gross receipts to the investor
    (``scheduled_int + scheduled_prin + prepay_prin + recovery``);
    losses are tracked separately via ``default_prin`` and ``recovery``
    so that callers can compose their own loss-adjusted series.
    """
    if nper <= 0:
        raise ValueError("nper must be positive")
    if recovery_lag_months < 0:
        raise ValueError("recovery_lag_months must be non-negative")

    cdr_arr = np.broadcast_to(np.asarray(cdr, dtype=np.float64), (nper,)).copy()
    cpr_arr = np.broadcast_to(np.asarray(cpr, dtype=np.float64), (nper,)).copy()
    sev_arr = np.broadcast_to(np.asarray(severity, dtype=np.float64), (nper,)).copy()

    mdr_arr = 1.0 - (1.0 - cdr_arr) ** (1.0 / 12.0)
    smm_arr = 1.0 - (1.0 - cpr_arr) ** (1.0 / 12.0)

    total_periods = nper + recovery_lag_months
    sched_prin = np.zeros(total_periods, dtype=np.float64)
    sched_int = np.zeros(total_periods, dtype=np.float64)
    prepay_prin = np.zeros(total_periods, dtype=np.float64)
    default_prin = np.zeros(total_periods, dtype=np.float64)
    recovery = np.zeros(total_periods, dtype=np.float64)
    surviving_fraction = np.zeros(total_periods, dtype=np.float64)

    bal = float(balance)
    for t in range(nper):
        default_amt = bal * mdr_arr[t]
        bal_after_default = bal - default_amt
        prepay_amt = bal_after_default * smm_arr[t]
        bal_after_prepay = bal_after_default - prepay_amt

        remaining = nper - t
        if bal_after_prepay > 0.0:
            pmt = level_payment(bal_after_prepay, rate_m, remaining)
            interest_sch = bal_after_prepay * rate_m
            principal_sch = min(pmt - interest_sch, bal_after_prepay)
        else:
            interest_sch = 0.0
            principal_sch = 0.0

        bal_end = bal_after_prepay - principal_sch

        sched_prin[t] = principal_sch
        sched_int[t] = interest_sch
        prepay_prin[t] = prepay_amt
        default_prin[t] = default_amt
        surviving_fraction[t] = bal_end / balance if balance > 0 else 0.0

        recovery_amt = default_amt * (1.0 - sev_arr[t])
        landing = t + recovery_lag_months
        if landing < total_periods:
            recovery[landing] += recovery_amt

        bal = bal_end

    net_cashflow = sched_int + sched_prin + prepay_prin + recovery

    return pd.DataFrame(
        {
            "mob": np.arange(1, total_periods + 1),
            "scheduled_prin": sched_prin,
            "scheduled_int": sched_int,
            "prepay_prin": prepay_prin,
            "default_prin": default_prin,
            "recovery": recovery,
            "net_cashflow": net_cashflow,
            "surviving_fraction": surviving_fraction,
        }
    )
