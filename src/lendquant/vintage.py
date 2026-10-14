"""Static-pool / vintage descriptive curves.

Loan-count-weighted / value-weighted curves by vintage × months-on-book.
Everything here is **descriptive** — no fitting, no forecasting (that's
Pro). Inputs are long-format DataFrames.

Expected input schemas
----------------------
``loans``:
    - ``loanid``         (any, unique)
    - ``vintage``        (int YYYYMM): origination month
    - ``original_balance`` (float): origination balance
    - ``term``           (int): original term in months

``events`` (one row per loan event):
    - ``loanid``         (foreign key)
    - ``mob``            (int, 1-indexed month-on-book)
    - ``event_type``     (str): one of ``{"default", "prepay"}``
    - ``net_charge_off`` (float): dollar loss (default minus recovery);
      zero or NaN for prepay events

References
----------
- Fabozzi, *Handbook of Mortgage-Backed Securities* §13
  (static-pool cumulative loss convention).
- Kalbfleisch & Prentice, *Statistical Analysis of Failure Time Data*
  §1 (empirical conditional hazard = events / at-risk).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

__all__ = [
    "cdr_by_mob",
    "combine_cohorts",
    "cpr_by_mob",
    "static_pool_cnl",
]


def _loan_outcomes(loans: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long-format event table to one row per loan.

    Columns: ``loanid, vintage, term, mob_default, mob_prepay``. Missing
    outcomes get ``INT_MAX`` so simple ``>=`` comparisons produce the
    correct at-risk semantics.
    """
    sentinel = np.iinfo(np.int64).max
    defaults = (
        events[events["event_type"] == "default"]
        .groupby("loanid")["mob"]
        .min()
        .rename("mob_default")
    )
    prepays = (
        events[events["event_type"] == "prepay"]
        .groupby("loanid")["mob"]
        .min()
        .rename("mob_prepay")
    )
    out = loans[["loanid", "vintage", "term"]].copy()
    out = out.merge(defaults, on="loanid", how="left")
    out = out.merge(prepays, on="loanid", how="left")
    out["mob_default"] = out["mob_default"].fillna(sentinel).astype(np.int64)
    out["mob_prepay"] = out["mob_prepay"].fillna(sentinel).astype(np.int64)
    return out


def static_pool_cnl(loans: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Cumulative net loss by vintage × MOB (Fabozzi §13).

    ``CNL(v, t) = cumsum_{mob<=t}(net_charge_off_{v,mob}) /
    sum(original_balance_v)``. Denominator is the vintage origination
    total — the standard static-pool convention.

    Returns a DataFrame with columns ``vintage, mob, loans_originated,
    cnl_rate, denom``, one row per (vintage, mob) with mob from 1 to the
    max observed in that vintage.
    """
    vintage_summary = (
        loans.groupby("vintage")
        .agg(
            loans_originated=("loanid", "count"),
            denom=("original_balance", "sum"),
        )
        .reset_index()
    )

    merged = events.merge(loans[["loanid", "vintage"]], on="loanid", how="inner")
    co = (
        merged.groupby(["vintage", "mob"])["net_charge_off"]
        .sum()
        .reset_index()
        .rename(columns={"net_charge_off": "charge_off"})
    )
    co["charge_off"] = co["charge_off"].fillna(0.0)

    rows = []
    for _, vrow in vintage_summary.iterrows():
        vintage_id = int(vrow["vintage"])
        n_loans = int(vrow["loans_originated"])
        denom = float(vrow["denom"])
        co_v = co[co["vintage"] == vintage_id]
        if len(co_v) == 0:
            continue
        max_mob = int(co_v["mob"].max())
        by_mob = (
            co_v.set_index("mob")["charge_off"]
            .reindex(range(1, max_mob + 1), fill_value=0.0)
            .sort_index()
        )
        cumulative = by_mob.cumsum().to_numpy()
        cnl_rate = cumulative / denom if denom > 0 else np.full_like(cumulative, np.nan)
        for mob, rate in zip(range(1, max_mob + 1), cnl_rate, strict=True):
            rows.append(
                {
                    "vintage": vintage_id,
                    "mob": mob,
                    "loans_originated": n_loans,
                    "cnl_rate": float(rate),
                    "denom": denom,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["vintage", "mob", "loans_originated", "cnl_rate", "denom"]
        )
    return pd.DataFrame(rows)


def _event_rate_by_mob(
    loans: pd.DataFrame, events: pd.DataFrame, event_type: str, out_col: str
) -> pd.DataFrame:
    """Empirical annualized hazard for a given event type (K&P §1).

    MDR = defaults_at_mob / at_risk_at_mob; annualized via
    ``cdr = 1 - (1 - mdr)^12``. Same shape for prepays.
    """
    outcomes = _loan_outcomes(loans, events)
    rows = []
    for vintage_id, group in outcomes.groupby("vintage"):
        max_mob = int(group["term"].max())
        term = group["term"].to_numpy()
        mob_default = group["mob_default"].to_numpy()
        mob_prepay = group["mob_prepay"].to_numpy()
        n_originated = len(group)
        for t in range(1, max_mob + 1):
            still_perf = (mob_default >= t) & (mob_prepay >= t) & (term >= t)
            at_risk = int(still_perf.sum())
            if event_type == "default":
                evt_count = int((mob_default[still_perf] == t).sum())
            else:
                evt_count = int((mob_prepay[still_perf] == t).sum())
            if at_risk > 0:
                mdr = evt_count / at_risk
                annualized = 1.0 - (1.0 - mdr) ** 12
            else:
                annualized = np.nan
            rows.append(
                {
                    "vintage": int(vintage_id),
                    "mob": t,
                    "loans_originated": n_originated,
                    "at_risk": at_risk,
                    out_col: annualized,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["vintage", "mob", "loans_originated", "at_risk", out_col]
        )
    return pd.DataFrame(rows)


def cdr_by_mob(loans: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Empirical annualized conditional default rate per MOB (K&P §1).

    Columns: ``vintage, mob, loans_originated, at_risk, cdr``.
    """
    return _event_rate_by_mob(loans, events, event_type="default", out_col="cdr")


def cpr_by_mob(loans: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Empirical annualized conditional prepayment rate per MOB.

    Columns: ``vintage, mob, loans_originated, at_risk, cpr``.
    """
    return _event_rate_by_mob(loans, events, event_type="prepay", out_col="cpr")


def combine_cohorts(
    cohort_curves: Iterable[pd.DataFrame],
    weights: ArrayLike | None = None,
    metric_col: str | None = None,
    loan_count_col: str = "loans_originated",
) -> pd.DataFrame:
    """Blend cohort curves into a single pooled curve.

    Default weighting is by loan count (first value of ``loan_count_col``
    in each cohort). Pass ``weights`` explicitly for arbitrary blending.
    Cohorts of different lengths are outer-aligned on ``mob``; mobs
    where a cohort has no data are excluded from that mob's weighted
    average (not treated as zero).

    If ``metric_col`` is None, the first column that isn't ``mob``,
    ``vintage``, ``loan_count_col``, ``denom``, or ``at_risk`` is used.
    """
    curves = list(cohort_curves)
    if not curves:
        return pd.DataFrame({"mob": [], "metric_weighted": []})

    reserved = {"mob", "vintage", loan_count_col, "denom", "at_risk"}
    if metric_col is None:
        candidates = [c for c in curves[0].columns if c not in reserved]
        if not candidates:
            raise ValueError(
                "could not infer metric column; pass metric_col explicitly"
            )
        metric_col = candidates[0]

    if weights is None:
        ws = np.array(
            [float(curve[loan_count_col].iloc[0]) for curve in curves],
            dtype=np.float64,
        )
    else:
        ws = np.asarray(weights, dtype=np.float64)
        if ws.size != len(curves):
            raise ValueError("weights length must match number of cohort_curves")

    aligned = pd.concat(
        [
            curve[["mob", metric_col]]
            .rename(columns={metric_col: i})
            .set_index("mob")
            for i, curve in enumerate(curves)
        ],
        axis=1,
    ).sort_index()

    values = aligned.to_numpy(dtype=np.float64)
    mask = ~np.isnan(values)
    weight_matrix = np.where(mask, ws[np.newaxis, :], 0.0)
    numer = np.where(mask, values, 0.0) * weight_matrix
    total_weight = weight_matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(total_weight > 0, numer.sum(axis=1) / total_weight, np.nan)

    return pd.DataFrame(
        {
            "mob": aligned.index.to_numpy(),
            "metric_weighted": combined,
        }
    )
