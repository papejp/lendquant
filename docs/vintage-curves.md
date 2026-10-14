# Measuring vintage loss curves

Build cumulative-net-loss, CDR, and CPR curves from a long-format
loans + events table, then blend cohorts into a single pool curve.

## Input shape

`lendquant.vintage` expects two DataFrames:

- **`loans`** — one row per loan. Required columns: `loanid`,
  `vintage` (YYYYMM int), `original_balance` (float), `term` (months).
- **`events`** — one row per event. Required columns: `loanid` (FK),
  `mob` (int, 1-indexed), `event_type` (one of `"default"`,
  `"prepay"`), `net_charge_off` (float — dollar loss net of
  recoveries; zero or NaN for prepay rows).

Loans that never default or prepay simply have no events rows.

## Build three vintages

```python
import numpy as np
import pandas as pd
from lendquant.vintage import (
    cdr_by_mob,
    combine_cohorts,
    cpr_by_mob,
    static_pool_cnl,
)

rng = np.random.default_rng(42)


def make_cohort(vintage: int, n: int, mdr: float, smm: float,
                term: int = 36, balance: float = 10_000.0):
    """Toy cohort: event times sampled from independent geometric hazards."""
    loans = pd.DataFrame({
        "loanid": [f"V{vintage}L{i}" for i in range(n)],
        "vintage": vintage,
        "original_balance": balance,
        "term": term,
    })
    rows = []
    for lid in loans["loanid"]:
        t_default = int(rng.geometric(p=mdr))
        t_prepay = int(rng.geometric(p=smm))
        if t_default < t_prepay and t_default <= term:
            rows.append({"loanid": lid, "mob": t_default,
                         "event_type": "default",
                         "net_charge_off": balance * 0.6})
        elif t_prepay <= term:
            rows.append({"loanid": lid, "mob": t_prepay,
                         "event_type": "prepay",
                         "net_charge_off": 0.0})
    return loans, pd.DataFrame(rows)


cohorts = [
    make_cohort(202301, 500, mdr=0.004, smm=0.012),
    make_cohort(202307, 500, mdr=0.006, smm=0.010),
    make_cohort(202401, 500, mdr=0.003, smm=0.015),
]
loans = pd.concat([c[0] for c in cohorts], ignore_index=True)
events = pd.concat([c[1] for c in cohorts], ignore_index=True)
```

## Static-pool CNL

Cumulative net loss by (vintage, MOB), with the denominator fixed at
origination (Fabozzi §13).

```python
cnl = static_pool_cnl(loans, events)
print(cnl.groupby("vintage")["cnl_rate"].max().round(4))
# vintage
# 202301    0.0576
# 202307    0.0756
# 202401    0.0576
```

The `202307` vintage has the highest MDR in this synthetic setup, so
its terminal CNL is highest — as expected.

## CDR and CPR by MOB

Empirical annualized hazards (Kalbfleisch & Prentice):
`events_at_mob / at_risk_at_mob`, annualized via `(1 - rate)^12`.

```python
cdr = cdr_by_mob(loans, events)
cpr = cpr_by_mob(loans, events)

print(cdr[cdr["mob"] == 12].set_index("vintage")[["at_risk", "cdr"]].round(4))
#          at_risk     cdr
# vintage
# 202301       414  0.1357
# 202307       429  0.0808
# 202401       407  0.0574
```

`at_risk` is the number of loans still performing at the start of
that MOB (not defaulted, not prepaid, within term). Noisy tail
periods have small `at_risk` and should be interpreted with care.

## Blend across cohorts

`combine_cohorts` outer-aligns the per-vintage curves on MOB and
takes a weighted average. By default it uses the loan count of each
cohort as the weight.

```python
per_vintage = [cdr[cdr["vintage"] == v].copy()
               for v in sorted(cdr["vintage"].unique())]
pool_cdr = combine_cohorts(per_vintage, metric_col="cdr")
print(pool_cdr.head(6).round(4).to_string(index=False))
#  mob  metric_weighted
#    1           0.0236
#    2           0.0241
#    3           0.0560
#    4           0.0480
#    5           0.0573
#    6           0.0170
```

- NaN cells (cohorts with no data at a given MOB) are excluded from
  the weighted average rather than treated as zero.
- Pass `weights=np.array([...])` to override the default
  loan-count weighting.
- `metric_col` is auto-detected when there's a single non-identifier
  column; pass it explicitly for CNL curves (`metric_col="cnl_rate"`)
  since those tables have both `cnl_rate` and `denom`.

## Scope

Everything in `lendquant.vintage` is **descriptive**: it reports what
the data says. Fitting a curve, projecting a vintage forward, or
recalibrating hazards onto a new score are Pro features — this layer
intentionally stays model-free.
