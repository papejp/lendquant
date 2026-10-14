# lendquant

Open-source loan-pool math.

- Rate conversions (CPR↔SMM, CDR↔MDR, APR↔monthly)
- Amortization + pool cash flows from user-supplied CDR / CPR / severity / recovery-lag vectors
- IRR, XIRR, NPV, price ↔ yield, WAL, Macaulay & modified duration
- Static-pool / cumulative-net-loss curves by vintage × MOB
- Canonical loan-tape schema + validation

Pre-alpha. See `SPEC.md` for the module-by-module API.

## Install

```bash
pip install lendquant
```

## Quickstart: price a loan pool end-to-end

A 10-year, \$250k pool at 7% APR with ramping defaults, steady prepays,
and lagged recoveries. Compute net cash flows, IRR, WAL, and duration.

```python
import numpy as np

from lendquant.cashflows import pool_cashflows
from lendquant.metrics import irr, macaulay_duration, modified_duration, wal
from lendquant.rates import monthly_from_apr

balance = 250_000.0
rate_m = monthly_from_apr(0.07)       # nominal APR → monthly rate
nper = 120

pool = pool_cashflows(
    balance=balance,
    rate_m=rate_m,
    nper=nper,
    cdr=np.linspace(0.00, 0.04, nper),  # defaults ramp 0 → 4%/yr
    cpr=0.08,                            # steady 8%/yr prepay
    severity=0.60,                       # 60% loss given default
    recovery_lag_months=6,
)

# Investor view: lend `balance` today, receive `net_cashflow` monthly.
cashflows = np.concatenate([[-balance], pool["net_cashflow"].to_numpy()])
annual_irr = irr(cashflows)

# WAL on the pool's principal cashflows (scheduled + prepaid).
# Prepend a zero so index 0 is "today" — matches the 0-indexed time
# convention used across the metrics module.
principal = (pool["scheduled_prin"] + pool["prepay_prin"]).to_numpy()
pool_wal = wal(np.concatenate([[0.0], principal]))

# Duration at YTM = IRR, computed on the receipt side only (so the
# denominator = price, not NPV = 0 at IRR).
receipts = np.concatenate([[0.0], pool["net_cashflow"].to_numpy()])
mac = macaulay_duration(receipts, annual_irr)
mod = modified_duration(receipts, annual_irr)

print(f"IRR:            {annual_irr:.4%}")   # 6.4687%
print(f"WAL:            {pool_wal:.2f} y")   # 4.13 y
print(f"Macaulay:       {mac:.2f} y")        # 3.46 y
print(f"Mod. duration:  {mod:.2f} y")        # 3.25 y
```

The pool DataFrame carries the full waterfall per MOB —
`scheduled_prin`, `scheduled_int`, `prepay_prin`, `default_prin`,
`recovery`, `net_cashflow`, `surviving_fraction` — so you can compose
loss-adjusted series, bucket by period, or hand them to any other
analytics layer.

## More recipes

- [**Rate conventions**](docs/rate-conventions.md) — CPR/SMM,
  CDR/MDR, and the nominal-vs-effective gotcha for interest rates.
- [**Measuring vintage loss curves**](docs/vintage-curves.md) —
  static-pool CNL, empirical CDR / CPR by MOB, and cohort blending
  from a long-format loans + events table.
- [**Canonicalizing and validating a loan tape**](docs/loan-tape.md)
  — regex alias map, multi-format vintage parsing, and the
  per-column validation report.

## Scope

`lendquant` does the math — scenario-driven, deterministic, zero model
opinions. If you want forecasting, calibration, scoring, segment
selection, or price recommendations, use [LendQuant Pro](https://lendquant.com).

> *Free does the math; Pro does the judgment.*

## License

Apache-2.0.
