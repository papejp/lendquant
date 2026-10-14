# Rate conventions

Credit math mixes two rate families that look nearly identical on
paper but produce different numbers:

- **Hazard rates** (CPR/SMM, CDR/MDR) use the SIFMA convention:
  `monthly = 1 - (1 - annual) ** (1/12)`. These are survival-style
  rates — the fraction of a surviving pool that experiences an event
  in a given period.
- **Interest rates** come in two flavors:
  - **Nominal APR** (Brealey §3): `r_m = APR / 12`. The US-market
    convention for monthly-pay loans. A 12% APR means 1%/month, flat.
  - **Effective annualized** (Hull §4.2):
    `r_m = (1 + ann) ** (1/12) - 1`. Compounds. A 12% effective
    annual rate means `1.009489…%/month`.

Mixing these up quietly undercounts (or overcounts) interest.
`lendquant.rates` gives you the eight conversion functions and
nothing else — pure numeric utilities, scalar- and array-polymorphic.

## SIFMA hazards: CPR ↔ SMM, CDR ↔ MDR

```python
from lendquant.rates import (
    cdr_from_mdr,
    cpr_from_smm,
    mdr_from_cdr,
    smm_from_cpr,
)

print(f"6% CPR → SMM: {smm_from_cpr(0.06):.6f}")   # 0.005143
```

That `0.005143` is the SIFMA worked example from Fabozzi — a 6%
annual prepayment rate translates to roughly 51 bps / month of
survivors prepaying. CDR ↔ MDR takes exactly the same shape; the
distinction between the two pairs is semantic (prepayments vs.
defaults), not mathematical.

## Nominal vs. effective: the gotcha

```python
from lendquant.rates import annualize_monthly, monthly_from_annualized, monthly_from_apr

apr_m = monthly_from_apr(0.12)         # 0.010000
eff_m = monthly_from_annualized(0.12)  # 0.009489
```

Both inputs are called "12%" by different authors. The nominal APR
convention gives a monthly rate exactly 5.1 bps higher than the
effective one. Compounded over 12 months, the APR implies an
effective annual rate of `(1.01)^12 - 1 = 12.6825%` — the gap
between "12% APR" and "12% effective" is real money.

```python
print(f"Effective annual implied by 12% APR: "
      f"{annualize_monthly(apr_m):.6f}")      # 0.126825
```

Pick one convention per analysis and stay with it. For US consumer
lending work, `monthly_from_apr` is almost always what you want
(it's what shows up on the loan contract). For bond-math-style
pricing, `monthly_from_annualized` matches the Hull / textbook
duration conventions and plays nicely with
[`lendquant.metrics.price_from_yield`](../README.md).

## Array inputs are first-class

Every conversion accepts a numpy array and returns one of matching
shape — no reshaping, no loops.

```python
import numpy as np
from lendquant.rates import smm_from_cpr

cprs = np.array([0.02, 0.06, 0.10, 0.20])
print(smm_from_cpr(cprs).round(5))
# [0.00168 0.00514 0.00874 0.01842]
```

This matters because the pool model
([`lendquant.cashflows.pool_cashflows`](../README.md)) takes
per-period CDR and CPR vectors directly. You can build a ramping
default curve, broadcast an industry-standard PSA/CPR profile, or
hand in a hand-tuned stress scenario — the rate module just does
the math.

## Round-trip identities

All eight functions compose to the identity to float64 precision:

```python
from lendquant.rates import (
    annualize_monthly,
    cpr_from_smm,
    monthly_from_annualized,
    smm_from_cpr,
)

# One ULP of noise is normal; tolerance should match float64 epsilon.
assert abs(cpr_from_smm(smm_from_cpr(0.06)) - 0.06) < 1e-14
assert abs(annualize_monthly(monthly_from_annualized(0.09)) - 0.09) < 1e-14
```

Treat equality with a tolerance (`1e-12` is comfortable). Exact
equality works only for a handful of inputs that happen to land on
exact float representations.

## Function reference

| Function                   | Formula                                 | Convention       |
|----------------------------|-----------------------------------------|------------------|
| `smm_from_cpr(cpr)`        | `1 - (1 - cpr) ** (1/12)`               | SIFMA            |
| `cpr_from_smm(smm)`        | `1 - (1 - smm) ** 12`                   | SIFMA            |
| `mdr_from_cdr(cdr)`        | `1 - (1 - cdr) ** (1/12)`               | SIFMA            |
| `cdr_from_mdr(mdr)`        | `1 - (1 - mdr) ** 12`                   | SIFMA            |
| `monthly_from_apr(apr)`    | `apr / 12`                              | Nominal          |
| `apr_from_monthly(r_m)`    | `12 * r_m`                              | Nominal          |
| `monthly_from_annualized(ann)` | `(1 + ann) ** (1/12) - 1`           | Effective        |
| `annualize_monthly(r_m)`   | `(1 + r_m) ** 12 - 1`                   | Effective        |
