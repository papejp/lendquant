# lendquant SPEC

**Status:** v0.1 spec draft. **This file drives the implementation**;
implementation is written from this spec + the cited public references,
not from any prior codebase (clean-room: see
`docs/LENDQUANT_MIGRATION_PLAN.md` §3 in the sibling repo).

**Validation rule:** every function in v0.1 is validated against
textbook worked examples, hand calculations, and (where applicable)
numpy-financial / Excel cross-checks. No parity tests against SAS
artifacts or any prior proprietary code.

---

## Public references

- **SIFMA Standard Formulas for the Analysis of Mortgage-Backed Securities and Other Related Securities** (CPR, SMM, PSA, CDR conventions).
- **Fabozzi, F.J.** *The Handbook of Mortgage-Backed Securities* (static-pool analysis; prepayment + default modeling conventions; weighted average life).
- **Brealey, R., Myers, S., Allen, F.** *Principles of Corporate Finance* (IRR, NPV, price ↔ yield, duration).
- **Hull, J.** *Options, Futures, and Other Derivatives* (modified duration + rate conventions).
- **Kalbfleisch, J.D., Prentice, R.L.** *The Statistical Analysis of Failure Time Data* (survival + hazard conventions for static-pool curves).

Specific equations in each module below cite the reference inline.

---

## Modules

### `lendquant.rates`

Rate conversions. All pure functions, scalar + array via numpy.

| Function | Signature | Spec |
|---|---|---|
| `smm_from_cpr` | `(cpr: float\|ndarray) -> float\|ndarray` | `smm = 1 - (1 - cpr) ** (1/12)` (SIFMA Standard Formulas) |
| `cpr_from_smm` | `(smm: float\|ndarray) -> float\|ndarray` | `cpr = 1 - (1 - smm) ** 12` |
| `mdr_from_cdr` | `(cdr: float\|ndarray) -> float\|ndarray` | `mdr = 1 - (1 - cdr) ** (1/12)` (SIFMA, same shape as SMM/CPR) |
| `cdr_from_mdr` | `(mdr: float\|ndarray) -> float\|ndarray` | `cdr = 1 - (1 - mdr) ** 12` |
| `monthly_from_apr` | `(apr: float\|ndarray) -> float\|ndarray` | `r_m = apr / 12` (nominal convention; what APR means on a US-market monthly-pay loan — Brealey §3) |
| `apr_from_monthly` | `(r_m: float\|ndarray) -> float\|ndarray` | `apr = 12 * r_m` |
| `monthly_from_annualized` | `(ann: float\|ndarray) -> float\|ndarray` | `r_m = (1 + ann) ** (1/12) - 1` (effective-rate convention — Hull §4.2) |
| `annualize_monthly` | `(r_m: float\|ndarray) -> float\|ndarray` | `ann = (1 + r_m) ** 12 - 1` |

Validation: round-trip identity to float64 precision; worked example from Fabozzi tables (CPR 6% → SMM 0.5143%).

---

### `lendquant.cashflows`

Level-pay amortization + pool-level cash flows from user-supplied
rate / default / prepay / severity / recovery-lag vectors.

| Function | Signature | Spec |
|---|---|---|
| `level_payment` | `(principal, rate_m, nper) -> float` | Standard amortization payment: `P * r_m / (1 - (1 + r_m) ** -n)`. Edge case `r_m == 0` returns `P / n`. |
| `amortization_schedule` | `(principal, rate_m, nper) -> DataFrame[mob, pay, interest, principal, balance]` | Monthly schedule for a performing level-pay loan. |
| `pool_cashflows` | `(balance, rate_m, nper, cdr: ndarray, cpr: ndarray, severity: ndarray \| float, recovery_lag_months: int = 0) -> DataFrame[mob, scheduled_prin, scheduled_int, prepay_prin, default_prin, recovery, net_cashflow, surviving_fraction]` | Vector-driven pool cashflow. Each MOB applies CDR → default, CPR → full-prepayment, then amortizes the survivors. Losses = severity × defaulted-balance. Recoveries land `recovery_lag_months` later. See Fabozzi §11–12 for the account-fraction convention. |

Validation: hand-calc a 3-period 10% / 0% / 50% severity example; cross-check level-pay schedule against `numpy_financial.pmt`.

---

### `lendquant.metrics`

Yield & valuation metrics on cash flow arrays.

| Function | Signature | Spec |
|---|---|---|
| `irr` | `(cashflows: ndarray) -> float` | Scalar IRR on uniformly-spaced monthly cashflows via Brent's method; annualized via `annualize_monthly`. Returns `nan` if no sign change. (Brealey §5) |
| `xirr` | `(cashflows: ndarray, dates: ndarray[datetime64]) -> float` | Date-aware IRR for irregular spacing; same solver, year-fraction `(d - d0) / 365.25`. (Excel XIRR convention) |
| `npv` | `(cashflows, rate_m) -> float` | `sum(cf / (1 + r_m)**t for t, cf in enumerate(cashflows))`. |
| `price_from_yield` | `(cashflows, ytm_annualized) -> float` | Discount cashflows at monthly rate from `monthly_from_annualized(ytm)`. |
| `yield_from_price` | `(cashflows, price) -> float` | Brent-solve on `price_from_yield` inverse. |
| `wal` | `(principal_cashflows: ndarray) -> float` | Weighted average life in years: `sum(t * prin_t) / sum(prin_t) / 12`. (Fabozzi §9) |
| `macaulay_duration` | `(cashflows, ytm_annualized) -> float` | `sum(t * cf_t * (1 + r_m)**-t) / sum(cf_t * (1 + r_m)**-t) / 12` in years. (Hull §4.11) |
| `modified_duration` | `(cashflows, ytm_annualized) -> float` | `macaulay / (1 + ytm_annualized)`. |
| `loss_adjusted_yield` | `(cashflows, ytm_annualized, loss_cashflows) -> float` | `yield_from_price(cashflows - loss_cashflows, price)` — expresses yield net of modeled losses. |

Validation: Excel IRR / XIRR / PRICE / DURATION cross-checks on a 10-cashflow test vector.

---

### `lendquant.vintage`

Static-pool / cumulative-net-loss curves by vintage × months-on-book.
**Descriptive only** — no forecasting, no model fitting (that's Pro).

| Function | Signature | Spec |
|---|---|---|
| `static_pool_cnl` | `(loans: DataFrame, events: DataFrame) -> DataFrame[vintage, mob, loans_originated, cnl_rate, denom]` | Cumulative net loss as `cumsum(net_charge_off_amount_at_mob) / sum(original_balance_at_vintage)`. Denominator is origination; standard static-pool convention. (Fabozzi §13) |
| `cdr_by_mob` | `(loans, events) -> DataFrame[vintage, mob, cdr]` | Empirical conditional default rate per Kalbfleisch & Prentice: `defaults_at_mob / at_risk_at_mob`, annualized. |
| `cpr_by_mob` | `(loans, events) -> DataFrame[vintage, mob, cpr]` | Same shape for full-prepayments. |
| `combine_cohorts` | `(cohort_curves: Iterable[DataFrame], weights: ndarray \| None = None) -> DataFrame[mob, metric_weighted]` | Loan-count-weighted (default) or arbitrary-weight blend across cohorts. |

Validation: synthetic pool where defaults are hand-counted → match.

---

### `lendquant.tape`

Canonical loan-tape schema + column alias + validation.

| Function | Signature | Spec |
|---|---|---|
| `canonical_columns()` | `-> tuple[str, ...]` | Allow-list: `loanid, loanamount, intrate, term, vintage_month, origination_date, fico_low, fico_high, dti, annualinc, grade, purpose, homeownership, state`. Payment-event columns separate. |
| `canonicalize_dataframe` | `(df: DataFrame, aliases: dict[str, Sequence[str]] \| None = None) -> DataFrame` | Lowercase + strip + regex alias map (user-supplied overrides). Raises on duplicate target. |
| `parse_vintage` | `(s: str \| int \| datetime) -> int` | Returns YYYYMM integer. Accepts ISO, US `MM/DD/YYYY`, SAS numeric days-since-1960, bare YYYYMM. |
| `validate_tape` | `(df: DataFrame, strict: bool = False) -> ValidationReport` | Dataclass with per-column findings: `required_present`, `type_ok`, `range_ok`, `null_rate`. Strict mode raises on first failure. |

Validation: round-trip canonicalize an input with every alias variant; validation report matches hand-check on a 10-row seed.

---

## Out of scope for v0.1 (explicitly)

- Scoring (JPSCORE9 or any portfolio-model-driven score). Pro.
- Calibration (fitting score → PD curves, vintage recal matrices). Pro.
- Feature engineering (derived credit attributes). Pro.
- Segment selection / slice ranking / recommended price. Pro.
- Forecasting PDs or prepays. Pro.
- SAS data readers. Will never be in `lendquant`.

---

## Release checklist

- [ ] All v0.1 modules + tests + docstrings implemented
- [ ] Validation suite passes on textbook examples
- [ ] mypy --strict clean
- [ ] ruff clean
- [ ] `pip install -e .` works on a clean venv
- [ ] 4 recipe docs published at lendquant.com/docs
- [ ] PyPI 0.1.0 upload
