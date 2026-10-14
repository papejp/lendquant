# Canonicalizing and validating a loan tape

`lendquant.tape` gives you three building blocks for taming a vendor
loan tape before the math layer touches it:

- **`canonicalize_dataframe`** — rename columns to the v0.1 canonical
  schema via a regex alias map that covers LendingClub-style names
  out of the box.
- **`parse_vintage`** — convert any reasonable date/int representation
  to a YYYYMM integer.
- **`validate_tape`** — report per-column type / range / null-rate
  findings; raise on first failure in `strict` mode.

## Start with a vendor DataFrame

Here's a slice of a LendingClub-flavored export. Note the typical
pain points: capitalized column names, LC's `loan_amnt` /
`int_rate` / `issue_d` conventions, and `addr_state` instead of a
plain `state`.

```python
from datetime import date

import pandas as pd
from lendquant.tape import (
    canonical_columns,
    canonicalize_dataframe,
    parse_vintage,
    validate_tape,
)

raw = pd.DataFrame({
    "ID": [1, 2, 3],
    "loan_amnt": [10_000, 20_000, 30_000],
    "int_rate": [0.105, 0.120, 0.085],      # fraction, not percent
    "term": [36, 60, 36],
    "issue_d": pd.to_datetime(["2024-01-15", "2024-02-10", "2024-03-20"]),
    "fico_range_low": [700, 720, 740],
    "fico_range_high": [720, 740, 760],
    "dti": [0.25, 0.30, 0.22],
    "annual_inc": [50_000, 75_000, 100_000],
    "grade": ["A", "B", "A"],
    "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
    "home_ownership": ["RENT", "OWN", "MORTGAGE"],
    "addr_state": ["CA", "NY", "TX"],
})
```

## Rename to the canonical schema

```python
tape = canonicalize_dataframe(raw)
print(tape.columns.tolist())
# ['loanid', 'loanamount', 'intrate', 'term', 'origination_date',
#  'fico_low', 'fico_high', 'dti', 'annualinc', 'grade', 'purpose',
#  'homeownership', 'state']
```

`canonicalize_dataframe` lowercases + strips column names first, then
applies the regex alias table. The full canonical allow-list is
exposed via `canonical_columns()`.

If two source columns resolve to the same canonical target (e.g.
both `loan_id` and `id`), the call raises. Fix the input before
retrying — silent collisions would be worse.

### Custom aliases

User-supplied aliases **extend** the defaults for a given canonical
target rather than replacing them:

```python
custom_tape = canonicalize_dataframe(
    raw.rename(columns={"int_rate": "coupon"}),
    aliases={"intrate": [r"^coupon$"]},
)
assert "intrate" in custom_tape.columns
```

## Derive `vintage_month` from the origination date

The canonical schema includes both `origination_date` (a datetime)
and `vintage_month` (an integer YYYYMM). `parse_vintage` accepts
every reasonable input format:

```python
for v in [202403, "2024-03-15", "03/15/2024", date(2024, 3, 15)]:
    print(f"{v!r:30} → {parse_vintage(v)}")
# 202403                         → 202403
# '2024-03-15'                   → 202403
# '03/15/2024'                   → 202403
# datetime.date(2024, 3, 15)     → 202403

tape["vintage_month"] = tape["origination_date"].apply(parse_vintage)
```

For integer inputs, `parse_vintage` first tests whether the value
looks like a valid YYYYMM (year 1900–9999, month 1–12). Anything
outside that window is interpreted as a SAS numeric
(days-since-1960) and converted via `date(1960, 1, 1) + days`.

## Validate

`validate_tape` returns a frozen dataclass of per-column findings.
Each `ColumnFinding` records `required_present`, `type_ok`,
`range_ok`, `null_rate`, and a tuple of human-readable messages.

```python
report = validate_tape(tape)
print(f"OK: {report.ok}")   # True

# Introduce a bug: rate given as a percentage, not a fraction.
dirty = tape.copy()
dirty.loc[0, "intrate"] = 10.5
report = validate_tape(dirty)
print(f"OK: {report.ok}")   # False
print(report.by_column()["intrate"].messages)
# ('1 values outside [0.0, 1.0]',)
```

Numeric ranges enforced today:

| Column       | Range          |
|--------------|----------------|
| `loanamount` | > 0            |
| `intrate`    | [0.0, 1.0]     |
| `term`       | [1, 600] mo    |
| `fico_low`   | [300, 850]     |
| `fico_high`  | [300, 850]     |
| `dti`        | [0.0, 5.0]     |
| `annualinc`  | ≥ 0            |

Plus: `state` must be a 2-letter uppercase code; `vintage_month`
must be a valid YYYYMM integer; `loanid` must be unique.

### Strict mode

Pass `strict=True` to raise `ValueError` on the first failing check
— useful at the top of a pipeline where downstream math assumes
clean inputs:

```python
validate_tape(dirty, strict=True)
# ValueError: validation failed on 'intrate': 1 values outside [0.0, 1.0]
```

## What this isn't

`lendquant.tape` does not transform *values* — it won't parse
`"36 months"` into `36`, strip `"%"` from a rate string, or coerce
dates from mixed string formats inside a column. Clean the values
upstream (`pd.read_csv(..., dtype=..., parse_dates=...)` or a one-off
cleaning function). This library's job is to make sure what you
hand it is schema-correct, not to guess what you meant.
