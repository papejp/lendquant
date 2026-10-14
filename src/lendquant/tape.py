"""Canonical loan-tape schema + column aliasing + validation.

The canonical column names are the v0.1 contract. ``canonicalize_dataframe``
maps common vendor variants (LendingClub, Prosper, generic exports) to
these names via a regex alias table that callers can override.

The validation report is purely descriptive by default (returns a
dataclass of per-column findings). ``strict=True`` raises on the first
failed check so callers can fail-fast in pipelines.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

__all__ = [
    "ColumnFinding",
    "ValidationReport",
    "canonical_columns",
    "canonicalize_dataframe",
    "parse_vintage",
    "validate_tape",
]


_CANONICAL: tuple[str, ...] = (
    "loanid",
    "loanamount",
    "intrate",
    "term",
    "vintage_month",
    "origination_date",
    "fico_low",
    "fico_high",
    "dti",
    "annualinc",
    "grade",
    "purpose",
    "homeownership",
    "state",
)


_DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "loanid": (r"^loan_?id$", r"^id$"),
    "loanamount": (
        r"^loan_?amount$",
        r"^loan_?amt$",
        r"^loan_?amnt$",
        r"^amount$",
        r"^principal$",
        r"^original_?balance$",
    ),
    "intrate": (
        r"^int_?rate$",
        r"^interest_?rate$",
        r"^rate$",
        r"^apr$",
    ),
    "term": (r"^term$", r"^term_?months$"),
    "vintage_month": (r"^vintage(_?month)?$",),
    "origination_date": (
        r"^origination_?date$",
        r"^orig_?date$",
        r"^issue_?d$",
        r"^issued$",
    ),
    "fico_low": (r"^fico_?low$", r"^fico_?min$", r"^fico_?range_?low$"),
    "fico_high": (r"^fico_?high$", r"^fico_?max$", r"^fico_?range_?high$"),
    "dti": (r"^dti$", r"^debt_?to_?income$"),
    "annualinc": (r"^annual_?inc(ome)?$",),
    "grade": (r"^grade$", r"^credit_?grade$"),
    "purpose": (r"^purpose$", r"^loan_?purpose$"),
    "homeownership": (r"^home_?ownership$",),
    "state": (r"^state$", r"^addr_?state$"),
}


def canonical_columns() -> tuple[str, ...]:
    """Return the v0.1 allow-list of canonical tape column names."""
    return _CANONICAL


def canonicalize_dataframe(
    df: pd.DataFrame,
    aliases: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Lowercase + strip column names, then rename via a regex alias map.

    Default aliases cover LendingClub-style field names (``loan_amnt``,
    ``int_rate``, ``issue_d``, etc.). User-supplied ``aliases`` are
    merged on top (same key extends rather than replaces the defaults).

    Raises ``ValueError`` if two input columns would rename to the same
    canonical target.
    """
    merged: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in _DEFAULT_ALIASES.items()}
    if aliases is not None:
        for key, patterns in aliases.items():
            merged[key] = tuple(merged.get(key, ())) + tuple(patterns)

    normalized_columns = {col: str(col).strip().lower() for col in df.columns}
    out = df.rename(columns=normalized_columns).copy()

    rename_map: dict[str, str] = {}
    seen_targets: set[str] = set()
    for col in list(out.columns):
        if col in _CANONICAL:
            if col in seen_targets:
                raise ValueError(f"duplicate canonical column: {col!r}")
            seen_targets.add(col)
            continue
        for canonical, patterns in merged.items():
            if any(re.match(p, col) for p in patterns):
                if canonical in seen_targets:
                    raise ValueError(
                        f"two source columns map to canonical target {canonical!r}: "
                        f"{col!r} collides with an earlier match"
                    )
                rename_map[col] = canonical
                seen_targets.add(canonical)
                break

    out = out.rename(columns=rename_map)
    return out


def _looks_like_yyyymm(n: int) -> bool:
    if not (190001 <= n <= 999912):
        return False
    month = n % 100
    return 1 <= month <= 12


def parse_vintage(s: str | int | date | datetime) -> int:
    """Return YYYYMM integer from various input formats.

    Accepts:
        - ``datetime`` / ``date``: year * 100 + month.
        - ``int``: a value that parses as YYYYMM (year 1900..9999, month
          1..12), else interpreted as SAS numeric days-since-1960.
        - ``str``: bare YYYYMM, ISO ``YYYY-MM-DD``, ``YYYY/MM/DD``, or
          US ``MM/DD/YYYY``.
    """
    if isinstance(s, datetime):
        return s.year * 100 + s.month
    if isinstance(s, date):
        return s.year * 100 + s.month

    if isinstance(s, int):
        return _parse_int_vintage(s)

    if isinstance(s, str):
        stripped = s.strip()
        if stripped.lstrip("-").isdigit():
            return _parse_int_vintage(int(stripped))
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
            try:
                parsed = datetime.strptime(stripped, fmt)
                return parsed.year * 100 + parsed.month
            except ValueError:
                continue
        raise ValueError(f"could not parse vintage: {s!r}")

    raise TypeError(f"unsupported vintage type: {type(s).__name__}")


def _parse_int_vintage(n: int) -> int:
    if _looks_like_yyyymm(n):
        return n
    # Treat as SAS numeric: days since 1960-01-01.
    try:
        d = date(1960, 1, 1) + timedelta(days=n)
    except (OverflowError, ValueError) as e:
        raise ValueError(f"could not parse vintage from int: {n}") from e
    return d.year * 100 + d.month


@dataclass(frozen=True)
class ColumnFinding:
    """Per-column validation finding."""

    column: str
    required_present: bool
    type_ok: bool
    range_ok: bool
    null_rate: float
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """Validation report; iterate over ``findings`` or use ``by_column``."""

    findings: tuple[ColumnFinding, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(
            f.required_present and f.type_ok and f.range_ok for f in self.findings
        )

    def by_column(self) -> dict[str, ColumnFinding]:
        return {f.column: f for f in self.findings}


_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "loanamount": (0.0, float("inf")),
    "intrate": (0.0, 1.0),
    "term": (1.0, 600.0),
    "fico_low": (300.0, 850.0),
    "fico_high": (300.0, 850.0),
    "dti": (0.0, 5.0),
    "annualinc": (0.0, float("inf")),
}

_NUMERIC_COLUMNS: frozenset[str] = frozenset(_NUMERIC_RANGES.keys())
_STRING_COLUMNS: frozenset[str] = frozenset(
    {"grade", "purpose", "homeownership", "state"}
)


def validate_tape(df: pd.DataFrame, strict: bool = False) -> ValidationReport:
    """Validate a canonicalized tape against expected types and ranges.

    Checks per canonical column:
        - ``required_present``: column exists in ``df``.
        - ``type_ok``: numeric columns are numeric; string columns are
          object/string; date columns parse as datetime.
        - ``range_ok``: numeric values fall inside documented ranges;
          ``state`` is a 2-letter uppercase code; ``vintage_month`` is
          a YYYYMM integer.
        - ``null_rate``: fraction of NaN / NA entries.

    In ``strict`` mode, raises ``ValueError`` on the first failing check.
    """
    findings: list[ColumnFinding] = []
    for col in _CANONICAL:
        finding = _validate_column(df, col)
        findings.append(finding)
        if strict and not (
            finding.required_present and finding.type_ok and finding.range_ok
        ):
            raise ValueError(
                f"validation failed on {col!r}: {'; '.join(finding.messages)}"
            )
    return ValidationReport(findings=tuple(findings))


def _validate_column(df: pd.DataFrame, col: str) -> ColumnFinding:
    messages: list[str] = []
    if col not in df.columns:
        return ColumnFinding(
            column=col,
            required_present=False,
            type_ok=False,
            range_ok=False,
            null_rate=1.0,
            messages=(f"missing column {col!r}",),
        )

    series = df[col]
    null_rate = float(series.isna().mean()) if len(series) > 0 else 0.0
    non_null = series.dropna()

    type_ok = True
    range_ok = True

    if col in _NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(series):
            type_ok = False
            messages.append(f"expected numeric dtype, got {series.dtype}")
        else:
            lo, hi = _NUMERIC_RANGES[col]
            out_of_range = ((non_null < lo) | (non_null > hi)).sum()
            if out_of_range > 0:
                range_ok = False
                messages.append(
                    f"{int(out_of_range)} values outside [{lo}, {hi}]"
                )

    elif col in _STRING_COLUMNS:
        if not (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
        ):
            type_ok = False
            messages.append(f"expected string/object dtype, got {series.dtype}")
        if col == "state":
            bad = non_null[~non_null.astype(str).str.fullmatch(r"[A-Z]{2}")]
            if len(bad) > 0:
                range_ok = False
                messages.append(f"{len(bad)} values not 2-letter uppercase")

    elif col == "vintage_month":
        if not pd.api.types.is_integer_dtype(series):
            type_ok = False
            messages.append(f"expected integer YYYYMM, got {series.dtype}")
        else:
            bad = non_null[~non_null.astype(np.int64).map(_looks_like_yyyymm)]
            if len(bad) > 0:
                range_ok = False
                messages.append(f"{len(bad)} values not valid YYYYMM")

    elif col == "origination_date":
        if not pd.api.types.is_datetime64_any_dtype(series):
            type_ok = False
            messages.append(f"expected datetime dtype, got {series.dtype}")

    elif col == "loanid":
        if non_null.duplicated().any():
            range_ok = False
            messages.append("loanid values are not unique")

    return ColumnFinding(
        column=col,
        required_present=True,
        type_ok=type_ok,
        range_ok=range_ok,
        null_rate=null_rate,
        messages=tuple(messages),
    )
