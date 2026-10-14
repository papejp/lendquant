"""Tests for lendquant.tape.

Validation strategy (per SPEC.md):
- Round-trip canonicalize an input with every alias variant.
- parse_vintage round-trip for every accepted input type.
- validate_tape hand-check on a seed DataFrame: known type / range
  / null-rate findings.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from lendquant.tape import (
    ColumnFinding,
    canonical_columns,
    canonicalize_dataframe,
    parse_vintage,
    validate_tape,
)


class TestCanonicalColumns:
    def test_contains_v01_fields(self) -> None:
        cols = canonical_columns()
        for required in [
            "loanid", "loanamount", "intrate", "term",
            "vintage_month", "origination_date",
            "fico_low", "fico_high", "dti", "annualinc",
            "grade", "purpose", "homeownership", "state",
        ]:
            assert required in cols

    def test_tuple_type(self) -> None:
        assert isinstance(canonical_columns(), tuple)


class TestCanonicalizeDataframe:
    def test_lendingclub_style_renames(self) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "loan_amnt": [1000, 2000],
                "int_rate": [0.10, 0.12],
                "term": [36, 60],
                "issue_d": ["2024-01", "2024-02"],
                "fico_range_low": [700, 720],
                "fico_range_high": [720, 740],
                "annual_inc": [50_000, 75_000],
                "home_ownership": ["RENT", "OWN"],
                "addr_state": ["CA", "NY"],
            }
        )
        out = canonicalize_dataframe(df)
        for col in [
            "loanid", "loanamount", "intrate", "term", "origination_date",
            "fico_low", "fico_high", "annualinc", "homeownership", "state",
        ]:
            assert col in out.columns

    def test_whitespace_and_case(self) -> None:
        df = pd.DataFrame({"  Loan_ID ": [1], "  Int_Rate  ": [0.1]})
        out = canonicalize_dataframe(df)
        assert "loanid" in out.columns
        assert "intrate" in out.columns

    def test_user_aliases_extend_defaults(self) -> None:
        df = pd.DataFrame({"custom_rate_field": [0.1]})
        out = canonicalize_dataframe(
            df, aliases={"intrate": [r"^custom_rate_field$"]}
        )
        assert "intrate" in out.columns

    def test_already_canonical_pass_through(self) -> None:
        df = pd.DataFrame({"loanid": [1], "loanamount": [100.0]})
        out = canonicalize_dataframe(df)
        assert set(out.columns) == {"loanid", "loanamount"}

    def test_duplicate_target_raises(self) -> None:
        # Two different source columns both match "loanid" aliases.
        df = pd.DataFrame({"loan_id": [1], "id": [2]})
        with pytest.raises(ValueError, match="loanid"):
            canonicalize_dataframe(df)

    def test_unknown_columns_preserved(self) -> None:
        df = pd.DataFrame({"loan_amnt": [100.0], "mystery_field": ["x"]})
        out = canonicalize_dataframe(df)
        assert "loanamount" in out.columns
        assert "mystery_field" in out.columns


class TestParseVintage:
    def test_datetime(self) -> None:
        assert parse_vintage(datetime(2024, 3, 15)) == 202403

    def test_date(self) -> None:
        assert parse_vintage(date(2024, 3, 15)) == 202403

    def test_yyyymm_int(self) -> None:
        assert parse_vintage(202403) == 202403

    def test_yyyymm_str(self) -> None:
        assert parse_vintage("202403") == 202403

    def test_iso_date_str(self) -> None:
        assert parse_vintage("2024-03-15") == 202403

    def test_us_date_str(self) -> None:
        assert parse_vintage("03/15/2024") == 202403

    def test_slash_iso_str(self) -> None:
        assert parse_vintage("2024/03/15") == 202403

    def test_sas_numeric(self) -> None:
        # SAS numeric: days since 1960-01-01. 2024-01-01 = 23376 days.
        # Confirm via pandas to avoid off-by-one speculation.
        n = (pd.Timestamp("2024-03-15") - pd.Timestamp("1960-01-01")).days
        assert parse_vintage(n) == 202403

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_vintage("not a date")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            parse_vintage(3.14)  # type: ignore[arg-type]

    def test_strip_whitespace(self) -> None:
        assert parse_vintage("  2024-03-15  ") == 202403


class TestValidateTape:
    @staticmethod
    def _seed_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "loanid": ["L1", "L2", "L3"],
                "loanamount": [10_000.0, 20_000.0, 30_000.0],
                "intrate": [0.08, 0.10, 0.15],
                "term": [36, 60, 36],
                "vintage_month": [202401, 202402, 202402],
                "origination_date": pd.to_datetime(["2024-01-15", "2024-02-10", "2024-02-20"]),
                "fico_low": [700, 720, 740],
                "fico_high": [720, 740, 760],
                "dti": [0.25, 0.30, 0.22],
                "annualinc": [50_000.0, 75_000.0, 100_000.0],
                "grade": ["A", "B", "A"],
                "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
                "homeownership": ["RENT", "OWN", "MORTGAGE"],
                "state": ["CA", "NY", "TX"],
            }
        )

    def test_clean_report_is_ok(self) -> None:
        report = validate_tape(self._seed_df())
        assert report.ok, {f.column: f.messages for f in report.findings if not (
            f.required_present and f.type_ok and f.range_ok
        )}

    def test_missing_column(self) -> None:
        df = self._seed_df().drop(columns=["intrate"])
        report = validate_tape(df)
        finding = report.by_column()["intrate"]
        assert not finding.required_present
        assert not report.ok

    def test_out_of_range_intrate(self) -> None:
        df = self._seed_df()
        df.loc[0, "intrate"] = 1.5  # > 1 → percentage, not fraction
        report = validate_tape(df)
        finding = report.by_column()["intrate"]
        assert finding.required_present and finding.type_ok
        assert not finding.range_ok

    def test_bad_fico(self) -> None:
        df = self._seed_df()
        df.loc[1, "fico_low"] = 100
        report = validate_tape(df)
        assert not report.by_column()["fico_low"].range_ok

    def test_bad_state_code(self) -> None:
        df = self._seed_df()
        df.loc[0, "state"] = "california"
        report = validate_tape(df)
        assert not report.by_column()["state"].range_ok

    def test_bad_vintage_month(self) -> None:
        df = self._seed_df()
        df.loc[0, "vintage_month"] = 202413  # month 13
        report = validate_tape(df)
        assert not report.by_column()["vintage_month"].range_ok

    def test_duplicate_loanid(self) -> None:
        df = self._seed_df()
        df.loc[2, "loanid"] = "L1"
        report = validate_tape(df)
        assert not report.by_column()["loanid"].range_ok

    def test_null_rate_recorded(self) -> None:
        df = self._seed_df()
        df.loc[0, "dti"] = np.nan
        report = validate_tape(df)
        finding = report.by_column()["dti"]
        assert finding.null_rate == pytest.approx(1.0 / 3)

    def test_strict_raises_on_first_failure(self) -> None:
        df = self._seed_df().drop(columns=["loanid"])
        with pytest.raises(ValueError, match="loanid"):
            validate_tape(df, strict=True)

    def test_clean_report_findings_cover_all_canonicals(self) -> None:
        report = validate_tape(self._seed_df())
        reported = {f.column for f in report.findings}
        assert reported == set(canonical_columns())

    def test_column_finding_frozen(self) -> None:
        finding = ColumnFinding(
            column="x",
            required_present=True,
            type_ok=True,
            range_ok=True,
            null_rate=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            finding.column = "y"  # type: ignore[misc]
