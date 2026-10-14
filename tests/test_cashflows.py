"""Tests for lendquant.cashflows.

Validation strategy (per SPEC.md):
- level_payment cross-checked against ``numpy_financial.pmt``.
- amortization_schedule internal consistency (interest + principal = pay,
  final balance = 0, sum of principal = original).
- pool_cashflows:
  * cdr=cpr=0 collapses to the clean amortization schedule.
  * cpr>0 shrinks the pool via prepay without losses.
  * cdr>0 with severity extremes isolates default vs. recovery accounting.
  * Hand-calc the 3-period / 10% CDR / 0% CPR / 50% severity example.
  * ``recovery_lag_months`` lands recoveries at the correct future row.
"""

from __future__ import annotations

import numpy as np
import numpy_financial as npf
import pandas as pd
import pytest

from lendquant.cashflows import (
    amortization_schedule,
    level_payment,
    pool_cashflows,
)
from lendquant.rates import mdr_from_cdr


class TestLevelPayment:
    def test_matches_numpy_financial_pmt(self) -> None:
        # npf.pmt returns the payment with a sign convention of outflow
        # from the payer; absolute value is the level payment amount.
        for principal, rate_m, nper in [
            (100.0, 0.01, 3),
            (10_000.0, 0.005, 60),
            (250_000.0, 0.004, 360),
        ]:
            expected = -npf.pmt(rate_m, nper, principal)
            assert level_payment(principal, rate_m, nper) == pytest.approx(
                expected, rel=1e-12
            )

    def test_zero_rate_edge(self) -> None:
        assert level_payment(1000.0, 0.0, 10) == 100.0
        assert level_payment(360.0, 0.0, 360) == 1.0

    def test_single_period(self) -> None:
        # One-period loan: must repay principal plus one period of interest.
        assert level_payment(100.0, 0.01, 1) == pytest.approx(101.0, rel=1e-15)

    def test_negative_nper_raises(self) -> None:
        with pytest.raises(ValueError):
            level_payment(100.0, 0.01, 0)
        with pytest.raises(ValueError):
            level_payment(100.0, 0.01, -5)


class TestAmortizationSchedule:
    def test_columns_and_length(self) -> None:
        df = amortization_schedule(100.0, 0.01, 12)
        assert list(df.columns) == ["mob", "pay", "interest", "principal", "balance"]
        assert len(df) == 12
        assert df["mob"].tolist() == list(range(1, 13))

    def test_pay_equals_interest_plus_principal(self) -> None:
        df = amortization_schedule(100.0, 0.01, 24)
        np.testing.assert_allclose(
            df["pay"].values, df["interest"].values + df["principal"].values, atol=1e-12
        )

    def test_terminal_balance_is_zero(self) -> None:
        df = amortization_schedule(100.0, 0.01, 36)
        assert df["balance"].iloc[-1] == pytest.approx(0.0, abs=1e-10)

    def test_sum_of_principal_equals_original(self) -> None:
        df = amortization_schedule(10_000.0, 0.005, 60)
        assert df["principal"].sum() == pytest.approx(10_000.0, rel=1e-12)

    def test_zero_rate_schedule(self) -> None:
        df = amortization_schedule(1000.0, 0.0, 10)
        np.testing.assert_allclose(df["interest"].values, 0.0)
        np.testing.assert_allclose(df["principal"].values, 100.0)
        assert df["balance"].iloc[-1] == pytest.approx(0.0, abs=1e-12)


class TestPoolCashflowsCleanPool:
    """A pool with zero defaults, zero prepays collapses to amortization."""

    def test_matches_amortization_schedule(self) -> None:
        amort = amortization_schedule(100.0, 0.01, 12)
        pool = pool_cashflows(
            balance=100.0,
            rate_m=0.01,
            nper=12,
            cdr=0.0,
            cpr=0.0,
            severity=0.0,
            recovery_lag_months=0,
        )
        np.testing.assert_allclose(
            pool["scheduled_prin"].values, amort["principal"].values, atol=1e-12
        )
        np.testing.assert_allclose(
            pool["scheduled_int"].values, amort["interest"].values, atol=1e-12
        )
        np.testing.assert_allclose(pool["prepay_prin"].values, 0.0, atol=1e-15)
        np.testing.assert_allclose(pool["default_prin"].values, 0.0, atol=1e-15)
        np.testing.assert_allclose(pool["recovery"].values, 0.0, atol=1e-15)

    def test_terminal_surviving_fraction_zero(self) -> None:
        pool = pool_cashflows(100.0, 0.01, 12, cdr=0.0, cpr=0.0, severity=0.0)
        assert pool["surviving_fraction"].iloc[-1] == pytest.approx(0.0, abs=1e-10)


class TestPoolCashflowsPrepaysOnly:
    """CPR > 0 shrinks the pool without producing defaults."""

    def test_no_defaults_no_recovery(self) -> None:
        pool = pool_cashflows(100.0, 0.01, 12, cdr=0.0, cpr=0.10, severity=0.5)
        assert pool["default_prin"].sum() == pytest.approx(0.0, abs=1e-15)
        assert pool["recovery"].sum() == pytest.approx(0.0, abs=1e-15)

    def test_prepays_accelerate_payoff(self) -> None:
        slow = pool_cashflows(100.0, 0.01, 12, cdr=0.0, cpr=0.0, severity=0.0)
        fast = pool_cashflows(100.0, 0.01, 12, cdr=0.0, cpr=0.5, severity=0.0)
        # Prepay-adjusted principal cashflows arrive earlier.
        assert fast["prepay_prin"].sum() > 0
        assert fast["surviving_fraction"].iloc[0] < slow["surviving_fraction"].iloc[0]

    def test_principal_conservation(self) -> None:
        pool = pool_cashflows(100.0, 0.01, 12, cdr=0.0, cpr=0.10, severity=0.0)
        total_principal = (
            pool["scheduled_prin"].sum() + pool["prepay_prin"].sum()
        )
        assert total_principal == pytest.approx(100.0, rel=1e-10)


class TestPoolCashflowsDefaults:
    """CDR > 0 with severity extremes isolates default vs. recovery accounting."""

    def test_full_severity_zero_recovery(self) -> None:
        pool = pool_cashflows(
            100.0, 0.01, 12, cdr=0.1, cpr=0.0, severity=1.0, recovery_lag_months=0
        )
        assert pool["default_prin"].sum() > 0
        assert pool["recovery"].sum() == pytest.approx(0.0, abs=1e-15)

    def test_zero_severity_full_recovery(self) -> None:
        pool = pool_cashflows(
            100.0, 0.01, 12, cdr=0.1, cpr=0.0, severity=0.0, recovery_lag_months=0
        )
        assert pool["recovery"].sum() == pytest.approx(
            pool["default_prin"].sum(), rel=1e-12
        )

    def test_principal_plus_default_equals_original(self) -> None:
        pool = pool_cashflows(
            100.0, 0.01, 12, cdr=0.1, cpr=0.05, severity=0.5, recovery_lag_months=0
        )
        accounted = (
            pool["scheduled_prin"].sum()
            + pool["prepay_prin"].sum()
            + pool["default_prin"].sum()
        )
        assert accounted == pytest.approx(100.0, rel=1e-10)


class TestPoolCashflowsHandCalc:
    """3-period, CDR=0.10, CPR=0, severity=0.5 (SPEC validation example).

    Hand-calc with balance=100, rate_m=0.01:
        MDR = 1 - 0.9**(1/12) ≈ 0.0087307...
        Month 1: default=100*MDR, bal_after=100-default, pmt on (nper-0=3),
                 scheduled_int = bal_after * 0.01, scheduled_prin = pmt - int,
                 recovery = default * 0.5 (lands month 1 with lag=0).
    The test cross-checks against a re-derivation of the same loop.
    """

    @staticmethod
    def _hand_calc(
        balance: float,
        rate_m: float,
        nper: int,
        cdr: float,
        severity: float,
    ) -> pd.DataFrame:
        mdr = float(mdr_from_cdr(cdr))
        rows = []
        bal = balance
        for _ in range(nper):
            d = bal * mdr
            b1 = bal - d
            # No prepay in this example.
            remaining = nper - _
            pmt = level_payment(b1, rate_m, remaining)
            i_sch = b1 * rate_m
            p_sch = min(pmt - i_sch, b1)
            rec = d * (1.0 - severity)
            rows.append((d, i_sch, p_sch, rec, b1 - p_sch))
            bal = b1 - p_sch
        return pd.DataFrame(
            rows,
            columns=["default", "interest", "principal", "recovery", "end_bal"],
        )

    def test_matches_hand_calc(self) -> None:
        expected = self._hand_calc(100.0, 0.01, 3, cdr=0.10, severity=0.5)
        pool = pool_cashflows(
            100.0, 0.01, 3, cdr=0.10, cpr=0.0, severity=0.5, recovery_lag_months=0
        )
        np.testing.assert_allclose(
            pool["default_prin"].values, expected["default"].values, atol=1e-10
        )
        np.testing.assert_allclose(
            pool["scheduled_int"].values, expected["interest"].values, atol=1e-10
        )
        np.testing.assert_allclose(
            pool["scheduled_prin"].values, expected["principal"].values, atol=1e-10
        )
        np.testing.assert_allclose(
            pool["recovery"].values, expected["recovery"].values, atol=1e-10
        )
        # Terminal balance (= 1 - surviving_fraction × original_balance after
        # defaults removed) sits at zero after all scheduled payments.
        assert pool["surviving_fraction"].iloc[-1] == pytest.approx(0.0, abs=1e-10)


class TestPoolCashflowsRecoveryLag:
    def test_lag_extends_frame(self) -> None:
        pool = pool_cashflows(
            100.0, 0.01, 12, cdr=0.1, cpr=0.0, severity=0.5, recovery_lag_months=3
        )
        assert len(pool) == 12 + 3
        # Scheduled columns beyond month 12 are zero.
        tail_sched = pool.iloc[12:][
            ["scheduled_prin", "scheduled_int", "prepay_prin", "default_prin"]
        ]
        assert (tail_sched.values == 0).all()

    def test_recovery_shifts_by_lag(self) -> None:
        lag0 = pool_cashflows(100.0, 0.01, 6, cdr=0.1, cpr=0.0, severity=0.5)
        lag2 = pool_cashflows(
            100.0, 0.01, 6, cdr=0.1, cpr=0.0, severity=0.5, recovery_lag_months=2
        )
        # The total recovery amount is the same; it's just shifted later.
        assert lag0["recovery"].sum() == pytest.approx(lag2["recovery"].sum(), rel=1e-12)
        # Recoveries from month-1 default land at index 0 with lag=0 and
        # index 2 with lag=2.
        assert lag2["recovery"].iloc[0] == pytest.approx(0.0, abs=1e-15)
        assert lag2["recovery"].iloc[1] == pytest.approx(0.0, abs=1e-15)
        assert lag2["recovery"].iloc[2] == pytest.approx(
            lag0["recovery"].iloc[0], rel=1e-12
        )

    def test_negative_lag_raises(self) -> None:
        with pytest.raises(ValueError):
            pool_cashflows(100.0, 0.01, 6, cdr=0.0, cpr=0.0, severity=0.0, recovery_lag_months=-1)


class TestPoolCashflowsInputShapes:
    def test_scalar_vectors_broadcast(self) -> None:
        pool_scalar = pool_cashflows(100.0, 0.01, 6, cdr=0.1, cpr=0.05, severity=0.5)
        pool_vector = pool_cashflows(
            100.0,
            0.01,
            6,
            cdr=np.full(6, 0.1),
            cpr=np.full(6, 0.05),
            severity=np.full(6, 0.5),
        )
        for col in ["scheduled_prin", "scheduled_int", "prepay_prin", "default_prin"]:
            np.testing.assert_allclose(
                pool_scalar[col].values, pool_vector[col].values, atol=1e-12
            )

    def test_time_varying_cdr(self) -> None:
        # CDR ramps up through the pool life.
        cdr = np.linspace(0.0, 0.2, 6)
        pool = pool_cashflows(100.0, 0.01, 6, cdr=cdr, cpr=0.0, severity=0.5)
        # First-period default is zero (CDR=0 at t=0); defaults strictly
        # increase in dollar terms through the array positions where CDR
        # is rising faster than the balance is shrinking — not a strict
        # monotonicity test, but the first and last period are a clean check.
        assert pool["default_prin"].iloc[0] == pytest.approx(0.0, abs=1e-15)
        assert pool["default_prin"].iloc[-1] > 0
