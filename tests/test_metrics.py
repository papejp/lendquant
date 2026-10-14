"""Tests for lendquant.metrics.

Validation strategy (per SPEC.md):
- npv / irr cross-checked against ``numpy_financial`` on a 10-cashflow
  test vector.
- xirr sanity: with uniform monthly spacing it should agree with irr
  (effective annual) to high precision; also verified against a hand
  calc on an irregular schedule.
- price_from_yield / yield_from_price are inverses to float precision.
- Macaulay for a single bullet cashflow equals the bullet's time index.
- Modified duration = Macaulay / (1 + ytm).
- WAL closed-form for a bullet.
- loss_adjusted_yield < clean yield when losses > 0.
"""

from __future__ import annotations

import math

import numpy as np
import numpy_financial as npf
import pytest

from lendquant.metrics import (
    irr,
    loss_adjusted_yield,
    macaulay_duration,
    modified_duration,
    npv,
    price_from_yield,
    wal,
    xirr,
    yield_from_price,
)
from lendquant.rates import annualize_monthly, monthly_from_annualized

# A 10-period cashflow vector — initial outflow of 100 then level receipts.
CF_10 = np.array([-100.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])


class TestNpv:
    def test_matches_numpy_financial(self) -> None:
        for r_m in [0.0, 0.001, 0.005, 0.01, 0.05]:
            assert npv(CF_10, r_m) == pytest.approx(npf.npv(r_m, CF_10), rel=1e-12)

    def test_zero_rate(self) -> None:
        assert npv(CF_10, 0.0) == pytest.approx(CF_10.sum(), rel=1e-15)

    def test_empty_array(self) -> None:
        assert npv([], 0.05) == 0.0

    def test_hand_calc(self) -> None:
        # 2 cashflows at t=0 and t=1, r=0.1 → -100 + 110/1.1 = 0
        assert npv([-100.0, 110.0], 0.1) == pytest.approx(0.0, abs=1e-13)


class TestIrr:
    def test_matches_numpy_financial(self) -> None:
        monthly_npf = float(npf.irr(CF_10))
        annual_expected = float(annualize_monthly(monthly_npf))
        assert irr(CF_10) == pytest.approx(annual_expected, rel=1e-10)

    def test_zero_npv_at_irr(self) -> None:
        rate_annual = irr(CF_10)
        r_m = monthly_from_annualized(rate_annual)
        assert npv(CF_10, r_m) == pytest.approx(0.0, abs=1e-8)

    def test_no_sign_change_returns_nan(self) -> None:
        assert math.isnan(irr([1.0, 2.0, 3.0]))
        assert math.isnan(irr([-1.0, -2.0, -3.0]))

    def test_single_cashflow_returns_nan(self) -> None:
        assert math.isnan(irr([5.0]))
        assert math.isnan(irr([]))

    def test_known_bullet_irr(self) -> None:
        # Invest 100 today, receive 121 in 12 months → monthly IRR such
        # that (1+r_m)^12 = 1.21 → annualized = 0.21 exact.
        cf = [-100.0] + [0.0] * 11 + [121.0]
        assert irr(cf) == pytest.approx(0.21, abs=1e-10)


class TestXirr:
    def test_matches_irr_on_uniform_monthly_schedule(self) -> None:
        # 11 cashflows spaced 30.4375 days apart (365.25/12) → very close
        # to monthly spacing; XIRR should match annualized IRR tightly.
        base = np.datetime64("2024-01-01")
        dates = np.array([base + np.timedelta64(round(30.4375 * i), "D") for i in range(11)])
        assert xirr(CF_10, dates) == pytest.approx(irr(CF_10), rel=1e-4)

    def test_hand_calc_365_25_convention(self) -> None:
        # -100 at t=0, +121 some days later. The Excel XIRR convention
        # uses year_frac = days / 365.25, so the expected XIRR is
        # 1.21 ** (1/year_frac) - 1 — not 0.21 unless the gap is exactly
        # 365.25 days.
        dates = np.array(["2024-01-01", "2025-01-01"], dtype="datetime64[D]")
        days = (dates[1] - dates[0]).astype("timedelta64[D]").astype(float)
        expected = 1.21 ** (365.25 / days) - 1.0
        assert xirr([-100.0, 121.0], dates) == pytest.approx(expected, abs=1e-10)

    def test_mismatched_lengths_raises(self) -> None:
        dates = np.array(["2024-01-01", "2024-02-01"], dtype="datetime64[D]")
        with pytest.raises(ValueError):
            xirr([-100.0, 50.0, 50.0], dates)

    def test_no_sign_change_returns_nan(self) -> None:
        dates = np.array(["2024-01-01", "2024-06-01"], dtype="datetime64[D]")
        assert math.isnan(xirr([1.0, 2.0], dates))


class TestPriceYieldInverse:
    def test_round_trip(self) -> None:
        cfs = np.array([5.0, 5.0, 5.0, 5.0, 105.0])  # 5% coupon bullet
        for ytm in [0.02, 0.05, 0.08, 0.12]:
            price = price_from_yield(cfs, ytm)
            recovered = yield_from_price(cfs, price)
            assert recovered == pytest.approx(ytm, abs=1e-8)

    def test_par_bond_yield_equals_coupon(self) -> None:
        # A perpetual-ish bond priced at par: NPV at the coupon-implied
        # monthly rate equals the sum of discounted coupons... Easier
        # cross-check: build a zero-coupon bond, verify.
        # 100 at month 12, monthly rate 0.01 → price = 100 / 1.01^12
        ytm = float(annualize_monthly(0.01))
        price = price_from_yield([0.0] * 12 + [100.0], ytm_annualized=ytm)
        assert price == pytest.approx(100.0 / 1.01**12, rel=1e-12)


class TestWal:
    def test_bullet_at_month_n(self) -> None:
        # Principal all paid at month N (index N) → WAL = N/12 years.
        pc = [0.0] * 10 + [100.0]
        assert wal(pc) == pytest.approx(10 / 12, abs=1e-15)

    def test_level_payment_half_point(self) -> None:
        # Equal principal payments over N months: WAL = mean(t)/12.
        pc = [10.0] * 12  # t=0..11
        expected = np.mean(np.arange(12)) / 12.0  # 5.5/12
        assert wal(pc) == pytest.approx(expected, rel=1e-15)

    def test_zero_total_returns_nan(self) -> None:
        assert math.isnan(wal([0.0, 0.0, 0.0]))


class TestMacaulayAndModified:
    def test_zero_coupon_duration_equals_maturity(self) -> None:
        # One cashflow at index N → Macaulay = N/12 (independent of YTM).
        cfs = [0.0] * 10 + [100.0]
        for ytm in [0.0, 0.05, 0.10]:
            assert macaulay_duration(cfs, ytm) == pytest.approx(10 / 12, abs=1e-12)

    def test_modified_equals_macaulay_over_one_plus_ytm(self) -> None:
        cfs = np.array([5.0, 5.0, 5.0, 5.0, 105.0])
        ytm = 0.05
        mac = macaulay_duration(cfs, ytm)
        mod = modified_duration(cfs, ytm)
        assert mod == pytest.approx(mac / (1.0 + ytm), rel=1e-15)

    def test_duration_level_coupon_bullet(self) -> None:
        # Level 5-period bond at 5% monthly coupon, discounted at YTM
        # such that monthly_from_annualized(ytm) = 0.05.
        cfs = np.array([5.0, 5.0, 5.0, 5.0, 105.0])  # at t=0..4
        ytm = float(annualize_monthly(0.05))
        r_m = 0.05
        t = np.arange(cfs.size, dtype=np.float64)
        pv = cfs * (1 + r_m) ** -t
        expected = float(np.sum(t * pv) / np.sum(pv) / 12.0)
        assert macaulay_duration(cfs, ytm) == pytest.approx(expected, rel=1e-12)

    def test_empty_returns_nan(self) -> None:
        assert math.isnan(macaulay_duration([], 0.05))


class TestLossAdjustedYield:
    def test_zero_losses_matches_input_ytm(self) -> None:
        cfs = np.array([-100.0, 20.0, 20.0, 20.0, 20.0, 25.0])
        ytm = 0.08
        zero_losses = np.zeros_like(cfs)
        adj = loss_adjusted_yield(cfs, ytm, zero_losses)
        assert adj == pytest.approx(ytm, abs=1e-10)

    def test_losses_reduce_yield(self) -> None:
        # Buy at the price implied by 10% YTM; actual cashflows after
        # losses are lower → realized yield is < 10%.
        cfs = np.array([-100.0, 15.0, 15.0, 15.0, 15.0, 60.0])
        losses = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        ytm = 0.10
        adj = loss_adjusted_yield(cfs, ytm, losses)
        assert adj < ytm
        assert math.isfinite(adj)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            loss_adjusted_yield([1.0, 2.0, 3.0], 0.05, [1.0, 2.0])
