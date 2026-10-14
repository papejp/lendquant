"""Tests for lendquant.rates.

Validation strategy (per SPEC.md):
- Round-trip identity: f(g(x)) == x within float64 precision.
- SIFMA worked example: CPR 6% → SMM ~0.5143%.
- Edge cases at 0 and 1.
- Nominal vs. effective rate conventions.
- Scalar + array parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from lendquant.rates import (
    annualize_monthly,
    apr_from_monthly,
    cdr_from_mdr,
    cpr_from_smm,
    mdr_from_cdr,
    monthly_from_annualized,
    monthly_from_apr,
    smm_from_cpr,
)


class TestSifmaWorkedExample:
    """SIFMA Standard Formulas / Fabozzi: CPR 6% → SMM 0.5143%."""

    def test_smm_from_cpr_6pct(self) -> None:
        # 1 - 0.94 ** (1/12) = 0.00514333...
        assert smm_from_cpr(0.06) == pytest.approx(0.005143, abs=1e-6)

    def test_cpr_from_smm_round_trips_sifma_value(self) -> None:
        assert cpr_from_smm(smm_from_cpr(0.06)) == pytest.approx(0.06, abs=1e-15)


class TestRoundTripIdentity:
    """f(g(x)) == x to float64 precision."""

    @pytest.mark.parametrize("cpr", [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.99])
    def test_cpr_smm_scalar(self, cpr: float) -> None:
        assert cpr_from_smm(smm_from_cpr(cpr)) == pytest.approx(cpr, abs=1e-12)

    @pytest.mark.parametrize("smm", [0.0, 0.001, 0.005, 0.01, 0.05, 0.10])
    def test_smm_cpr_scalar(self, smm: float) -> None:
        assert smm_from_cpr(cpr_from_smm(smm)) == pytest.approx(smm, abs=1e-12)

    @pytest.mark.parametrize("cdr", [0.0, 0.01, 0.05, 0.10, 0.30, 0.75])
    def test_cdr_mdr_scalar(self, cdr: float) -> None:
        assert cdr_from_mdr(mdr_from_cdr(cdr)) == pytest.approx(cdr, abs=1e-12)

    @pytest.mark.parametrize("apr", [0.0, 0.03, 0.065, 0.12, 0.24])
    def test_apr_monthly_scalar(self, apr: float) -> None:
        assert apr_from_monthly(monthly_from_apr(apr)) == pytest.approx(apr, abs=1e-15)

    @pytest.mark.parametrize("ann", [0.0, 0.03, 0.065, 0.12, 0.24])
    def test_effective_scalar(self, ann: float) -> None:
        assert annualize_monthly(monthly_from_annualized(ann)) == pytest.approx(
            ann, abs=1e-12
        )

    def test_cpr_smm_array(self) -> None:
        cprs = np.array([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.99])
        np.testing.assert_allclose(cpr_from_smm(smm_from_cpr(cprs)), cprs, atol=1e-12)

    def test_cdr_mdr_array(self) -> None:
        cdrs = np.array([0.0, 0.02, 0.08, 0.20, 0.60])
        np.testing.assert_allclose(cdr_from_mdr(mdr_from_cdr(cdrs)), cdrs, atol=1e-12)


class TestEdgeCases:
    """Zero and one map to themselves under both conventions."""

    def test_smm_from_cpr_zero(self) -> None:
        assert smm_from_cpr(0.0) == 0.0

    def test_cpr_from_smm_zero(self) -> None:
        assert cpr_from_smm(0.0) == 0.0

    def test_smm_from_cpr_one(self) -> None:
        # (1 - 1) ** (1/12) == 0, so SMM = 1.
        assert smm_from_cpr(1.0) == 1.0

    def test_cpr_from_smm_one(self) -> None:
        assert cpr_from_smm(1.0) == 1.0

    def test_mdr_from_cdr_zero(self) -> None:
        assert mdr_from_cdr(0.0) == 0.0

    def test_cdr_from_mdr_zero(self) -> None:
        assert cdr_from_mdr(0.0) == 0.0

    def test_mdr_from_cdr_one(self) -> None:
        assert mdr_from_cdr(1.0) == 1.0

    def test_monthly_from_apr_zero(self) -> None:
        assert monthly_from_apr(0.0) == 0.0

    def test_monthly_from_annualized_zero(self) -> None:
        assert monthly_from_annualized(0.0) == 0.0


class TestNominalVsEffectiveConvention:
    """APR is nominal (apr/12); 'annualized' is effective ((1+r)^12 - 1)."""

    def test_apr_6pct_is_half_pct_monthly(self) -> None:
        assert monthly_from_apr(0.06) == pytest.approx(0.005, abs=1e-15)

    def test_effective_6pct_monthly_matches_closed_form(self) -> None:
        # 1.06 ** (1/12) - 1 = 0.00486755...
        r = monthly_from_annualized(0.06)
        assert r == pytest.approx(0.004867551, abs=1e-7)

    def test_effective_monthly_lt_nominal_monthly(self) -> None:
        # For positive rates, the effective monthly rate is strictly less
        # than the nominal apr/12 — this is the whole point of distinguishing
        # the two conventions.
        for ann in [0.03, 0.06, 0.12, 0.24]:
            assert monthly_from_annualized(ann) < monthly_from_apr(ann)


class TestArrayShapePreservation:
    """Array inputs return arrays of matching shape; math is elementwise."""

    def test_smm_from_cpr_1d(self) -> None:
        cprs = np.array([0.0, 0.06, 0.12, 1.0])
        smms = smm_from_cpr(cprs)
        assert smms.shape == cprs.shape
        assert smms[0] == 0.0
        assert smms[-1] == 1.0

    def test_monthly_from_apr_elementwise(self) -> None:
        aprs = np.array([0.06, 0.12, 0.18])
        np.testing.assert_allclose(monthly_from_apr(aprs), aprs / 12.0)

    def test_annualize_monthly_2d(self) -> None:
        r_m = np.array([[0.001, 0.005], [0.01, 0.02]])
        ann = annualize_monthly(r_m)
        assert ann.shape == r_m.shape
        np.testing.assert_allclose(ann, (1.0 + r_m) ** 12 - 1.0)
