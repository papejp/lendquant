"""Tests for lendquant.vintage.

Validation strategy (per SPEC.md):
- Hand-counted synthetic pool for CDR and CPR by MOB.
- CNL closed-form: sum of charge-offs / origination balance.
- combine_cohorts: loan-count-weighted average matches arithmetic
  weighted mean; outer-aligned over different-length cohorts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lendquant.vintage import (
    cdr_by_mob,
    combine_cohorts,
    cpr_by_mob,
    static_pool_cnl,
)


def _synthetic_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    """10 loans, vintage 202401, 36-month term, 100 original balance each.

    Defaults: loan 1 at mob 6, loan 2 at mob 12.
    Prepays: loan 3 at mob 10.
    Each default charges off its full 100 (severity = 1).
    """
    loans = pd.DataFrame(
        {
            "loanid": [f"L{i}" for i in range(1, 11)],
            "vintage": [202401] * 10,
            "original_balance": [100.0] * 10,
            "term": [36] * 10,
        }
    )
    events = pd.DataFrame(
        {
            "loanid": ["L1", "L2", "L3"],
            "mob": [6, 12, 10],
            "event_type": ["default", "default", "prepay"],
            "net_charge_off": [100.0, 100.0, 0.0],
        }
    )
    return loans, events


class TestStaticPoolCNL:
    def test_columns(self) -> None:
        loans, events = _synthetic_pool()
        df = static_pool_cnl(loans, events)
        assert list(df.columns) == [
            "vintage", "mob", "loans_originated", "cnl_rate", "denom"
        ]

    def test_denom_is_origination_total(self) -> None:
        loans, events = _synthetic_pool()
        df = static_pool_cnl(loans, events)
        assert (df["denom"] == 1000.0).all()

    def test_loans_originated(self) -> None:
        loans, events = _synthetic_pool()
        df = static_pool_cnl(loans, events)
        assert (df["loans_originated"] == 10).all()

    def test_cumulative_at_each_mob(self) -> None:
        loans, events = _synthetic_pool()
        df = static_pool_cnl(loans, events).sort_values("mob").reset_index(drop=True)
        # Charge-off=100 at mob 6, 100 at mob 12. CNL = cumsum/1000.
        at_mob = dict(zip(df["mob"], df["cnl_rate"], strict=True))
        assert at_mob[6] == pytest.approx(100.0 / 1000.0)
        assert at_mob[11] == pytest.approx(100.0 / 1000.0)
        assert at_mob[12] == pytest.approx(200.0 / 1000.0)

    def test_no_events_returns_empty(self) -> None:
        loans = pd.DataFrame(
            {
                "loanid": ["L1"],
                "vintage": [202401],
                "original_balance": [100.0],
                "term": [36],
            }
        )
        events = pd.DataFrame(
            {"loanid": [], "mob": [], "event_type": [], "net_charge_off": []}
        )
        df = static_pool_cnl(loans, events)
        assert len(df) == 0


class TestCdrByMob:
    def test_zero_before_first_default(self) -> None:
        loans, events = _synthetic_pool()
        df = cdr_by_mob(loans, events).set_index("mob")
        # No defaults in mobs 1..5.
        for mob in range(1, 6):
            assert df.loc[mob, "cdr"] == pytest.approx(0.0)

    def test_hand_count_at_default_mobs(self) -> None:
        loans, events = _synthetic_pool()
        df = cdr_by_mob(loans, events).set_index("mob")
        # At mob 6: 10 at-risk, 1 default → MDR=0.1 → CDR = 1 - 0.9^12.
        expected_cdr_6 = 1.0 - 0.9**12
        assert df.loc[6, "cdr"] == pytest.approx(expected_cdr_6, rel=1e-12)
        assert df.loc[6, "at_risk"] == 10
        # At mob 12: loan 1 defaulted (mob 6), loan 3 prepaid (mob 10)
        # → 8 at-risk, 1 default → MDR=1/8 → CDR = 1 - (7/8)^12.
        expected_cdr_12 = 1.0 - (7.0 / 8.0) ** 12
        assert df.loc[12, "cdr"] == pytest.approx(expected_cdr_12, rel=1e-12)
        assert df.loc[12, "at_risk"] == 8

    def test_zero_in_quiet_months(self) -> None:
        loans, events = _synthetic_pool()
        df = cdr_by_mob(loans, events).set_index("mob")
        for mob in [7, 8, 9, 11, 13, 14]:
            assert df.loc[mob, "cdr"] == pytest.approx(0.0)

    def test_at_risk_decreases_monotonically(self) -> None:
        loans, events = _synthetic_pool()
        df = cdr_by_mob(loans, events).sort_values("mob").reset_index(drop=True)
        at_risk = df["at_risk"].to_numpy()
        assert np.all(np.diff(at_risk) <= 0)

    def test_mob_grid_covers_term(self) -> None:
        loans, events = _synthetic_pool()
        df = cdr_by_mob(loans, events)
        assert df["mob"].min() == 1
        assert df["mob"].max() == 36


class TestCprByMob:
    def test_prepay_at_hand_counted_mob(self) -> None:
        loans, events = _synthetic_pool()
        df = cpr_by_mob(loans, events).set_index("mob")
        # At mob 10: loan 1 defaulted (mob 6), so 9 at-risk. One prepays.
        # MDR_prepay = 1/9 → CPR = 1 - (8/9)^12.
        expected = 1.0 - (8.0 / 9.0) ** 12
        assert df.loc[10, "cpr"] == pytest.approx(expected, rel=1e-12)
        assert df.loc[10, "at_risk"] == 9

    def test_no_prepay_elsewhere(self) -> None:
        loans, events = _synthetic_pool()
        df = cpr_by_mob(loans, events).set_index("mob")
        for mob in [1, 5, 9, 11, 20]:
            assert df.loc[mob, "cpr"] == pytest.approx(0.0)


class TestCombineCohorts:
    def test_single_cohort_echoes(self) -> None:
        curve = pd.DataFrame(
            {"mob": [1, 2, 3], "cdr": [0.01, 0.02, 0.03], "loans_originated": [100] * 3}
        )
        out = combine_cohorts([curve])
        np.testing.assert_allclose(out["metric_weighted"].values, curve["cdr"].values)

    def test_loan_count_weighted(self) -> None:
        a = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.1, 0.2], "loans_originated": [100] * 2}
        )
        b = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.3, 0.4], "loans_originated": [50] * 2}
        )
        out = combine_cohorts([a, b])
        # mob 1: (100*0.1 + 50*0.3) / 150 = 25/150 = 0.1666...
        # mob 2: (100*0.2 + 50*0.4) / 150 = 40/150 = 0.2666...
        np.testing.assert_allclose(
            out["metric_weighted"].values,
            [(100 * 0.1 + 50 * 0.3) / 150, (100 * 0.2 + 50 * 0.4) / 150],
            rtol=1e-12,
        )

    def test_explicit_weights(self) -> None:
        a = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.1, 0.2], "loans_originated": [1] * 2}
        )
        b = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.3, 0.4], "loans_originated": [1] * 2}
        )
        # Equal counts but explicit 1:3 weighting
        out = combine_cohorts([a, b], weights=[1.0, 3.0])
        # mob 1: (1*0.1 + 3*0.3) / 4 = 1.0/4 = 0.25
        np.testing.assert_allclose(
            out["metric_weighted"].values,
            [(0.1 + 3 * 0.3) / 4, (0.2 + 3 * 0.4) / 4],
            rtol=1e-12,
        )

    def test_different_length_cohorts(self) -> None:
        a = pd.DataFrame(
            {"mob": [1, 2, 3], "cdr": [0.1, 0.1, 0.1], "loans_originated": [100] * 3}
        )
        b = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.3, 0.3], "loans_originated": [100] * 2}
        )
        out = combine_cohorts([a, b]).set_index("mob")
        # Mob 3 is only in a → weighted avg = a's value.
        assert out.loc[3, "metric_weighted"] == pytest.approx(0.1)
        # Mob 1/2: equal weights → 0.2.
        assert out.loc[1, "metric_weighted"] == pytest.approx(0.2)
        assert out.loc[2, "metric_weighted"] == pytest.approx(0.2)

    def test_wrong_weights_length_raises(self) -> None:
        a = pd.DataFrame({"mob": [1], "cdr": [0.1], "loans_originated": [1]})
        with pytest.raises(ValueError):
            combine_cohorts([a], weights=[1.0, 2.0])

    def test_empty_input(self) -> None:
        out = combine_cohorts([])
        assert len(out) == 0

    def test_nan_metric_excluded(self) -> None:
        a = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.1, math.nan], "loans_originated": [100] * 2}
        )
        b = pd.DataFrame(
            {"mob": [1, 2], "cdr": [0.3, 0.3], "loans_originated": [100] * 2}
        )
        out = combine_cohorts([a, b]).set_index("mob")
        # Mob 2: only b contributes → 0.3.
        assert out.loc[2, "metric_weighted"] == pytest.approx(0.3)
