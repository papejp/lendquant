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

## Scope

`lendquant` does the math — scenario-driven, deterministic, zero model
opinions. If you want forecasting, calibration, scoring, segment
selection, or price recommendations, use [LendQuant Pro](https://lendquant.com).

> *Free does the math; Pro does the judgment.*

## License

Apache-2.0.
