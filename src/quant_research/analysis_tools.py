from __future__ import annotations

import pandas as pd


def compare_series(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if joined.empty:
        return float("nan")
    return float((joined.iloc[:, 0] - joined.iloc[:, 1]).mean())


def improvement_summary(a: pd.Series, b: pd.Series) -> dict[str, float]:
    joined = pd.concat([a, b], axis=1).dropna()
    if joined.empty:
        return {"average_edge": float("nan"), "positive_days": float("nan")}
    diff = joined.iloc[:, 0] - joined.iloc[:, 1]
    return {
        "average_edge": float(diff.mean()),
        "positive_days": float((diff > 0).mean()),
    }
