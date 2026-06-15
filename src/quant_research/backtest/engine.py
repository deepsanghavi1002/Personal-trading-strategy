from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.backtest.metrics import performance_summary


@dataclass
class BacktestResult:
    name: str
    weights: pd.Series
    strategy_returns: pd.Series
    equity: pd.Series
    benchmark_equity: pd.Series
    metrics: dict[str, float]


def run_backtest(
    name: str,
    prices: pd.DataFrame,
    weights: pd.Series,
    target: str,
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 5.0,
    annualization: int = 252,
) -> BacktestResult:
    price = prices[target].dropna()
    asset_returns = price.pct_change().fillna(0.0)
    weights = weights.reindex(price.index).fillna(0.0).clip(lower=0.0)

    turnover = weights.diff().abs().fillna(weights.abs())
    costs = turnover * (transaction_cost_bps / 10000.0)
    strategy_returns = (weights * asset_returns - costs).rename(name)

    equity = (1.0 + strategy_returns).cumprod() * initial_capital
    benchmark_equity = (1.0 + asset_returns).cumprod() * initial_capital
    metrics = performance_summary(equity, strategy_returns, annualization)

    return BacktestResult(
        name=name,
        weights=weights.rename("weight"),
        strategy_returns=strategy_returns,
        equity=equity.rename(name),
        benchmark_equity=benchmark_equity.rename(f"buy_hold_{target}"),
        metrics=metrics,
    )
