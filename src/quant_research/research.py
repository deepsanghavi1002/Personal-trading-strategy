from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research.analysis_tools import improvement_summary
from quant_research.backtest.engine import BacktestResult, run_backtest
from quant_research.data import daily_returns, load_prices
from quant_research.features import volatility_adjusted_momentum
from quant_research.pca import rolling_first_pc
from quant_research.strategies import BollingerEMAStrategy, PCAMomentumStrategy, TrendFollowingStrategy


def build_features(prices: pd.DataFrame, config: dict) -> dict:
    target = config["data"]["target"]
    feature_cfg = config["features"]
    pca_cfg = config["pca"]

    returns = daily_returns(prices)
    pc_result = rolling_first_pc(
        returns,
        window=int(pca_cfg["window"]),
        min_periods=int(pca_cfg["min_periods"]),
        positive_tickers=list(pca_cfg.get("positive_tickers", [])),
    )

    features = {
        "returns": returns,
        "pc1_factor": pc_result.factor,
        "pc1_explained_variance": pc_result.explained_variance,
        "pc1_loadings": pc_result.loadings,
        "vol_adj_momentum": volatility_adjusted_momentum(
            prices[target],
            returns[target],
            int(feature_cfg["momentum_window"]),
            int(feature_cfg["volatility_window"]),
            int(feature_cfg["annualization"]),
        ),
    }
    return features


def run_research(config: dict, output_dir: Path | None = None) -> dict:
    data_cfg = config["data"]
    strategy_cfg = config["strategy"]
    feature_cfg = config["features"]
    target = data_cfg["target"]

    prices = load_prices(
        data_cfg["universe"],
        start=data_cfg["start"],
        end=data_cfg.get("end"),
        cache_path=(output_dir / "data" / "prices.csv") if output_dir else None,
    )
    features = build_features(prices, config)

    strategies = [
        TrendFollowingStrategy(),
        PCAMomentumStrategy(),
        BollingerEMAStrategy(),
    ]

    results: dict[str, BacktestResult] = {}
    for strategy in strategies:
        weights = strategy.generate_weights(prices, features, config)
        results[strategy.name] = run_backtest(
            strategy.name,
            prices,
            weights,
            target=target,
            initial_capital=float(strategy_cfg["initial_capital"]),
            transaction_cost_bps=float(strategy_cfg["transaction_cost_bps"]),
            annualization=int(feature_cfg["annualization"]),
        )

    metrics = pd.DataFrame({name: res.metrics for name, res in results.items()}).T
    comparison = improvement_summary(
        results["pca_momentum"].strategy_returns,
        results["trend_following"].strategy_returns,
    )

    if output_dir is not None:
        metrics.to_csv(output_dir / "reports" / "metrics.csv")
        pd.DataFrame({name: res.equity for name, res in results.items()}).to_csv(
            output_dir / "reports" / "equity_curves.csv"
        )
        pd.Series(comparison).to_csv(output_dir / "reports" / "pca_vs_trend_comparison.csv")
        features["pc1_explained_variance"].to_csv(output_dir / "reports" / "pc1_explained_variance.csv")
        features["pc1_loadings"].to_csv(output_dir / "reports" / "pc1_loadings.csv")

    return {
        "prices": prices,
        "features": features,
        "results": results,
        "metrics": metrics,
        "pca_vs_trend": comparison,
    }
