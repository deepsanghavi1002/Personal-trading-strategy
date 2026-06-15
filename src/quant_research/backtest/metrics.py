from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.features import drawdown


def cagr(equity: pd.Series, annualization: int = 252) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = len(equity) / annualization
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    returns = returns.dropna()
    stdev = returns.std()
    if stdev == 0 or returns.empty:
        return float("nan")
    return float(np.sqrt(annualization) * returns.mean() / stdev)


def sortino_ratio(returns: pd.Series, annualization: int = 252) -> float:
    returns = returns.dropna()
    negative_returns = returns[returns < 0]
    stdev = negative_returns.std()
    if stdev == 0 or negative_returns.empty:
        return float("nan")
    return float(np.sqrt(annualization) * returns.mean() / stdev)


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def calmar_ratio(equity: pd.Series, annualization: int = 252) -> float:
    dd_abs = abs(max_drawdown(equity))
    if dd_abs == 0:
        return float("nan")
    return float(cagr(equity, annualization) / dd_abs)


def win_rate(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def performance_summary(equity: pd.Series, returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    clean_equity = equity.dropna()
    final_equity = float(clean_equity.iloc[-1]) if not clean_equity.empty else float("nan")
    return {
        "CAGR": cagr(equity, annualization),
        "Sharpe": sharpe_ratio(returns, annualization),
        "Sortino": sortino_ratio(returns, annualization),
        "Max Drawdown": max_drawdown(equity),
        "Calmar": calmar_ratio(equity, annualization),
        "Win Rate": win_rate(returns),
        "Final Equity": final_equity,
    }
