from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(prices: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return prices.rolling(window=window, min_periods=window).mean()


def realized_volatility(returns: pd.Series | pd.DataFrame, window: int, annualization: int = 252) -> pd.Series | pd.DataFrame:
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(annualization)


def momentum(prices: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    return prices.pct_change(window)


def volatility_adjusted_momentum(
    prices: pd.Series,
    returns: pd.Series,
    momentum_window: int,
    volatility_window: int,
    annualization: int = 252,
) -> pd.Series:
    """Momentum divided by annualized realized volatility.

    High values mean the asset has been rising with controlled volatility.
    """
    mom = momentum(prices, momentum_window)
    vol = realized_volatility(returns, volatility_window, annualization)
    return mom / vol.replace(0, np.nan)


def drawdown(equity: pd.Series) -> pd.Series:
    running_high = equity.cummax()
    return equity / running_high - 1.0


def target_volatility_weight(
    returns: pd.Series,
    target_volatility: float,
    volatility_window: int,
    max_leverage: float,
    annualization: int = 252,
) -> pd.Series:
    vol = realized_volatility(returns, volatility_window, annualization)
    weight = target_volatility / vol.replace(0, np.nan)
    return weight.clip(lower=0.0, upper=max_leverage).fillna(0.0)
