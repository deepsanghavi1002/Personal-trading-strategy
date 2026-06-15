from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def load_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download adjusted daily close prices from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    symbols = list(dict.fromkeys(tickers))
    if not symbols:
        raise ValueError("At least one ticker is required")

    if cache_path is not None:
        cache_path = Path(cache_path)
        if use_cache and cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return cached.sort_index()

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError("No price data returned by yfinance")

    if isinstance(raw.columns, pd.MultiIndex):
        field = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
        prices = raw[field].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = symbols

    prices = prices.sort_index().ffill().dropna(how="all")
    prices = prices.loc[:, [col for col in symbols if col in prices.columns]]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)

    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
