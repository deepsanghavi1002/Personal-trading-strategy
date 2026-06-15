from __future__ import annotations

import pandas as pd

from quant_research.features import moving_average
from quant_research.strategies.base import Strategy


class TrendFollowingStrategy(Strategy):
    name = "trend_following"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        features: dict[str, pd.Series | pd.DataFrame],
        config: dict,
    ) -> pd.Series:
        target = config["data"]["target"]
        trend_window = int(config["features"]["trend_window"])
        price = prices[target]
        trend = moving_average(price, trend_window)
        weights = (price > trend).astype(float)
        return weights.shift(1).fillna(0.0).rename(self.name)
