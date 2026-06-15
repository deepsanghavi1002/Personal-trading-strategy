from __future__ import annotations

import pandas as pd

from quant_research.strategies.base import Strategy


class BollingerEMAStrategy(Strategy):
    name = "bollinger_ema"

    def generate_weights(self, prices: pd.DataFrame, features: dict, config: dict) -> pd.Series:
        target = config["data"]["target"]
        price = prices[target]
        ema50 = price.ewm(span=50, adjust=False).mean()
        avg20 = price.rolling(20, min_periods=20).mean()
        dev20 = price.rolling(20, min_periods=20).std()
        low_line = avg20.subtract(dev20.multiply(2.0))
        high_line = avg20.add(dev20.multiply(2.0))
        enter_now = price.gt(ema50) & price.lt(low_line)
        leave_now = price.gt(high_line) | price.lt(ema50)
        result = pd.Series(0.0, index=price.index, name=self.name)
        active = False
        for date in result.index:
            if not active and bool(enter_now.loc[date]):
                active = True
            elif active and bool(leave_now.loc[date]):
                active = False
            result.loc[date] = 1.0 if active else 0.0
        return result.shift(1).fillna(0.0).rename(self.name)
