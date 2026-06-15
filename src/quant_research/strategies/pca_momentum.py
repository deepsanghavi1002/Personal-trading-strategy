from __future__ import annotations

import pandas as pd

from quant_research.features import moving_average, target_volatility_weight
from quant_research.strategies.base import Strategy


class PCAMomentumStrategy(Strategy):
    name = "pca_momentum"

    def generate_weights(self, prices: pd.DataFrame, features: dict, config: dict) -> pd.Series:
        target = config["data"]["target"]
        settings = config["strategy"]
        feature_cfg = config["features"]

        price = prices[target]
        ret = features["returns"][target]
        pc1 = features["pc1_factor"].reindex(price.index)
        vam = features["vol_adj_momentum"].reindex(price.index)

        trend = moving_average(price, int(feature_cfg["trend_window"]))
        fast_line = moving_average(price, int(feature_cfg["fast_exit_window"]))

        buy_rule = (pc1 > 0) & (price > trend) & (vam > float(settings["momentum_entry"]))
        sell_rule = (pc1 < 0) | (price < fast_line) | (vam < float(settings["momentum_exit"]))

        raw = pd.Series(0.0, index=price.index, name=self.name)
        is_invested = False
        for date in raw.index:
            if (not is_invested) and bool(buy_rule.loc[date]):
                is_invested = True
            elif is_invested and bool(sell_rule.loc[date]):
                is_invested = False
            raw.loc[date] = 1.0 if is_invested else 0.0

        if bool(settings.get("use_volatility_targeting", True)):
            vol_weight = target_volatility_weight(
                ret,
                float(settings["target_volatility"]),
                int(feature_cfg["volatility_window"]),
                float(settings["max_leverage"]),
                int(feature_cfg["annualization"]),
            )
            raw = raw * vol_weight.reindex(raw.index).fillna(0.0)

        return raw.shift(1).fillna(0.0).rename(self.name)
