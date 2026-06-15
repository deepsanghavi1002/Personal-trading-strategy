from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all research strategies.

    A strategy receives historical prices and precomputed features and returns
    a target weight series for the traded asset. A weight of 1.0 means fully
    invested, 0.0 means cash, and values between them support volatility sizing.
    """

    name: str = "base"

    @abstractmethod
    def generate_weights(
        self,
        prices: pd.DataFrame,
        features: dict[str, pd.Series | pd.DataFrame],
        config: dict,
    ) -> pd.Series:
        raise NotImplementedError
