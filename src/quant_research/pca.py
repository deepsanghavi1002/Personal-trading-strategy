from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class RollingPCAResult:
    factor: pd.Series
    explained_variance: pd.Series
    loadings: pd.DataFrame


def _standardize(window: pd.DataFrame) -> pd.DataFrame:
    centered = window - window.mean()
    scaled = centered / window.std(ddof=0).replace(0, np.nan)
    return scaled.dropna(axis=1, how="any")


def rolling_first_pc(
    returns: pd.DataFrame,
    window: int = 252,
    min_periods: int = 200,
    positive_tickers: list[str] | None = None,
) -> RollingPCAResult:
    """Compute a rolling first principal component factor."""
    positive_tickers = positive_tickers or list(returns.columns)
    factors = {}
    explained = {}
    loading_rows = []

    for end_idx in range(min_periods - 1, len(returns)):
        start_idx = max(0, end_idx - window + 1)
        sample = returns.iloc[start_idx : end_idx + 1].dropna(how="any")
        if len(sample) < min_periods:
            continue

        standardized = _standardize(sample)
        if standardized.shape[1] < 2:
            continue

        model = PCA(n_components=1)
        scores = model.fit_transform(standardized.values).ravel()
        loadings = pd.Series(model.components_[0], index=standardized.columns)

        metals = [ticker for ticker in positive_tickers if ticker in loadings.index]
        if metals and loadings.loc[metals].mean() < 0:
            scores = -scores
            loadings = -loadings

        date = returns.index[end_idx]
        factors[date] = float(scores[-1])
        explained[date] = float(model.explained_variance_ratio_[0])
        loadings.name = date
        loading_rows.append(loadings)

    return RollingPCAResult(
        factor=pd.Series(factors, name="pc1_factor").sort_index(),
        explained_variance=pd.Series(explained, name="pc1_explained_variance").sort_index(),
        loadings=pd.DataFrame(loading_rows).sort_index(),
    )
