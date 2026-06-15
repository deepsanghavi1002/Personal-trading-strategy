import pandas as pd

from quant_research.backtest.metrics import win_rate


def test_win_rate():
    series = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert win_rate(series) == 0.5
