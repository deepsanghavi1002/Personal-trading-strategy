# Personal Trading Strategy Research Platform

A mini hedge-fund style research platform for testing precious metals ETF strategies.

The first research question is simple:

> Does a rolling PCA regime filter improve a leveraged precious metals ETF strategy compared with simpler trend-following rules?

The platform is designed so future strategies can plug into the same engine: PCA, Bollinger/EMA, momentum, volatility targeting, regime switching, machine learning, and options overlays.

## Current universe

Default symbols are configured in `config/strategy.yaml`:

- `AGQ` - 2x silver ETF, target strategy instrument
- `UGL` - 2x gold ETF
- `GDX` - gold miners
- `SIL` - silver miners
- `GLD` - gold ETF
- `SLV` - silver ETF
- `TLT` - long treasury proxy
- `UUP` - US dollar proxy

## Strategies included

### 1. Trend following baseline

Long the target ETF when price is above the 200-day moving average.

### 2. PCA momentum strategy

Long the target ETF only when all conditions line up:

- PC1 regime factor is positive
- Target ETF is above its long-term trend line
- Volatility-adjusted momentum is strong
- Optional volatility targeting sizes the position

### 3. Bollinger/EMA template

A reusable pullback strategy template that can be tuned later.

## Project structure

```text
config/
  strategy.yaml
scripts/
  run_research.py
src/quant_research/
  data.py
  features.py
  pca.py
  research.py
  analysis_tools.py
  plotting.py
  backtest/
    engine.py
    metrics.py
  strategies/
    base.py
    trend.py
    pca_momentum.py
    bollinger_ema.py
tests/
  test_metrics.py
```

## Setup

```bash
git clone https://github.com/deepsanghavi1002/Personal-trading-strategy.git
cd Personal-trading-strategy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run the research

```bash
python scripts/run_research.py
```

This downloads Yahoo Finance prices and writes outputs to:

```text
outputs/reports/
outputs/plots/
```

Generated reports include:

- `metrics.csv`
- `equity_curves.csv`
- `pca_vs_trend_comparison.csv`
- `pc1_explained_variance.csv`
- `pc1_loadings.csv`

Generated plots include:

- `equity_curves.png`
- `drawdowns.png`
- `pca_regime.png`

## How to interpret the PCA test

The important comparison is not only whether the PCA strategy makes money. The better question is:

> Does PCA improve risk-adjusted performance versus a simple 200-day moving average strategy?

Look at:

- Sharpe ratio
- Max drawdown
- Calmar ratio
- Final equity
- Average daily edge in `pca_vs_trend_comparison.csv`

If PCA improves drawdown and Sharpe but not raw return, that can still be valuable because leveraged ETFs are highly sensitive to bad regimes.

## Add a new strategy

Create a new file under `src/quant_research/strategies/` and inherit from `Strategy`.

```python
from quant_research.strategies.base import Strategy

class MyStrategy(Strategy):
    name = "my_strategy"

    def generate_weights(self, prices, features, config):
        # return a pandas Series of target weights
        pass
```

Then add it to `src/quant_research/research.py` in the `strategies` list.

## Important note

This is a research and education project, not investment advice. Leveraged ETFs can lose value quickly, especially during sideways or volatile markets. Always validate assumptions with out-of-sample tests before risking capital.
