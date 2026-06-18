# ML Time-Series Project: Scikit-Learn Quant Signal Model

This is a second project inside the repo. It is designed for an entry / mid-level quant or QIS interview.

## Goal

Use supervised machine learning to predict whether a stock or ETF will have a positive future return.

The project uses time-series-safe validation, meaning it trains on older data and tests on newer data. This avoids look-ahead bias.

## Instruments

Default example:

- `SPY` as the main asset
- Optional extension to `AGQ`, `GLD`, `SLV`, `TLT`, `UUP`

## Libraries used

- `pandas`: time-series manipulation
- `numpy`: numerical calculations
- `yfinance`: market data download
- `scikit-learn`: machine learning models and time-series validation
- `matplotlib`: plots

## Models included

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

## Features

The model uses only past information:

- 1-day return
- 5-day return
- 20-day return
- 20-day realized volatility
- 20-day moving-average distance
- 50-day moving-average distance
- Volume change

## Target

The default target is:

```text
1 if next-day return is positive
0 otherwise
```

## Why this matters in quant interviews

This project demonstrates:

- Feature engineering
- Classification
- Time-series train/test split
- Avoiding look-ahead bias
- Turning probability predictions into trading signals
- Backtesting model outputs
- Comparing accuracy with trading metrics

## How to run

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
python ml_time_series_project/run_ml_timeseries.py
```

Outputs are saved under:

```text
ml_time_series_project/outputs/
```

## Interview summary

You can say:

> I built a time-series ML project using scikit-learn to classify whether an ETF would have a positive next-day return. I engineered lagged returns, volatility, moving-average distance, and volume features. I used TimeSeriesSplit instead of random train-test split to avoid future leakage. Then I compared Logistic Regression, Random Forest, and Gradient Boosting, and converted model probabilities into trading signals for a simple backtest.
