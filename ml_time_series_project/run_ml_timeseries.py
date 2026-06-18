from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_data(ticker: str = "SPY", start: str = "2010-01-01") -> pd.DataFrame:
    data = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("No data returned. Check ticker or internet connection.")
    return data[["Close", "Volume"]].dropna()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["ret_1d"] = df["Close"].pct_change()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)

    df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)

    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["dist_ma20"] = df["Close"] / df["ma_20"] - 1
    df["dist_ma50"] = df["Close"] / df["ma_50"] - 1

    df["volume_change_5d"] = df["Volume"].pct_change(5)

    df["future_return_1d"] = df["Close"].pct_change().shift(-1)
    df["target"] = (df["future_return_1d"] > 0).astype(int)

    return df.dropna()


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=25,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            random_state=42,
        ),
    }


def walk_forward_train_predict(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[features]
    y = df["target"]

    splitter = TimeSeriesSplit(n_splits=5)
    models = get_models()

    prediction_frames = []
    score_rows = []

    for model_name, model in models.items():
        fold_predictions = []

        for fold, (train_index, test_index) in enumerate(splitter.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            prob_up = model.predict_proba(X_test)[:, 1]
            pred = (prob_up > 0.5).astype(int)

            score_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "accuracy": accuracy_score(y_test, pred),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                }
            )

            fold_df = pd.DataFrame(
                {
                    "date": X_test.index,
                    "model": model_name,
                    "prob_up": prob_up,
                    "prediction": pred,
                    "actual": y_test.values,
                    "future_return_1d": df.loc[X_test.index, "future_return_1d"].values,
                }
            )
            fold_predictions.append(fold_df)

        prediction_frames.append(pd.concat(fold_predictions, ignore_index=True))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    return predictions, scores


def backtest_predictions(predictions: pd.DataFrame, threshold: float = 0.55, cost_bps: float = 5.0) -> pd.DataFrame:
    rows = []

    for model_name, group in predictions.groupby("model"):
        bt = group.sort_values("date").copy()
        bt["signal"] = (bt["prob_up"] > threshold).astype(int)
        bt["position"] = bt["signal"].shift(1).fillna(0)
        bt["turnover"] = bt["position"].diff().abs().fillna(bt["position"].abs())
        bt["cost"] = bt["turnover"] * (cost_bps / 10000.0)
        bt["strategy_return"] = bt["position"] * bt["future_return_1d"] - bt["cost"]
        bt["equity"] = (1 + bt["strategy_return"]).cumprod()
        rows.append(bt)

    return pd.concat(rows, ignore_index=True)


def summarize_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for model_name, group in backtest.groupby("model"):
        ret = group["strategy_return"].dropna()
        equity = group["equity"].dropna()
        if ret.empty or equity.empty:
            continue

        ann_return = equity.iloc[-1] ** (252 / len(equity)) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
        drawdown = equity / equity.cummax() - 1

        summary_rows.append(
            {
                "model": model_name,
                "annual_return": ann_return,
                "annual_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": drawdown.min(),
                "final_equity": equity.iloc[-1],
            }
        )

    return pd.DataFrame(summary_rows)


def plot_equity(backtest: pd.DataFrame) -> None:
    pivot = backtest.pivot(index="date", columns="model", values="equity")
    ax = pivot.plot(figsize=(12, 6), title="ML time-series model equity curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png")
    plt.close()


def main() -> None:
    ticker = "SPY"
    raw = download_data(ticker)
    df = build_features(raw)

    features = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_20d",
        "dist_ma20",
        "dist_ma50",
        "volume_change_5d",
    ]

    predictions, scores = walk_forward_train_predict(df, features)
    backtest = backtest_predictions(predictions)
    summary = summarize_backtest(backtest)

    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "model_scores.csv", index=False)
    backtest.to_csv(OUTPUT_DIR / "backtest.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
    plot_equity(backtest)

    print("Model scores:")
    print(scores.groupby("model")[["accuracy", "precision", "recall"]].mean().round(4))
    print("\nBacktest summary:")
    print(summary.round(4))
    print("\nOutputs saved to " + str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
