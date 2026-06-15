from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_research.features import drawdown


def plot_equity_curves(equity: pd.DataFrame, path: str | Path) -> None:
    ax = equity.dropna(how="all").plot(figsize=(12, 6), title="Strategy Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value")
    ax.grid(True, alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_drawdowns(equity: pd.DataFrame, path: str | Path) -> None:
    dd = equity.apply(drawdown)
    ax = dd.plot(figsize=(12, 6), title="Drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pca_regime(pc1: pd.Series, path: str | Path) -> None:
    ax = pc1.dropna().plot(figsize=(12, 4), title="Rolling PC1 Regime Factor")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("PC1 score")
    ax.grid(True, alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
