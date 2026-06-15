from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_research.config import ensure_output_dirs, load_config
from quant_research.plotting import plot_drawdowns, plot_equity_curves, plot_pca_regime
from quant_research.research import run_research


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the precious metals ETF research platform.")
    parser.add_argument("--config", default="config/strategy.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_output_dirs(config)
    research = run_research(config, output_dir=output_dir)

    print("\nPerformance metrics")
    print(research["metrics"].round(4).to_string())

    print("\nPCA vs trend comparison")
    print(research["pca_vs_trend"])

    equity_df = pd.DataFrame({name: result.equity for name, result in research["results"].items()})
    plot_equity_curves(equity_df, output_dir / "plots" / "equity_curves.png")
    plot_drawdowns(equity_df, output_dir / "plots" / "drawdowns.png")
    plot_pca_regime(research["features"]["pc1_factor"], output_dir / "plots" / "pca_regime.png")

    print("\nReports saved to: " + str(output_dir / "reports"))
    print("Plots saved to: " + str(output_dir / "plots"))


if __name__ == "__main__":
    main()
