from __future__ import annotations

from quant_research.strategies import BollingerEMAStrategy, PCAMomentumStrategy, Strategy, TrendFollowingStrategy


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    TrendFollowingStrategy.name: TrendFollowingStrategy,
    PCAMomentumStrategy.name: PCAMomentumStrategy,
    BollingerEMAStrategy.name: BollingerEMAStrategy,
}


def get_strategy(name: str) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise KeyError("Unknown strategy: " + name + ". Available: " + available)
    return STRATEGY_REGISTRY[name]()


def list_strategies() -> list[str]:
    return sorted(STRATEGY_REGISTRY)
