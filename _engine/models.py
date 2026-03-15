from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Mode = Literal["long", "short", "both"]
Objective = Literal[
    "net_profit_pct",
    "profit_factor",
    "win_rate_pct",
    "max_drawdown_pct",
    "trade_count",
]
SearchMethod = Literal["grid", "bayesian"]


# ---------------------------------------------------------------------------
# Request and input models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandleRequest:
    symbol: str
    exchange: str
    timeframe: str
    start: str | None = None
    end: str | None = None
    session: str = "regular"
    adjustment: str = "splits"
    mintick: float = 0.01


@dataclass(frozen=True)
class RiskParameters:
    long_stoploss_pct: float
    long_takeprofit_pct: float
    short_stoploss_pct: float
    short_takeprofit_pct: float


@dataclass(frozen=True)
class OptimizationRequest:
    candle_request: CandleRequest
    mode: Mode
    objective: Objective
    sl_min: float
    sl_max: float
    tp_min: float
    tp_max: float
    top_n: int = 10
    initial_equity: float = 100_000.0
    search_method: SearchMethod = "grid"
    n_trials: int = 200


# ---------------------------------------------------------------------------
# Runtime and result models
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: int
    exit_time: int
    direction: Literal["long", "short"]
    entry_price: float
    exit_price: float
    exit_reason: str
    return_pct: float
    equity_before: float
    equity_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestMetrics:
    symbol: str
    timeframe: str
    start: str | None
    end: str | None
    mode: Mode
    sl_pct: float
    tp_pct: float
    net_profit_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    trade_count: int
    equity_final: float
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    expectancy_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    worst_trade_pct: float = 0.0
    max_consec_losses: int = 0
    sl_exit_pct: float = 0.0
    tp_exit_pct: float = 0.0
    signal_exit_pct: float = 0.0
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _rounded_metrics_dict(self)


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.metrics.to_dict()
        payload["trades"] = [trade.to_dict() for trade in self.trades]
        payload["equity_curve"] = self.equity_curve
        return payload


@dataclass
class OptimizationBundle:
    request: OptimizationRequest
    results: list[BacktestMetrics]
    coarse_results: list[BacktestMetrics]
    output_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": {
                "candle_request": asdict(self.request.candle_request),
                "mode": self.request.mode,
                "objective": self.request.objective,
                "sl_min": self.request.sl_min,
                "sl_max": self.request.sl_max,
                "tp_min": self.request.tp_min,
                "tp_max": self.request.tp_max,
                "top_n": self.request.top_n,
                "initial_equity": self.request.initial_equity,
                "search_method": self.request.search_method,
                "n_trials": self.request.n_trials,
            },
            "results": [result.to_dict() for result in self.results],
            "coarse_results": [result.to_dict() for result in self.coarse_results],
        }


@dataclass
class MultiPairCandidate:
    """A single SL/TP candidate evaluated across multiple pairs."""
    sl_pct: float
    tp_pct: float
    aggregate_net_profit_pct: float
    aggregate_max_drawdown_pct: float
    aggregate_objective: float
    per_pair_metrics: list[BacktestMetrics]
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sl_pct": round(self.sl_pct, 4),
            "tp_pct": round(self.tp_pct, 4),
            "aggregate_net_profit_pct": round(self.aggregate_net_profit_pct, 2),
            "aggregate_max_drawdown_pct": round(self.aggregate_max_drawdown_pct, 2),
            "aggregate_objective": round(self.aggregate_objective, 4),
            "rank": self.rank,
            "per_pair_metrics": [m.to_dict() for m in self.per_pair_metrics],
        }


@dataclass
class MultiPairOptimizationBundle:
    """Optimization results for multi-pair simultaneous optimization."""
    request: OptimizationRequest
    pairs: list[tuple[str, str]]  # (symbol, exchange) tuples
    results: list[MultiPairCandidate]
    coarse_results: list[MultiPairCandidate]
    output_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": {
                "candle_request": {
                    "timeframe": self.request.candle_request.timeframe,
                    "session": self.request.candle_request.session,
                },
                "pairs": [f"{ex}:{sym}" for sym, ex in self.pairs],
                "mode": self.request.mode,
                "objective": self.request.objective,
                "sl_min": self.request.sl_min,
                "sl_max": self.request.sl_max,
                "tp_min": self.request.tp_min,
                "tp_max": self.request.tp_max,
                "top_n": self.request.top_n,
                "initial_equity": self.request.initial_equity,
                "search_method": self.request.search_method,
                "n_trials": self.request.n_trials,
            },
            "results": [c.to_dict() for c in self.results],
            "coarse_results": [c.to_dict() for c in self.coarse_results],
        }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

_ROUNDED_METRIC_FIELDS = (
    "sl_pct",
    "tp_pct",
    "net_profit_pct",
    "max_drawdown_pct",
    "win_rate_pct",
    "profit_factor",
    "equity_final",
    "sharpe_ratio",
    "calmar_ratio",
    "expectancy_pct",
    "avg_win_pct",
    "avg_loss_pct",
    "worst_trade_pct",
    "sl_exit_pct",
    "tp_exit_pct",
    "signal_exit_pct",
)


def _rounded_metrics_dict(metrics: BacktestMetrics) -> dict[str, Any]:
    data = asdict(metrics)
    for key in _ROUNDED_METRIC_FIELDS:
        if isinstance(data.get(key), float):
            data[key] = round(data[key], 2)
    return data
