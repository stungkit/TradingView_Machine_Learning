from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from .._utils import _format_time, build_risk
from ..backtest.engine import TradingViewLikeBacktester
from ..models import BacktestMetrics, CandleRequest, Mode, Objective, OptimizationBundle, OptimizationRequest, RiskParameters
from ..models import MultiPairCandidate, MultiPairOptimizationBundle
from strategy import BaseStrategy


def _print_progress(current: int, total: int, start_time: float, phase: str) -> None:
    elapsed = time.time() - start_time
    pct = current / total * 100
    bar_width = 30
    filled = int(bar_width * current // total)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = f"ETA {_format_time(eta)}"
    else:
        eta_str = "ETA --"
    sys.stdout.write(
        f"\r  {phase}: |{bar}| {pct:5.1f}%  [{current}/{total}]  "
        f"elapsed {_format_time(elapsed)}  {eta_str}   "
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def run_optimization(
    signal_frame: pd.DataFrame,
    candle_request: CandleRequest,
    strategy: BaseStrategy,
    request: OptimizationRequest,
    output_path: Path,
    initial_equity: float = 100_000.0,
) -> OptimizationBundle:
    """Top-level entry point for SL/TP optimization."""
    optimizer = _Optimizer(
        signal_frame=signal_frame,
        candle_request=candle_request,
        strategy=strategy,
        initial_equity=initial_equity,
    )
    return optimizer.optimize(request, output_path)


class _Optimizer:
    def __init__(
        self,
        signal_frame: pd.DataFrame,
        candle_request: CandleRequest,
        strategy: BaseStrategy,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.signal_frame = signal_frame
        self.candle_request = candle_request
        self.strategy = strategy
        self.initial_equity = initial_equity
        self.backtester = TradingViewLikeBacktester(
            candle_request=candle_request,
            initial_equity=initial_equity,
        )

    def optimize(self, request: OptimizationRequest, output_path: Path) -> OptimizationBundle:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if request.search_method == "bayesian":
            return self._optimize_bayesian(request, output_path)
        return self._optimize_grid(request, output_path)

    def _optimize_grid(self, request: OptimizationRequest, output_path: Path) -> OptimizationBundle:
        coarse_sl_step = _auto_step(request.sl_min, request.sl_max)
        coarse_tp_step = _auto_step(request.tp_min, request.tp_max)
        sl_values = _build_range(request.sl_min, request.sl_max, coarse_sl_step)
        tp_values = _build_range(request.tp_min, request.tp_max, coarse_tp_step)
        print(f"\n[Coarse grid] {len(sl_values)} SL x {len(tp_values)} TP = {len(sl_values) * len(tp_values)} combinations")
        coarse_results = self._evaluate_grid(
            mode=request.mode,
            objective=request.objective,
            sl_values=sl_values,
            tp_values=tp_values,
            phase="Coarse",
        )

        best_coarse = coarse_results[0]
        fine_sl_step = max(coarse_sl_step / 2.0, 0.0001)
        fine_tp_step = max(coarse_tp_step / 2.0, 0.0001)
        fine_sl = _refine_range(best_coarse.sl_pct, request.sl_min, request.sl_max, coarse_sl_step, fine_sl_step)
        fine_tp = _refine_range(best_coarse.tp_pct, request.tp_min, request.tp_max, coarse_tp_step, fine_tp_step)
        print(f"\n[Fine grid] {len(fine_sl)} SL x {len(fine_tp)} TP = {len(fine_sl) * len(fine_tp)} combinations  (best coarse SL={best_coarse.sl_pct:.4f}% TP={best_coarse.tp_pct:.4f}%)")
        fine_results = self._evaluate_grid(
            mode=request.mode,
            objective=request.objective,
            sl_values=fine_sl,
            tp_values=fine_tp,
            phase="Fine",
        )

        combined = _deduplicate_and_rank(coarse_results + fine_results, request.objective)
        top_results = combined[: request.top_n]
        bundle = OptimizationBundle(
            request=request,
            results=top_results,
            coarse_results=coarse_results[: request.top_n],
            output_path=output_path,
        )
        _write_output(bundle)
        return bundle

    def _optimize_bayesian(self, request: OptimizationRequest, output_path: Path) -> OptimizationBundle:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = request.n_trials
        objective_name = request.objective
        minimize = objective_name == "max_drawdown_pct"

        all_results: list[BacktestMetrics] = []
        start_time = time.time()

        print(f"\n[Bayesian] {n_trials} trials via Optuna TPE  (SL {request.sl_min}–{request.sl_max}, TP {request.tp_min}–{request.tp_max})")

        def objective(trial: optuna.Trial) -> float:
            sl_value = trial.suggest_float("sl_pct", request.sl_min, request.sl_max, step=0.01)
            tp_value = trial.suggest_float("tp_pct", request.tp_min, request.tp_max, step=0.01)
            risk = build_risk(request.mode, sl_value, tp_value)
            result = self.backtester.run(self.signal_frame, risk, request.mode)
            all_results.append(result.metrics)
            _print_progress(len(all_results), n_trials, start_time, "Bayesian")
            return _objective_value(result.metrics, objective_name)

        direction = "minimize" if minimize else "maximize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        ranked = _rank_results(all_results, objective_name)
        top_results = ranked[: request.top_n]
        bundle = OptimizationBundle(
            request=request,
            results=top_results,
            coarse_results=[],
            output_path=output_path,
        )
        _write_output(bundle)
        return bundle

    def _evaluate_grid(
        self,
        mode: Mode,
        objective: Objective,
        sl_values: list[float],
        tp_values: list[float],
        phase: str = "Grid",
    ) -> list[BacktestMetrics]:
        results: list[BacktestMetrics] = []
        total = len(sl_values) * len(tp_values)
        start_time = time.time()
        count = 0
        for sl_value in sl_values:
            for tp_value in tp_values:
                risk = build_risk(mode, sl_value, tp_value)
                result = self.backtester.run(self.signal_frame, risk, mode)
                results.append(result.metrics)
                count += 1
                _print_progress(count, total, start_time, phase)
        return _rank_results(results, objective)


# ------------------------------------------------------------------
# Multi-pair optimizer
# ------------------------------------------------------------------

PairData = tuple[str, str, pd.DataFrame, BaseStrategy, CandleRequest]
"""(symbol, exchange, signal_frame, strategy, candle_request)"""


def run_multi_optimization(
    pair_data: list[PairData],
    request: OptimizationRequest,
    output_path: Path,
    initial_equity: float = 100_000.0,
) -> MultiPairOptimizationBundle:
    """Optimize SL/TP across all pairs simultaneously."""
    optimizer = _MultiPairOptimizer(
        pair_data=pair_data,
        initial_equity=initial_equity,
    )
    return optimizer.optimize(request, output_path)


class _MultiPairOptimizer:
    def __init__(
        self,
        pair_data: list[PairData],
        initial_equity: float = 100_000.0,
    ) -> None:
        self.pair_data = pair_data
        self.initial_equity = initial_equity
        # One backtester per pair
        self.backtesters = [
            TradingViewLikeBacktester(candle_request=cr, initial_equity=initial_equity)
            for _, _, _, _, cr in pair_data
        ]

    def _evaluate_candidate(self, mode: Mode, sl_value: float, tp_value: float, objective: Objective = "net_profit_pct") -> MultiPairCandidate:
        """Run backtest on every pair for a single SL/TP combo and aggregate."""
        risk = build_risk(mode, sl_value, tp_value)
        per_pair: list[BacktestMetrics] = []
        for (sym, ex, sf, strat, cr), bt in zip(self.pair_data, self.backtesters):
            result = bt.run(sf, risk, mode)
            per_pair.append(result.metrics)
        return _build_candidate(sl_value, tp_value, per_pair, request_objective=objective)

    def optimize(self, request: OptimizationRequest, output_path: Path) -> MultiPairOptimizationBundle:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if request.search_method == "bayesian":
            return self._optimize_bayesian(request, output_path)
        return self._optimize_grid(request, output_path)

    def _optimize_grid(self, request: OptimizationRequest, output_path: Path) -> MultiPairOptimizationBundle:
        coarse_sl_step = _auto_step(request.sl_min, request.sl_max)
        coarse_tp_step = _auto_step(request.tp_min, request.tp_max)
        sl_values = _build_range(request.sl_min, request.sl_max, coarse_sl_step)
        tp_values = _build_range(request.tp_min, request.tp_max, coarse_tp_step)
        print(f"\n[Coarse grid] {len(sl_values)} SL x {len(tp_values)} TP = {len(sl_values) * len(tp_values)} combinations  ({len(self.pair_data)} pairs each)")
        coarse_results = self._evaluate_grid(
            mode=request.mode,
            objective=request.objective,
            sl_values=sl_values,
            tp_values=tp_values,
            phase="Coarse",
        )

        best_coarse = coarse_results[0]
        fine_sl_step = max(coarse_sl_step / 2.0, 0.0001)
        fine_tp_step = max(coarse_tp_step / 2.0, 0.0001)
        fine_sl = _refine_range(best_coarse.sl_pct, request.sl_min, request.sl_max, coarse_sl_step, fine_sl_step)
        fine_tp = _refine_range(best_coarse.tp_pct, request.tp_min, request.tp_max, coarse_tp_step, fine_tp_step)
        print(f"\n[Fine grid] {len(fine_sl)} SL x {len(fine_tp)} TP = {len(fine_sl) * len(fine_tp)} combinations  (best coarse SL={best_coarse.sl_pct:.4f}% TP={best_coarse.tp_pct:.4f}%)")
        fine_results = self._evaluate_grid(
            mode=request.mode,
            objective=request.objective,
            sl_values=fine_sl,
            tp_values=fine_tp,
            phase="Fine",
        )

        combined = _deduplicate_and_rank_multi(coarse_results + fine_results, request.objective)
        top_results = combined[: request.top_n]
        pairs = [(sym, ex) for sym, ex, _, _, _ in self.pair_data]
        bundle = MultiPairOptimizationBundle(
            request=request,
            pairs=pairs,
            results=top_results,
            coarse_results=coarse_results[: request.top_n],
            output_path=output_path,
        )
        _write_multi_output(bundle)
        return bundle

    def _optimize_bayesian(self, request: OptimizationRequest, output_path: Path) -> MultiPairOptimizationBundle:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = request.n_trials
        objective_name = request.objective
        minimize = objective_name == "max_drawdown_pct"

        all_candidates: list[MultiPairCandidate] = []
        start_time = time.time()

        print(f"\n[Bayesian] {n_trials} trials via Optuna TPE  (SL {request.sl_min}\u2013{request.sl_max}, TP {request.tp_min}\u2013{request.tp_max}, {len(self.pair_data)} pairs)")

        def objective(trial: optuna.Trial) -> float:
            sl_value = trial.suggest_float("sl_pct", request.sl_min, request.sl_max, step=0.01)
            tp_value = trial.suggest_float("tp_pct", request.tp_min, request.tp_max, step=0.01)
            candidate = self._evaluate_candidate(request.mode, sl_value, tp_value, objective_name)
            all_candidates.append(candidate)
            _print_progress(len(all_candidates), n_trials, start_time, "Bayesian")
            return _multi_objective_value(candidate, objective_name)

        direction = "minimize" if minimize else "maximize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))

        # Recompute aggregate objective for ranking after the study
        study.optimize(objective, n_trials=n_trials)

        ranked = _rank_multi_candidates(all_candidates, objective_name)
        top_results = ranked[: request.top_n]
        pairs = [(sym, ex) for sym, ex, _, _, _ in self.pair_data]
        bundle = MultiPairOptimizationBundle(
            request=request,
            pairs=pairs,
            results=top_results,
            coarse_results=[],
            output_path=output_path,
        )
        _write_multi_output(bundle)
        return bundle

    def _evaluate_grid(
        self,
        mode: Mode,
        objective: Objective,
        sl_values: list[float],
        tp_values: list[float],
        phase: str = "Grid",
    ) -> list[MultiPairCandidate]:
        candidates: list[MultiPairCandidate] = []
        total = len(sl_values) * len(tp_values)
        start_time = time.time()
        count = 0
        for sl_value in sl_values:
            for tp_value in tp_values:
                candidate = self._evaluate_candidate(mode, sl_value, tp_value, objective)
                candidates.append(candidate)
                count += 1
                _print_progress(count, total, start_time, phase)
        return _rank_multi_candidates(candidates, objective)


# ------------------------------------------------------------------
# Stateless helpers
# ------------------------------------------------------------------

def _write_output(bundle: OptimizationBundle) -> None:
    bundle.output_path.write_text(
        json.dumps(bundle.to_dict(), indent=2),
        encoding="utf-8",
    )


def _deduplicate_and_rank(results: list[BacktestMetrics], objective: Objective) -> list[BacktestMetrics]:
    deduped: dict[tuple[float, float, str], BacktestMetrics] = {}
    for result in results:
        key = (result.sl_pct, result.tp_pct, result.mode)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = result
            continue
        if objective == "max_drawdown_pct":
            if result.max_drawdown_pct < existing.max_drawdown_pct:
                deduped[key] = result
        elif _objective_value(result, objective) > _objective_value(existing, objective):
            deduped[key] = result
    return _rank_results(list(deduped.values()), objective)


def _rank_results(results: list[BacktestMetrics], objective: Objective) -> list[BacktestMetrics]:
    reverse = objective != "max_drawdown_pct"
    ranked = sorted(results, key=lambda item: _objective_value(item, objective), reverse=reverse)
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def _objective_value(result: BacktestMetrics, objective: Objective) -> float:
    if objective == "net_profit_pct":
        return result.net_profit_pct
    if objective == "profit_factor":
        return result.profit_factor
    if objective == "win_rate_pct":
        return result.win_rate_pct
    if objective == "trade_count":
        return float(result.trade_count)
    return result.max_drawdown_pct


_GRID_POINTS = 29


def _auto_step(low: float, high: float, points: int = _GRID_POINTS) -> float:
    """Derive a step size that divides [low, high] into *points* intervals."""
    span = high - low
    if span <= 0 or points < 2:
        return max(span, 0.0001)
    return round(span / (points - 1), 8)


def _build_range(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be greater than zero.")
    values: list[float] = []
    current = start
    while current <= end + (step / 10.0):
        values.append(round(current, 8))
        current += step
    return values


def _refine_range(center: float, lower_bound: float, upper_bound: float, coarse_step: float, fine_step: float) -> list[float]:
    start = max(lower_bound, center - coarse_step)
    end = min(upper_bound, center + coarse_step)
    return _build_range(start, end, fine_step)


# ------------------------------------------------------------------
# Multi-pair helpers
# ------------------------------------------------------------------

def _build_candidate(
    sl_value: float,
    tp_value: float,
    per_pair: list[BacktestMetrics],
    request_objective: Objective | None,
) -> MultiPairCandidate:
    """Aggregate per-pair metrics into a single candidate."""
    n = len(per_pair)
    avg_net = sum(m.net_profit_pct for m in per_pair) / n
    worst_dd = max(m.max_drawdown_pct for m in per_pair)

    # Compute aggregate objective value
    if request_objective == "max_drawdown_pct":
        agg_obj = worst_dd
    elif request_objective == "net_profit_pct":
        agg_obj = avg_net
    elif request_objective == "profit_factor":
        agg_obj = sum(m.profit_factor for m in per_pair) / n
    elif request_objective == "win_rate_pct":
        agg_obj = sum(m.win_rate_pct for m in per_pair) / n
    elif request_objective == "trade_count":
        agg_obj = sum(m.trade_count for m in per_pair) / n
    else:
        agg_obj = avg_net

    return MultiPairCandidate(
        sl_pct=sl_value,
        tp_pct=tp_value,
        aggregate_net_profit_pct=avg_net,
        aggregate_max_drawdown_pct=worst_dd,
        aggregate_objective=agg_obj,
        per_pair_metrics=per_pair,
    )


def _write_multi_output(bundle: MultiPairOptimizationBundle) -> None:
    bundle.output_path.write_text(
        json.dumps(bundle.to_dict(), indent=2),
        encoding="utf-8",
    )


def _multi_objective_value(candidate: MultiPairCandidate, objective: Objective) -> float:
    """Return the sortable objective value for a multi-pair candidate."""
    if objective == "max_drawdown_pct":
        return candidate.aggregate_max_drawdown_pct
    return candidate.aggregate_objective


def _rank_multi_candidates(
    candidates: list[MultiPairCandidate],
    objective: Objective,
) -> list[MultiPairCandidate]:
    reverse = objective != "max_drawdown_pct"
    ranked = sorted(
        candidates,
        key=lambda c: _multi_objective_value(c, objective),
        reverse=reverse,
    )
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def _deduplicate_and_rank_multi(
    candidates: list[MultiPairCandidate],
    objective: Objective,
) -> list[MultiPairCandidate]:
    deduped: dict[tuple[float, float], MultiPairCandidate] = {}
    for candidate in candidates:
        key = (candidate.sl_pct, candidate.tp_pct)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = candidate
            continue
        if objective == "max_drawdown_pct":
            if candidate.aggregate_max_drawdown_pct < existing.aggregate_max_drawdown_pct:
                deduped[key] = candidate
        elif _multi_objective_value(candidate, objective) > _multi_objective_value(existing, objective):
            deduped[key] = candidate
    return _rank_multi_candidates(list(deduped.values()), objective)
