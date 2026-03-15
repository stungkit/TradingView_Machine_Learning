from __future__ import annotations

import argparse
import time
import logging
from pathlib import Path
from typing import Any

from .._utils import _format_time
from ..models import CandleRequest, OptimizationRequest
from ._helpers import _resolve, load_candles, generate_signals, resolve_pairlist, print_summary_table
from strategy import list_strategies

# Moved inline imports to the top level for cleaner architecture
from ..hyperopt import run_optimization, run_multi_optimization


def _print_signal_density(signal_frame: Any, pair_label: str = "") -> None:
    """Display signal density info for a signal frame."""
    buy_count = int(signal_frame["buy_signal"].sum())
    sell_count = int(signal_frame["sell_signal"].sum())
    total_signals = buy_count + sell_count
    in_range_bars = int(signal_frame["in_date_range"].sum())
    
    # Safely avoid division by zero
    density_per_1k = (total_signals / max(in_range_bars, 1)) * 1000
    
    if density_per_1k < 5:
        density_label = "low"
    elif density_per_1k < 20:
        density_label = "medium"
    else:
        density_label = "high"
        
    prefix = f"  {pair_label}: " if pair_label else "      "
    print(
        f"{prefix}Signal density: {density_label} ({density_per_1k:.2f}/1k bars) "
        f"over {in_range_bars} in-range bars"
    )


def _run_single_hyperopt(
    symbol: str,
    exchange: str,
    timeframe: str,
    session_type: str,
    adjustment: str,
    strategy_name: str,
    initial_capital: float,
    data_dir: str,
    output_dir_base: str,
    mode: str,
    start: str | None,
    end: str | None,
    sl_min: float,
    sl_max: float,
    tp_min: float,
    tp_max: float,
    objective: str,
    top_n: int,
    search_method: str,
    n_trials: int,
) -> int:
    
    candle_request = CandleRequest(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start=start,
        end=end,
        session=session_type,
        adjustment=adjustment,
    )

    method_label = "Bayesian (Optuna TPE)" if search_method == "bayesian" else "Two-stage grid"

    print(f"\n{'=' * 60}")
    print("  HyperView - Hyper-Optimization")
    print(f"  {exchange}:{symbol} | {timeframe} | mode={mode}")
    print(f"  Strategy: {strategy_name}")
    print(f"  SL range: {sl_min}-{sl_max}")
    print(f"  TP range: {tp_min}-{tp_max}")
    print(f"  Search:   {method_label}")
    if search_method == "bayesian":
        print(f"  Trials:   {n_trials}")
    print(f"{'=' * 60}")

    try:
        # 1. Load data
        candles = load_candles(candle_request, data_dir, step="1/4")
        if candles is None or len(candles) == 0:
            print("Error: No candle data found. Optimization aborted.")
            return 1

        # 2. Generate signals
        signal_frame, strategy = generate_signals(
            strategy_name, candles, mode, start, end, step="2/4",
        )
        if signal_frame is None or len(signal_frame) == 0:
            print("Error: No signals generated. Optimization aborted.")
            return 1

        _print_signal_density(signal_frame)

        # 3. Run optimization
        print("\n[3/4] Running optimization...")
        t_opt = time.time()
        request = OptimizationRequest(
            candle_request=candle_request,
            mode=mode,
            objective=objective,
            sl_min=sl_min,
            sl_max=sl_max,
            tp_min=tp_min,
            tp_max=tp_max,
            top_n=top_n,
            search_method=search_method,
            n_trials=n_trials,
            initial_equity=initial_capital,
        )
        
        output_path = Path(output_dir_base) / f"{strategy_name}_{exchange}_{symbol}_{timeframe}_{mode}.json"
        
        bundle = run_optimization(
            signal_frame=signal_frame,
            candle_request=candle_request,
            strategy=strategy,
            request=request,
            output_path=output_path,
            initial_equity=initial_capital,
        )
        print(f"      Optimization complete ({_format_time(time.time() - t_opt)})")

        # 4. Display results using the shared summary table
        print("\n[4/4] Results")
        for rank, metrics in enumerate(bundle.results, start=1):
            table_rows = [(symbol, exchange, metrics)]
            print_summary_table(
                table_rows,
                title=f"Rank #{rank}  \u2014  SL={metrics.sl_pct:.4f}%  TP={metrics.tp_pct:.4f}%",
            )

        print(f"Results written to {output_path}")
        return 0

    except Exception as e:
        print(f"\nError: Optimization failed ({e})")
        logging.exception(f"Exception during single hyperopt for {exchange}:{symbol}")
        return 1


def _run_multi_hyperopt(
    pairs: list[tuple[str, str]],
    timeframe: str,
    session_type: str,
    adjustment: str,
    strategy_name: str,
    initial_capital: float,
    data_dir: str,
    output_dir_base: str,
    mode: str,
    start: str | None,
    end: str | None,
    sl_min: float,
    sl_max: float,
    tp_min: float,
    tp_max: float,
    objective: str,
    top_n: int,
    search_method: str,
    n_trials: int,
) -> int:
    
    pair_labels = ", ".join(f"{ex}:{sym}" for sym, ex in pairs)
    method_label = "Bayesian (Optuna TPE)" if search_method == "bayesian" else "Two-stage grid"

    print(f"\n{'=' * 60}")
    print("  HyperView - Multi-Pair Hyper-Optimization")
    print(f"  Pairs: {pair_labels}")
    print(f"  {timeframe} | mode={mode}")
    print(f"  Strategy: {strategy_name}")
    print(f"  SL range: {sl_min}-{sl_max}")
    print(f"  TP range: {tp_min}-{tp_max}")
    print(f"  Search:   {method_label}")
    if search_method == "bayesian":
        print(f"  Trials:   {n_trials}")
    print(f"{'=' * 60}")

    # 1. Load candles and generate signals for each pair
    pair_data_list = []
    for idx, (symbol, exchange) in enumerate(pairs, 1):
        pair_label = f"{exchange}:{symbol}"
        step_load = f"1/{len(pairs) + 2} [{idx}/{len(pairs)}]"
        
        candle_request = CandleRequest(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start=start,
            end=end,
            session=session_type,
            adjustment=adjustment,
        )
        
        print(f"\n  Loading {pair_label}...")
        candles = load_candles(candle_request, data_dir, step=step_load, quiet=True)
        if candles is None or len(candles) == 0:
            print(f"  Skipping {pair_label} (no candle data)")
            continue
            
        signal_frame, strategy = generate_signals(
            strategy_name, candles, mode, start, end, step=step_load, quiet=True,
        )
        if signal_frame is None or len(signal_frame) == 0:
            print(f"  Skipping {pair_label} (no signals generated)")
            continue
            
        _print_signal_density(signal_frame, pair_label)
        pair_data_list.append((symbol, exchange, signal_frame, strategy, candle_request))

    if not pair_data_list:
        print("Error: No valid pairs to optimize (no data/signals).")
        return 1

    try:
        # Use the first valid pair's candle_request as a template
        _, _, _, _, template_cr = pair_data_list[0]

        # 2. Run multi-pair optimization
        print(f"\n  Running optimization across {len(pair_data_list)} pairs...")
        t_opt = time.time()
        request = OptimizationRequest(
            candle_request=template_cr,
            mode=mode,
            objective=objective,
            sl_min=sl_min,
            sl_max=sl_max,
            tp_min=tp_min,
            tp_max=tp_max,
            top_n=top_n,
            search_method=search_method,
            n_trials=n_trials,
            initial_equity=initial_capital,
        )
        
        output_path = Path(output_dir_base) / f"{strategy_name}_{timeframe}_{mode}_multi.json"
        
        bundle = run_multi_optimization(
            pair_data=pair_data_list,
            request=request,
            output_path=output_path,
            initial_equity=initial_capital,
        )
        print(f"      Optimization complete ({_format_time(time.time() - t_opt)})")

        # 3. Display results — show top 5 in descending order (best last)
        display_candidates = bundle.results[:5]
        print(f"\n  Top {len(display_candidates)} candidates across {len(pair_data_list)} pairs:")

        for candidate in reversed(display_candidates):
            # Build per-pair rows for the shared summary table
            table_rows = []
            for (sym, ex, _, _, _), m in zip(pair_data_list, candidate.per_pair_metrics):
                table_rows.append((sym, ex, m))

            print_summary_table(
                table_rows,
                title=(
                    f"Rank #{candidate.rank}  \u2014  SL={candidate.sl_pct:.4f}%  TP={candidate.tp_pct:.4f}%  "
                    f"(avg net={candidate.aggregate_net_profit_pct:+.2f}%  worst dd={candidate.aggregate_max_drawdown_pct:.2f}%)"
                ),
            )

        print(f"Results written to {output_path}")
        return 0

    except Exception as e:
        print(f"\nError: Multi-pair optimization failed ({e})")
        logging.exception("Exception during multi-pair hyperopt")
        return 1


def run_hyperopt(args: argparse.Namespace, config: dict) -> int:
    # Validate strategy early
    strategy_name = _resolve(args, config, "strategy")
    available_strategies = list_strategies()
    
    if not strategy_name or strategy_name not in available_strategies:
        print("Error: Invalid or no strategy specified.")
        print(f"Available strategies: {', '.join(available_strategies) or '(none)'}")
        return 1

    timeframe = _resolve(args, config, "timeframe")
    session_type = _resolve(args, config, "session")
    adjustment = args.adjustment
    initial_capital = config.get("initial_capital", 1000.0)
    data_dir = config.get("data_dir", "./data")
    
    # Safely create the output directory if it doesn't exist
    output_dir_base = config.get("output_dir", "./outputs")
    Path(output_dir_base).mkdir(parents=True, exist_ok=True)

    opt_cfg = config.get("optimization", {})
    sl_range = opt_cfg.get("sl_range", {})
    tp_range = opt_cfg.get("tp_range", {})
    
    # Safely unpack optimization bounds
    sl_min = args.sl_min if args.sl_min is not None else sl_range.get("min", 0.5)
    sl_max = args.sl_max if args.sl_max is not None else sl_range.get("max", 5.0)
    tp_min = args.tp_min if args.tp_min is not None else tp_range.get("min", 1.0)
    tp_max = args.tp_max if args.tp_max is not None else tp_range.get("max", 10.0)
    
    objective = args.objective or opt_cfg.get("objective", "net_profit")
    top_n = args.top_n if args.top_n is not None else opt_cfg.get("top_n", 5)
    search_method = args.search_method or opt_cfg.get("search_method", "grid")
    n_trials = args.n_trials if args.n_trials is not None else opt_cfg.get("n_trials", 100)
    
    symbols = resolve_pairlist(args, config)
    if not symbols:
        print("Error: No trading pairs resolved. Please check your config or arguments.")
        return 1

    # Multi-pair path: optimize across all pairs simultaneously
    if len(symbols) > 1:
        return _run_multi_hyperopt(
            pairs=symbols,
            timeframe=timeframe,
            session_type=session_type,
            adjustment=adjustment,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            data_dir=data_dir,
            output_dir_base=output_dir_base,
            mode=args.mode,
            start=args.start,
            end=args.end,
            sl_min=sl_min,
            sl_max=sl_max,
            tp_min=tp_min,
            tp_max=tp_max,
            objective=objective,
            top_n=top_n,
            search_method=search_method,
            n_trials=n_trials,
        )

    # Single-pair path
    symbol, pair_exchange = symbols[0]
    return _run_single_hyperopt(
        symbol=symbol,
        exchange=pair_exchange,
        timeframe=timeframe,
        session_type=session_type,
        adjustment=adjustment,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        data_dir=data_dir,
        output_dir_base=output_dir_base,
        mode=args.mode,
        start=args.start,
        end=args.end,
        sl_min=sl_min,
        sl_max=sl_max,
        tp_min=tp_min,
        tp_max=tp_max,
        objective=objective,
        top_n=top_n,
        search_method=search_method,
        n_trials=n_trials,
    )