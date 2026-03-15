from __future__ import annotations

import argparse
import time
import logging

from .._utils import _format_time, build_risk
from ..models import BacktestMetrics, CandleRequest
from ..backtest.engine import TradingViewLikeBacktester
from ._helpers import _resolve, load_candles, generate_signals, resolve_pairlist, print_summary_table
from strategy import list_strategies


def _run_single_backtest(
    symbol: str,
    exchange: str,
    timeframe: str,
    session: str,
    adjustment: str,
    strategy_name: str,
    initial_capital: float,
    data_dir: str,
    mode: str,
    sl: float,
    tp: float,
    start: str | None,
    end: str | None,
    *,
    pair_index: int = 1,
    pair_total: int = 1,
) -> BacktestMetrics | None:
    
    pair_label = f"{exchange}:{symbol}"
    tag = f"[{pair_index}/{pair_total}] {pair_label} "
    dot_width = max(40 - len(tag), 3)
    print(f"\n  {tag}{'.' * dot_width} ", end="", flush=True)

    candle_request = CandleRequest(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start=start,
        end=end,
        session=session,
        adjustment=adjustment,
    )

    t_total = time.time()

    try:
        # 1. Load data
        candles = load_candles(candle_request, data_dir, step="1/3", quiet=True)
        if candles is None or len(candles) == 0:
            print("skipped (no candle data)")
            return None

        # 2. Generate signals
        signal_frame, _ = generate_signals(
            strategy_name, candles, mode, start, end, step="2/3", quiet=True
        )
        if signal_frame is None or len(signal_frame) == 0:
            print("skipped (no signals generated)")
            return None

        # 3. Run backtest
        risk = build_risk(mode, sl, tp)
        backtester = TradingViewLikeBacktester(
            candle_request=candle_request, 
            initial_equity=initial_capital
        )
        
        result = backtester.run(signal_frame, risk, mode)
        elapsed = _format_time(time.time() - t_total)

        print(f"done ({elapsed})")
        return result.metrics

    except Exception as e:
        print(f"failed ({e})")
        logging.exception(f"Error during backtest of {pair_label}")
        return None


def run_backtest(args: argparse.Namespace, config: dict) -> int:
    strategy_name = _resolve(args, config, "strategy")
    available_strategies = list_strategies()
    
    if not strategy_name or strategy_name not in available_strategies:
        print("Error: Invalid or no strategy specified.")
        print(f"Available strategies: {', '.join(available_strategies) or '(none)'}")
        return 1

    timeframe = _resolve(args, config, "timeframe")
    session = _resolve(args, config, "session")
    adjustment = args.adjustment
    initial_capital = config.get("initial_capital", 1000.0)
    data_dir = config.get("data_dir", "./data")

    pairs = resolve_pairlist(args, config)
    if not pairs:
        print("Error: No trading pairs resolved. Please check your config or arguments.")
        return 1

    print(f"\n{'#' * 60}")
    print(f"  Running backtest for {len(pairs)} pair(s)")
    print(f"  Strategy: {strategy_name} | SL={args.sl}% TP={args.tp}%")
    print(f"{'#' * 60}")

    all_metrics: list[tuple[str, str, BacktestMetrics]] = []
    rc = 0
    
    for idx, (symbol, pair_exchange) in enumerate(pairs, 1):
        m = _run_single_backtest(
            symbol=symbol,
            exchange=pair_exchange,
            timeframe=timeframe,
            session=session,
            adjustment=adjustment,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            data_dir=data_dir,
            mode=args.mode,
            sl=args.sl,
            tp=args.tp,
            start=args.start,
            end=args.end,
            pair_index=idx,
            pair_total=len(pairs),
        )
        if m is None:
            rc = 1
        else:
            all_metrics.append((symbol, pair_exchange, m))

    print_summary_table(all_metrics)

    return rc