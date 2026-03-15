from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

import pandas as pd

from .._utils import _format_time
from ..downloader import TradingViewDataClient
from strategy import get_strategy

if TYPE_CHECKING:
    from ..models import CandleRequest


def _resolve(args: argparse.Namespace, config: dict, key: str, default=None):
    """Return CLI override if set, else config value, else default."""
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def parse_pair(entry: str) -> tuple[str, str]:
    """Parse 'EXCHANGE:SYMBOL' into (symbol, exchange)."""
    if ":" not in entry:
        raise SystemExit(
            f"Error: Invalid pair format '{entry}'. "
            f"Expected 'EXCHANGE:SYMBOL' (e.g. 'NASDAQ:AAPL')."
        )
    exchange, symbol = entry.split(":", 1)
    exchange, symbol = exchange.strip(), symbol.strip()
    if not exchange or not symbol:
        raise SystemExit(f"Error: Invalid pair format '{entry}'. Expected 'EXCHANGE:SYMBOL'.")
    return symbol, exchange


def resolve_pairlist(args: argparse.Namespace, config: dict) -> list[tuple[str, str]]:
    """Return list of (symbol, exchange) tuples from --pairs, --symbol, or config pairlist."""
    # Explicit --pairs (download-data)
    pairs_arg = getattr(args, "pairs", None)
    if pairs_arg:
        return [parse_pair(p) for p in pairs_arg]
    # Explicit --symbol (backtest / hyperopt)
    symbol = getattr(args, "symbol", None)
    if symbol:
        return [parse_pair(symbol)]
    # Fall back to config pairlist
    entries = config.get("pairlist", [])
    if not entries:
        raise SystemExit(
            "Error: No pairs specified. Either use --pairs / --symbol on the CLI,\n"
            "or define a 'pairlist' in your config file."
        )
    return [parse_pair(e) for e in entries]


def load_candles(
    candle_request: "CandleRequest",
    data_dir: str,
    *,
    step: str = "",
    quiet: bool = False,
) -> pd.DataFrame:
    """Load cached candle data and print progress."""
    if not quiet:
        print(f"\n[{step}] Loading candle data...")
    t0 = time.time()
    client = TradingViewDataClient(cache_dir=data_dir)
    candles = client.get_history(candle_request, cache_only=True)
    if not quiet:
        print(f"      {len(candles)} candles loaded ({_format_time(time.time() - t0)})")
    return candles


def generate_signals(
    strategy_name: str,
    candles: pd.DataFrame,
    mode: str,
    start: str | None,
    end: str | None,
    *,
    step: str = "",
    quiet: bool = False,
) -> tuple[pd.DataFrame, "get_strategy"]:
    """Instantiate strategy, generate signals, and print progress."""
    if not quiet:
        print(f"\n[{step}] Generating signals...")
    t0 = time.time()
    strategy = get_strategy(strategy_name)
    signal_settings = {
        "enable_long": mode in {"long", "both"},
        "enable_short": mode in {"short", "both"},
        "start": start,
        "end": end,
    }
    signal_frame = strategy.generate_signals(candles, signal_settings)
    buy_count = int(signal_frame["buy_signal"].sum())
    sell_count = int(signal_frame["sell_signal"].sum())
    if not quiet:
        print(f"      {buy_count} buy / {sell_count} sell signals ({_format_time(time.time() - t0)})")
    return signal_frame, strategy


# ---------------------------------------------------------------------------
# Shared summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    all_metrics: list[tuple[str, str, "BacktestMetrics"]],
    *,
    title: str = "Backtest Summary",
) -> None:
    """Print a tabular summary of all pair results plus an aggregate row.

    Re-used by both the backtest and hyperopt CLI commands so that output
    formatting stays consistent.
    """
    from ..models import BacktestMetrics  # noqa: F811 – lazy to avoid circular

    if not all_metrics:
        return

    # Dynamically size the Pair column
    pair_col_name = "Pair"
    max_pair = max(len(f"{ex}:{sym}") for sym, ex, _ in all_metrics)
    max_pair = max(max_pair, len(pair_col_name))

    # Reusable format string for perfect column alignment
    row_fmt = (
        "  {idx:>3}  {pair:<{max_pair}}  {net:>+8.2f}%  {dd:>8.2f}%  "
        "{sharpe:>7.2f}  {calmar:>7.2f}  {wr:>9.2f}%  {pf:>9}  "
        "{exp:>+8.2f}%  {avg_w:>+9.2f}%  {avg_l:>+10.2f}%  {worst:>+11.2f}%  "
        "{mcl:>11}  {trades:>7}  {sl_tp_sig:>15}  ${eq:>13,.2f}"
    )

    hdr = (
        f"  {'':>3}  {pair_col_name:<{max_pair}}  {'Return %':>9}  {'Max DD %':>9}  "
        f"{'Sharpe':>7}  {'Calmar':>7}  {'Win Rate %':>10}  {'Prof Fact':>9}  "
        f"{'Expect %':>9}  {'Avg Win %':>10}  {'Avg Loss %':>11}  {'Worst Trade':>12}  "
        f"{'Loss Streak':>11}  {'Trades':>7}  {'Exits(SL/TP/Sg)':>15}  {'Final Equity':>14}"
    )
    sep = "  " + "-" * (len(hdr) - 2)

    print(f"\n{'=' * len(hdr)}")
    print(f"  {title}")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)

    # Accumulators for aggregate row
    total_net, total_trades, worst_dd, total_equity = 0.0, 0, 0.0, 0.0
    sharpe_sum, calmar_sum, exp_sum = 0.0, 0.0, 0.0
    avg_w_sum, avg_l_sum = 0.0, 0.0
    worst_trade_all = 0.0
    max_consec_all = 0

    # Back-calculate raw counts for aggregate percentages
    total_wins, total_sl, total_tp, total_sig = 0, 0.0, 0.0, 0.0

    for i, (symbol, exchange, m) in enumerate(all_metrics, 1):
        pair = f"{exchange}:{symbol}"
        sl_tp_sig = f"{m.sl_exit_pct:.0f}/{m.tp_exit_pct:.0f}/{m.signal_exit_pct:.0f}"

        print(row_fmt.format(
            idx=i, pair=pair, max_pair=max_pair, net=m.net_profit_pct, dd=m.max_drawdown_pct,
            sharpe=m.sharpe_ratio, calmar=m.calmar_ratio, wr=m.win_rate_pct, pf=f"{m.profit_factor:.2f}",
            exp=m.expectancy_pct, avg_w=m.avg_win_pct, avg_l=m.avg_loss_pct, worst=m.worst_trade_pct,
            mcl=m.max_consec_losses, trades=m.trade_count, sl_tp_sig=sl_tp_sig, eq=m.equity_final
        ))

        # Accumulate metrics
        total_net += m.net_profit_pct
        total_trades += m.trade_count
        total_equity += m.equity_final
        sharpe_sum += m.sharpe_ratio
        calmar_sum += m.calmar_ratio
        exp_sum += m.expectancy_pct
        avg_w_sum += m.avg_win_pct
        avg_l_sum += m.avg_loss_pct

        worst_dd = max(worst_dd, m.max_drawdown_pct)
        worst_trade_all = min(worst_trade_all, m.worst_trade_pct)
        max_consec_all = max(max_consec_all, m.max_consec_losses)

        if m.trade_count > 0:
            total_wins += round(m.win_rate_pct / 100 * m.trade_count)
            total_sl += m.sl_exit_pct / 100 * m.trade_count
            total_tp += m.tp_exit_pct / 100 * m.trade_count
            total_sig += m.signal_exit_pct / 100 * m.trade_count

    print(sep)

    # Print TOTAL row if there are multiple pairs
    n = len(all_metrics)
    if n > 1:
        avg_net = total_net / n
        avg_win = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
        avg_sharpe = sharpe_sum / n
        avg_calmar = calmar_sum / n
        avg_expect = exp_sum / n
        avg_w = avg_w_sum / n
        avg_l = avg_l_sum / n

        agg_sl = (total_sl / total_trades * 100) if total_trades > 0 else 0.0
        agg_tp = (total_tp / total_trades * 100) if total_trades > 0 else 0.0
        agg_sig = (total_sig / total_trades * 100) if total_trades > 0 else 0.0
        sl_tp_sig_agg = f"{agg_sl:.0f}/{agg_tp:.0f}/{agg_sig:.0f}"

        print(row_fmt.format(
            idx="", pair="TOTAL / AVG", max_pair=max_pair, net=avg_net, dd=worst_dd,
            sharpe=avg_sharpe, calmar=avg_calmar, wr=avg_win, pf="",
            exp=avg_expect, avg_w=avg_w, avg_l=avg_l, worst=worst_trade_all,
            mcl=max_consec_all, trades=total_trades, sl_tp_sig=sl_tp_sig_agg, eq=total_equity
        ))

    print(f"{'=' * len(hdr)}\n")
