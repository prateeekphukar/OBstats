"""Run full backtest and print detailed performance report.

Supports two data sources:
  --source synthetic   (default) Generate synthetic data in-memory
  --source db          Load from backtesting database (data/backtest.db)

Examples:
  python run_backtest.py
  python run_backtest.py --source db --symbol NIFTY
  python run_backtest.py --source db --symbol RELIANCE --start 2026-01-01
"""
import sys, os, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.metrics import PerformanceMetrics
import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Options HFT Bot — Backtest Runner")
    parser.add_argument(
        "--source", choices=["synthetic", "db"], default="synthetic",
        help="Data source: 'synthetic' (generated) or 'db' (backtest.db)"
    )
    parser.add_argument("--symbol", default="NIFTY", help="Symbol to backtest (default: NIFTY)")
    parser.add_argument("--start", default=None, help="Start date (ISO format, e.g. 2025-12-01)")
    parser.add_argument("--end", default=None, help="End date (ISO format, e.g. 2026-02-22)")
    parser.add_argument("--db-path", default="data/backtest.db", help="Path to backtest database")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to load from DB")
    parser.add_argument("--bars", type=int, default=2000, help="Number of bars for synthetic data")
    return parser.parse_args()


def main():
    args = parse_args()
    loader = HistoricalDataLoader()

    print("=" * 60)
    print("  OPTIONS HFT BOT — BACKTEST")
    print("=" * 60)
    print("")

    # ── Load data ──
    if args.source == "db":
        print(f"[1/4] Loading from database: {args.db_path}")
        print(f"      Symbol: {args.symbol}")
        if args.start:
            print(f"      Start:  {args.start}")
        if args.end:
            print(f"      End:    {args.end}")

        # Show database info
        info = loader.get_db_info(args.db_path)
        if "error" in info:
            print(f"\n  ERROR: {info['error']}")
            print(f"  Run: python -m src.backtest.build_db")
            return

        print(f"      DB has {info.get('option_rows', '?'):,} rows across "
              f"{info.get('symbols', '?')} symbols")

        data = loader.load_from_db(
            db_path=args.db_path,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            limit=args.limit,
        )
        print(f"      -> Loaded {len(data):,} rows")
    else:
        print("[1/4] Generating synthetic market data...")
        data = loader.generate_sample_data(
            symbol=args.symbol, spot=22000,
            num_bars=args.bars, num_strikes=10, num_expiries=2,
        )
        print(f"      -> {len(data):,} rows | {args.bars:,} bars | 10 strikes | 2 expiries")

    print("")

    if len(data) == 0:
        print("  ERROR: No data loaded. Check your filters.")
        return

    # ── Load config ──
    import yaml
    with open("config/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)

    strategy_config = settings.get("strategy", {})
    strategy_config.update({
        "risk_free_rate": settings.get("pricing", {}).get("risk_free_rate", 0.07),
        "dividend_yield": settings.get("pricing", {}).get("dividend_yield", 0.012),
    })

    risk_config = settings.get("risk", {})

    bt_config = BacktestConfig(
        slippage_bps=5.0,
        commission_per_contract=20.0,   # INR per lot for Indian markets
        initial_capital=12_000,         # 12,000 INR
        max_signals_per_bar=5,
    )

    # ── Run backtest ──
    print("[2/4] Running backtest engine...")
    engine = BacktestEngine(strategy_config, risk_config, bt_config)
    result = engine.run(data)
    print(f"      -> Complete!")
    print("")

    # ── Print report ──
    print("[3/4] Computing performance metrics...")
    metrics = result["metrics"]

    if "error" in metrics:
        print(f"      ERROR: {metrics['error']}")
    else:
        report = PerformanceMetrics.format_report(metrics)
        print(report)

    # ── Trade summary ──
    print("")
    print("[4/4] Trade Summary")
    print("-" * 60)
    trades = result["trades"]
    signals = result["signals"]
    print(f"  Total signals generated:   {len(signals):>8,}")
    print(f"  Total trades executed:     {len(trades):>8,}")

    if trades:
        from collections import Counter
        type_counts = Counter(t["signal_type"] for t in trades)
        print(f"\n  Signal Type Breakdown:")
        for sig_type, count in type_counts.most_common():
            print(f"    {sig_type:<20} {count:>6}")

        side_counts = Counter(t["side"] for t in trades)
        print(f"\n  Side Breakdown:")
        for side, count in side_counts.most_common():
            pnl_for_side = sum(t.get("pnl", 0) for t in trades if t["side"] == side)
            print(f"    {side:<20} {count:>6} trades | P&L: ${pnl_for_side:>12,.2f}")

        pnls = [t.get("pnl", 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        print(f"\n  Winning trades:  {len(wins):>6} | Avg: ${sum(wins)/len(wins) if wins else 0:>10,.2f}")
        print(f"  Losing trades:   {len(losses):>6} | Avg: ${sum(losses)/len(losses) if losses else 0:>10,.2f}")

    print(f"\n{'=' * 60}")
    print(f"  Backtest complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
