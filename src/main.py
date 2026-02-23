"""
Options HFT Bot — Main Entry Point.

Wires all components together and runs the trading bot
or backtesting engine based on configuration.

Supports brokers: Dhan, Interactive Brokers, Paper (simulated).
"""

import asyncio
import logging
import sys
import os
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.chain import OptionsChain
from src.data.feed import DataFeed, FeedConfig
from src.data.storage import TickStorage
from src.pricing.black_scholes import BlackScholesPricer
from src.pricing.iv_solver import ImpliedVolSolver
from src.pricing.vol_surface import VolSurface
from src.strategy.signals import SignalGenerator
from src.strategy.filters import SignalFilter
from src.risk.portfolio import Portfolio
from src.risk.manager import RiskManager
from src.risk.kill_switch import KillSwitch
from src.execution.broker import IBBroker, PaperBroker
from src.execution.dhan_broker import DhanBroker
from src.execution.oms import OrderManager
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.data_loader import HistoricalDataLoader
from src.backtest.metrics import PerformanceMetrics


def load_config(config_dir: str = "config") -> tuple[dict, dict]:
    """Load strategy and risk configuration files."""
    settings_path = os.path.join(config_dir, "settings.yaml")
    risk_path = os.path.join(config_dir, "risk_limits.yaml")

    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    with open(risk_path, "r") as f:
        risk_config = yaml.safe_load(f)

    return settings, risk_config


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def create_broker(broker_config: dict, logger):
    """
    Create the appropriate broker based on config.

    Supports: 'dhan', 'interactive_brokers', 'paper'
    """
    broker_name = broker_config.get("name", "paper").lower()

    if broker_name == "dhan":
        dhan_config = broker_config.get("dhan", {})
        if not dhan_config.get("client_id") or not dhan_config.get("access_token"):
            logger.warning(
                "Dhan client_id/access_token not set in config. "
                "Set them in config/settings.yaml → broker → dhan"
            )
        logger.info("Using DHAN broker (NSE/BSE India)")
        return DhanBroker(dhan_config)

    elif broker_name == "interactive_brokers":
        ib_config = broker_config.get("ib", {})
        logger.info(f"Using INTERACTIVE BROKERS (port {ib_config.get('port', 7497)})")
        return IBBroker(ib_config)

    else:
        logger.info("Using PAPER TRADING broker (simulated fills)")
        return PaperBroker()


async def create_data_feed(settings: dict, chains: dict, on_tick, on_underlying, logger):
    """
    Create the appropriate data feed based on config.

    Uses DhanDataFeed for Dhan, or generic WebSocket DataFeed otherwise.
    """
    feed_config = settings.get("data_feed", {})
    broker_config = settings.get("broker", {})
    provider = feed_config.get("provider", broker_config.get("name", "paper")).lower()

    if provider == "dhan":
        from src.data.dhan_feed import DhanDataFeed, DhanFeedConfig

        dhan_config = broker_config.get("dhan", {})
        symbols = feed_config.get("symbols", ["NIFTY"])
        underlyings = feed_config.get("dhan_underlyings", {})

        # Build instrument list from option chain
        instruments = []
        dhan_broker = DhanBroker(dhan_config)
        connected = await dhan_broker.connect()

        if connected:
            for symbol in symbols:
                sec_id = underlyings.get(symbol)
                if not sec_id:
                    continue

                # Fetch expiry list
                expiries = await dhan_broker.get_expiry_list(sec_id, "IDX_I")
                if expiries and isinstance(expiries, list):
                    # Take nearest expiry
                    nearest_expiry = expiries[0] if isinstance(expiries[0], str) else ""
                    if nearest_expiry:
                        # Fetch option chain for this expiry
                        chain_data = await dhan_broker.get_option_chain(sec_id, "IDX_I", nearest_expiry)
                        if chain_data and isinstance(chain_data, dict):
                            for entry in chain_data.get("data", []):
                                instruments.append({
                                    "security_id": str(entry.get("ce_security_id", "")),
                                    "exchange_segment": "NSE_FNO",
                                    "symbol": symbol,
                                    "strike": entry.get("strike_price", 0),
                                    "expiry": nearest_expiry,
                                    "option_type": "CE",
                                })
                                instruments.append({
                                    "security_id": str(entry.get("pe_security_id", "")),
                                    "exchange_segment": "NSE_FNO",
                                    "symbol": symbol,
                                    "strike": entry.get("strike_price", 0),
                                    "expiry": nearest_expiry,
                                    "option_type": "PE",
                                })

                        logger.info(f"{symbol}: loaded {len(instruments)} instruments for {nearest_expiry}")

            await dhan_broker.disconnect()
        else:
            logger.warning("Could not connect to Dhan to fetch instruments — feed may be empty")

        # Also add underlying instruments
        for symbol, sec_id in underlyings.items():
            instruments.append({
                "security_id": str(sec_id),
                "exchange_segment": "IDX_I",
                "symbol": symbol,
                "strike": 0,
                "expiry": "",
                "option_type": "",
            })

        dhan_feed_config = DhanFeedConfig(
            client_id=dhan_config.get("client_id", ""),
            access_token=dhan_config.get("access_token", ""),
            version="v2",
            subscription_type=feed_config.get("subscription_type", "Full"),
            poll_interval_ms=feed_config.get("update_interval_ms", 200),
        )

        logger.info(f"Created Dhan feed with {len(instruments)} instruments")
        return DhanDataFeed(dhan_feed_config, chains, instruments, on_tick, on_underlying)

    else:
        # Generic WebSocket feed (for IB or custom)
        ib_config = broker_config.get("ib", broker_config)
        ws_config = FeedConfig(
            url=f"ws://{ib_config.get('host', '127.0.0.1')}:{ib_config.get('port', 7497)}",
            symbols=feed_config.get("symbols", ["SPY"]),
        )
        return DataFeed(ws_config, chains, on_tick=on_tick, on_underlying=on_underlying)


async def run_live(settings: dict, risk_config: dict) -> None:
    """Run the bot in live/paper trading mode."""
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  OPTIONS HFT BOT — Starting Live Mode")
    logger.info("=" * 60)

    # Initialize components
    strategy_config = {**settings.get("strategy", {}), **settings.get("pricing", {})}
    broker_config = settings.get("broker", {})

    # Choose broker
    broker = create_broker(broker_config, logger)

    portfolio = Portfolio()
    risk_manager = RiskManager(risk_config, portfolio)
    signal_gen = SignalGenerator(strategy_config)
    signal_filter = SignalFilter(strategy_config)
    oms = OrderManager(broker, risk_manager, portfolio, settings)

    # Kill switch
    kill_switch = KillSwitch(
        config=risk_config,
        portfolio=portfolio,
        cancel_all_fn=oms.cancel_all,
        flatten_all_fn=oms.flatten_all,
    )

    # Options chains
    symbols = settings.get("data_feed", {}).get("symbols", ["NIFTY"])
    chains = {sym: OptionsChain(sym) for sym in symbols}

    # Storage
    storage_config = settings.get("storage", {})
    storage = TickStorage(
        engine=storage_config.get("engine", "sqlite"),
        sqlite_path=storage_config.get("sqlite_path", "data/ticks.db"),
    )

    # Connect broker
    connected = await broker.connect()
    if not connected:
        logger.error("Failed to connect to broker — exiting")
        return

    logger.info("All components initialized")
    logger.info(f"Broker: {broker_config.get('name', 'paper')}")
    logger.info(f"Tracking symbols: {symbols}")
    logger.info(f"Kill switch: {'ENABLED' if kill_switch.enabled else 'DISABLED'}")

    # Data feed callbacks
    async def on_tick(tick):
        symbol = tick.symbol
        if symbol in chains:
            chain = chains[symbol]

            # Store tick
            storage.store_tick(tick)

            # Update price history
            if chain.underlying:
                signal_gen.update_price_history(symbol, chain.underlying.price)

            # Generate and filter signals
            signals = signal_gen.generate(chain)
            filtered = signal_filter.apply(signals, chain)

            # Execute signals (unless kill switch is active)
            if not kill_switch.is_triggered and filtered:
                await oms.process_signals(filtered, chain)

    async def on_underlying(tick):
        symbol = tick.symbol
        if symbol in chains:
            signal_gen.update_price_history(symbol, tick.price)

    # Create data feed (Dhan or generic WebSocket)
    feed = await create_data_feed(settings, chains, on_tick, on_underlying, logger)

    try:
        tasks = [
            asyncio.create_task(feed.start()),
            asyncio.create_task(kill_switch.monitor()),
        ]

        # Status reporting every 30 seconds
        async def report_status():
            while True:
                await asyncio.sleep(30)
                logger.info(f"Portfolio: {portfolio.summary()}")
                logger.info(f"Risk: {risk_manager.status()}")
                logger.info(f"OMS: {oms.stats}")
                logger.info(f"Kill Switch: {kill_switch.status()}")
                if hasattr(feed, 'stats'):
                    logger.info(f"Feed: {feed.stats}")

        tasks.append(asyncio.create_task(report_status()))

        # Run until interrupted
        await asyncio.gather(*tasks)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await feed.stop()
        await broker.disconnect()
        storage.close()
        logger.info("Shutdown complete")


def run_backtest(settings: dict, risk_config: dict, data_path: str = None) -> None:
    """Run a backtest."""
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("  OPTIONS HFT BOT — Backtest Mode")
    logger.info("=" * 60)

    strategy_config = {**settings.get("strategy", {}), **settings.get("pricing", {})}

    # Load data
    loader = HistoricalDataLoader()
    if data_path:
        if data_path.endswith(".parquet"):
            data = loader.load_parquet(data_path)
        else:
            data = loader.load_csv(data_path)
    else:
        logger.info("No data file specified — generating sample data")
        data = loader.generate_sample_data(num_bars=2000)

    # Run backtest
    costs = risk_config.get("costs", {})
    bt_config = BacktestConfig(
        slippage_bps=costs.get("slippage_bps", 5),
        commission_per_contract=costs.get("commission_per_contract", 0.65),
    )

    engine = BacktestEngine(strategy_config, risk_config, bt_config)
    results = engine.run(data)

    # Print report
    report = PerformanceMetrics.format_report(results["metrics"])
    print(report)

    logger.info(f"Signals generated: {len(results['signals'])}")
    logger.info(f"Trades executed: {len(results['trades'])}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Options HFT Bot")
    parser.add_argument(
        "mode",
        choices=["live", "paper", "backtest"],
        help="Running mode: live, paper, or backtest",
    )
    parser.add_argument(
        "--broker",
        default=None,
        choices=["dhan", "ib", "paper"],
        help="Override broker selection (dhan, ib, paper)",
    )
    parser.add_argument(
        "--config",
        default="config",
        help="Config directory path (default: config/)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Data file for backtesting (CSV or Parquet)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Load config
    settings, risk_config = load_config(args.config)

    # Setup logging
    log_config = settings.get("logging", {})
    setup_logging(
        level=args.log_level or log_config.get("level", "INFO"),
        log_file=log_config.get("log_file"),
    )

    # Override broker from CLI if specified
    if args.broker:
        broker_name_map = {"dhan": "dhan", "ib": "interactive_brokers", "paper": "paper"}
        settings.setdefault("broker", {})["name"] = broker_name_map.get(args.broker, args.broker)

    # Run
    if args.mode == "backtest":
        run_backtest(settings, risk_config, args.data)
    elif args.mode in ("live", "paper"):
        if args.mode == "paper":
            settings.setdefault("broker", {})["name"] = "paper"
        asyncio.run(run_live(settings, risk_config))


if __name__ == "__main__":
    main()
