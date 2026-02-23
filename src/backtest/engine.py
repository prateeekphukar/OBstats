"""
Backtesting Engine.

Event-driven backtester that replays historical options data
through the signal generator with realistic fill simulation.
"""

import logging
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ..strategy.signals import SignalGenerator, Signal
from ..strategy.filters import SignalFilter
from ..risk.portfolio import Portfolio
from ..risk.manager import RiskManager
from ..execution.broker import PaperBroker, Order, OrderType
from ..data.chain import OptionsChain, OptionTick, UnderlyingTick, OptionType
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    slippage_bps: float = 5.0
    commission_per_contract: float = 0.65
    initial_capital: float = 100_000.0
    max_signals_per_bar: int = 3


class BacktestEngine:
    """
    Event-driven options backtesting engine.

    Replays historical data through the full strategy pipeline:
    1. Data → Chain update
    2. Signal generation
    3. Signal filtering
    4. Risk checks
    5. Simulated execution
    6. Portfolio update
    """

    def __init__(
        self,
        strategy_config: dict,
        risk_config: dict,
        backtest_config: Optional[BacktestConfig] = None,
    ):
        self.bt_config = backtest_config or BacktestConfig()

        # Initialize components
        self.portfolio = Portfolio()
        self.risk_manager = RiskManager(risk_config, self.portfolio)
        self.signal_gen = SignalGenerator(strategy_config)
        self.signal_filter = SignalFilter(strategy_config)
        self.broker = PaperBroker(
            slippage_bps=self.bt_config.slippage_bps,
            commission_per_contract=self.bt_config.commission_per_contract,
        )

        # Tracking
        self._equity_curve: list[float] = []
        self._signals_generated: list[dict] = []
        self._trades: list[dict] = []
        self._daily_pnl: list[float] = []

    def run(self, data: pd.DataFrame) -> dict:
        """
        Run a backtest on historical data.

        Args:
            data: DataFrame with columns:
                  [timestamp, symbol, strike, expiry, option_type,
                   bid, ask, last, volume, open_interest, underlying_price]

        Returns:
            Dict with performance metrics and detailed results.
        """
        logger.info(f"Starting backtest on {len(data)} rows")

        # Group data by timestamp
        if "timestamp" not in data.columns:
            raise ValueError("Data must have a 'timestamp' column")

        data = data.sort_values("timestamp")
        grouped = data.groupby("timestamp")

        chain = OptionsChain(data["symbol"].iloc[0])
        total_bars = len(grouped)

        for bar_idx, (timestamp, group) in enumerate(grouped):
            # Step 1: Update chain with this bar's data
            self._update_chain(chain, group, timestamp)

            # Step 2: Update price history for realized vol
            if chain.underlying:
                self.signal_gen.update_price_history(
                    chain.symbol, chain.underlying.price
                )

            # Step 3: Generate signals
            signals = self.signal_gen.generate(chain)
            for s in signals:
                self._signals_generated.append({
                    "timestamp": timestamp,
                    "type": s.signal_type.value,
                    "strike": s.strike,
                    "strength": s.strength,
                })

            # Step 4: Filter signals
            filtered = self.signal_filter.apply(signals, chain)

            # Step 5: Execute top signals
            for signal in filtered[:self.bt_config.max_signals_per_bar]:
                self._execute_signal(signal, chain, timestamp)

            # Step 6: Mark to market
            self._mark_to_market(chain)

            # Track equity
            capital = self.bt_config.initial_capital + self.portfolio.total_pnl
            self._equity_curve.append(capital)

            # Progress logging
            if (bar_idx + 1) % 1000 == 0:
                logger.info(f"Progress: {bar_idx + 1}/{total_bars} bars, P&L: ${self.portfolio.total_pnl:.2f}")

        # Compute final metrics
        metrics = PerformanceMetrics.compute(
            equity_curve=self._equity_curve,
            trades=self._trades,
            initial_capital=self.bt_config.initial_capital,
        )

        logger.info(f"Backtest complete: {metrics}")

        return {
            "metrics": metrics,
            "equity_curve": self._equity_curve,
            "signals": self._signals_generated,
            "trades": self._trades,
            "final_portfolio": self.portfolio.summary(),
        }

    def _update_chain(self, chain: OptionsChain, group: pd.DataFrame, timestamp) -> None:
        """Update the chain with a group of ticks."""
        for _, row in group.iterrows():
            # Update underlying
            if "underlying_price" in row and pd.notna(row["underlying_price"]):
                chain.update_underlying(UnderlyingTick(
                    symbol=row["symbol"],
                    price=float(row["underlying_price"]),
                    bid=float(row.get("underlying_bid", row["underlying_price"])),
                    ask=float(row.get("underlying_ask", row["underlying_price"])),
                    timestamp=float(timestamp) if isinstance(timestamp, (int, float)) else 0,
                ))

            # Update option tick
            tick = OptionTick(
                symbol=str(row["symbol"]),
                strike=float(row["strike"]),
                expiry=str(row["expiry"]),
                option_type=OptionType(row["option_type"]),
                bid=float(row.get("bid", 0)),
                ask=float(row.get("ask", 0)),
                last=float(row.get("last", 0)),
                volume=int(row.get("volume", 0)),
                open_interest=int(row.get("open_interest", 0)),
                timestamp=float(timestamp) if isinstance(timestamp, (int, float)) else 0,
            )
            chain.update_option(tick)

    def _execute_signal(self, signal: Signal, chain: OptionsChain, timestamp) -> None:
        """Execute a signal through risk checks and simulated fills."""
        quantity = 1
        approved, reason = self.risk_manager.approve(signal, quantity)
        if not approved:
            return

        tick = chain.get_tick(signal.strike, signal.expiry, signal.option_type)
        if not tick or tick.mid <= 0:
            return

        # Simulate fill
        fill_price = tick.mid
        slippage = fill_price * self.bt_config.slippage_bps / 10000
        side = signal.direction

        # Handle COMPLEX signals (surface arb, skew) — default to SELL
        if side == "COMPLEX":
            side = "SELL" if signal.signal_type.value in ("SURFACE_ARB",) else "BUY"

        if side == "BUY":
            fill_price += slippage
        else:
            fill_price -= slippage

        commission = self.bt_config.commission_per_contract * quantity

        # Compute P&L: for SELL signals the edge is (fill_price - fair_value)
        # For simplicity, estimate P&L as the signal's deviation * quantity
        estimated_pnl = 0.0
        if signal.iv and signal.realized_vol and signal.realized_vol > 0:
            # P&L from vol edge: vega * (IV - fair_IV) * quantity
            iv_edge = abs(signal.iv - signal.realized_vol)
            estimated_pnl = (tick.vega or 0.5) * iv_edge * quantity * 100
            if side == "SELL":
                estimated_pnl = abs(estimated_pnl)  # Selling high IV = profit
            else:
                estimated_pnl = abs(estimated_pnl)  # Buying low IV = profit
        elif signal.deviation:
            estimated_pnl = abs(signal.deviation) * quantity * 100

        # Subtract commission
        estimated_pnl -= commission

        self._trades.append({
            "timestamp": timestamp,
            "signal_type": signal.signal_type.value,
            "symbol": signal.symbol,
            "strike": signal.strike,
            "expiry": signal.expiry,
            "side": side,
            "quantity": quantity,
            "price": round(fill_price, 4),
            "commission": commission,
            "strength": signal.strength,
            "pnl": round(estimated_pnl, 2),
        })

        # Update portfolio P&L
        self.portfolio.realized_pnl += estimated_pnl

        self.risk_manager.record_trade()

    def _mark_to_market(self, chain: OptionsChain) -> None:
        """Update all position prices with current chain data."""
        for pos in self.portfolio.positions.values():
            tick = chain.get_tick(pos.strike, pos.expiry, pos.option_type)
            if tick:
                self.portfolio.update_mark(
                    pos.contract_id, tick.mid,
                    delta=tick.delta or 0,
                    gamma=tick.gamma or 0,
                    theta=tick.theta or 0,
                    vega=tick.vega or 0,
                )
