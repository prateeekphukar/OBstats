"""
Risk Manager.

Pre-trade risk checks: position limits, daily loss, Greek limits,
rate limits, and concentration checks. Every order must pass all
checks before execution.
"""

import logging
import time
from typing import Optional

from .portfolio import Portfolio
from ..strategy.signals import Signal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Pre-trade and portfolio-level risk management.

    Every signal/order passes through a series of risk checks.
    If any check fails, the order is rejected with a reason.
    """

    def __init__(self, config: dict, portfolio: Portfolio):
        """
        Args:
            config: Risk limits from risk_limits.yaml.
            portfolio: Portfolio tracker instance.
        """
        self.config = config
        self.portfolio = portfolio

        # Position limits
        pos_limits = config.get("position_limits", {})
        self.max_total_contracts = pos_limits.get("max_total_contracts", 50)
        self.max_per_symbol = pos_limits.get("max_contracts_per_symbol", 20)
        self.max_per_strike = pos_limits.get("max_contracts_per_strike", 10)
        self.max_expiry_concentration = pos_limits.get("max_single_expiry_pct", 0.40)

        # P&L limits
        pnl_limits = config.get("pnl_limits", {})
        self.max_daily_loss = pnl_limits.get("max_daily_loss", 5000.0)
        self.max_unrealized_loss = pnl_limits.get("max_unrealized_loss", 3000.0)

        # Greeks limits
        greeks = config.get("greeks_limits", {})
        self.max_delta = greeks.get("max_portfolio_delta", 100)
        self.max_gamma = greeks.get("max_portfolio_gamma", 50)
        self.max_vega = greeks.get("max_portfolio_vega", 500)

        # Rate limits
        rates = config.get("rate_limits", {})
        self.max_trades_per_minute = rates.get("max_trades_per_minute", 30)
        self.max_orders_per_second = rates.get("max_orders_per_second", 5)

        # Rate tracking
        self._trade_timestamps: list[float] = []
        self._order_timestamps: list[float] = []

    def approve(self, signal: Signal, quantity: int = 1) -> tuple[bool, str]:
        """
        Run all pre-trade risk checks on a signal.

        Args:
            signal: The trading signal to evaluate.
            quantity: Proposed order quantity.

        Returns:
            Tuple of (approved: bool, reason: str).
        """
        checks = [
            self._check_daily_loss(),
            self._check_unrealized_loss(),
            self._check_position_limits(signal, quantity),
            self._check_symbol_concentration(signal, quantity),
            self._check_expiry_concentration(signal),
            self._check_greeks(),
            self._check_trade_rate(),
            self._check_order_rate(),
        ]

        for passed, reason in checks:
            if not passed:
                logger.warning(f"Risk check FAILED: {reason} | Signal: {signal.signal_type.value} {signal.strike}")
                return False, reason

        return True, "approved"

    def record_trade(self) -> None:
        """Record a trade for rate limiting."""
        self._trade_timestamps.append(time.time())

    def record_order(self) -> None:
        """Record an order for rate limiting."""
        self._order_timestamps.append(time.time())

    # ── Individual Risk Checks ─────────────────────────

    def _check_daily_loss(self) -> tuple[bool, str]:
        """Check if daily P&L exceeds loss limit."""
        daily_pnl = self.portfolio.daily_pnl
        if daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit exceeded: ${daily_pnl:.2f} < -${self.max_daily_loss}"
        return True, ""

    def _check_unrealized_loss(self) -> tuple[bool, str]:
        """Check if unrealized losses are too large."""
        for pos in self.portfolio.positions.values():
            if pos.unrealized_pnl < -self.max_unrealized_loss:
                return False, f"Unrealized loss on {pos.contract_id}: ${pos.unrealized_pnl:.2f}"
        return True, ""

    def _check_position_limits(self, signal: Signal, quantity: int) -> tuple[bool, str]:
        """Check total position size limits."""
        current = self.portfolio.total_contracts
        if current + quantity > self.max_total_contracts:
            return False, f"Total contracts would exceed limit: {current + quantity} > {self.max_total_contracts}"
        return True, ""

    def _check_symbol_concentration(self, signal: Signal, quantity: int) -> tuple[bool, str]:
        """Check per-symbol position concentration."""
        symbol_positions = self.portfolio.get_positions_by_symbol(signal.symbol)
        symbol_total = sum(abs(p.quantity) for p in symbol_positions)
        if symbol_total + quantity > self.max_per_symbol:
            return False, f"Per-symbol limit: {signal.symbol} would have {symbol_total + quantity} > {self.max_per_symbol}"
        return True, ""

    def _check_expiry_concentration(self, signal: Signal) -> tuple[bool, str]:
        """Check that no single expiry has too much concentration."""
        concentration = self.portfolio.expiry_concentration()
        expiry_pct = concentration.get(signal.expiry, 0)
        if expiry_pct > self.max_expiry_concentration:
            return False, f"Expiry concentration: {signal.expiry} = {expiry_pct:.0%} > {self.max_expiry_concentration:.0%}"
        return True, ""

    def _check_greeks(self) -> tuple[bool, str]:
        """Check portfolio-level Greek limits."""
        if abs(self.portfolio.net_delta) > self.max_delta:
            return False, f"Portfolio delta: {self.portfolio.net_delta:.1f} > ±{self.max_delta}"
        if abs(self.portfolio.net_gamma) > self.max_gamma:
            return False, f"Portfolio gamma: {self.portfolio.net_gamma:.1f} > ±{self.max_gamma}"
        if abs(self.portfolio.net_vega) > self.max_vega:
            return False, f"Portfolio vega: {self.portfolio.net_vega:.1f} > ±{self.max_vega}"
        return True, ""

    def _check_trade_rate(self) -> tuple[bool, str]:
        """Check trades-per-minute rate limit."""
        now = time.time()
        cutoff = now - 60
        self._trade_timestamps = [t for t in self._trade_timestamps if t > cutoff]
        if len(self._trade_timestamps) >= self.max_trades_per_minute:
            return False, f"Trade rate limit: {len(self._trade_timestamps)} trades in last minute"
        return True, ""

    def _check_order_rate(self) -> tuple[bool, str]:
        """Check orders-per-second rate limit."""
        now = time.time()
        cutoff = now - 1
        self._order_timestamps = [t for t in self._order_timestamps if t > cutoff]
        if len(self._order_timestamps) >= self.max_orders_per_second:
            return False, f"Order rate limit: {len(self._order_timestamps)} orders in last second"
        return True, ""

    def status(self) -> dict:
        """Current risk status summary."""
        return {
            "daily_pnl": round(self.portfolio.daily_pnl, 2),
            "daily_loss_limit": self.max_daily_loss,
            "daily_loss_used_pct": round(abs(min(0, self.portfolio.daily_pnl)) / self.max_daily_loss * 100, 1),
            "total_contracts": self.portfolio.total_contracts,
            "max_contracts": self.max_total_contracts,
            "net_delta": round(self.portfolio.net_delta, 2),
            "net_vega": round(self.portfolio.net_vega, 2),
            "trades_last_minute": len([t for t in self._trade_timestamps if t > time.time() - 60]),
        }
