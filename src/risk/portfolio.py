"""
Portfolio Tracker.

Tracks all open positions, aggregates portfolio-level Greeks,
computes real-time P&L, and maintains position history.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import time

from ..data.chain import OptionType

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open options position."""
    contract_id: str         # Unique contract identifier
    symbol: str
    strike: float
    expiry: str
    option_type: OptionType
    side: str                # "BUY" or "SELL" (net direction)
    quantity: int
    entry_price: float
    current_price: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    entry_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in dollars (per contract = 100 shares)."""
        multiplier = 100  # Options multiplier
        sign = 1 if self.side == "BUY" else -1
        return sign * (self.current_price - self.entry_price) * self.quantity * multiplier

    @property
    def signed_quantity(self) -> int:
        """Positive for long, negative for short."""
        return self.quantity if self.side == "BUY" else -self.quantity

    @property
    def dollar_delta(self) -> float:
        """Delta exposure in dollar terms."""
        return self.delta * self.signed_quantity * 100

    @property
    def holding_time_seconds(self) -> float:
        """How long this position has been open."""
        return time.time() - self.entry_time


@dataclass
class TradeRecord:
    """Record of a completed trade (for P&L tracking)."""
    contract_id: str
    symbol: str
    strike: float
    expiry: str
    option_type: str
    side: str
    quantity: int
    price: float
    commission: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def net_cost(self) -> float:
        """Total cost including commissions."""
        sign = 1 if self.side == "BUY" else -1
        return sign * self.price * self.quantity * 100 + self.commission


class Portfolio:
    """
    Real-time portfolio tracker.

    Maintains all open positions, tracks P&L, and computes
    aggregate risk metrics (net Greeks, concentration, etc.).
    """

    def __init__(self):
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []
        self.realized_pnl: float = 0.0
        self.daily_realized_pnl: float = 0.0
        self._daily_reset_date: Optional[str] = None

    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        cid = position.contract_id
        if cid in self.positions:
            existing = self.positions[cid]
            # Netting: if same direction, increase; if opposite, reduce
            if existing.side == position.side:
                total_cost = (
                    existing.entry_price * existing.quantity
                    + position.entry_price * position.quantity
                )
                existing.quantity += position.quantity
                existing.entry_price = total_cost / existing.quantity
            else:
                # Partial or full close
                close_qty = min(existing.quantity, position.quantity)
                pnl = self._compute_close_pnl(existing, position.entry_price, close_qty)
                self.realized_pnl += pnl
                self.daily_realized_pnl += pnl

                existing.quantity -= close_qty
                remaining_new = position.quantity - close_qty

                if existing.quantity <= 0:
                    del self.positions[cid]

                if remaining_new > 0:
                    position.quantity = remaining_new
                    self.positions[cid] = position
        else:
            self.positions[cid] = position

        logger.info(f"Position updated: {cid} qty={position.quantity}")

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a trade for history tracking."""
        self.trade_history.append(trade)

    def update_mark(self, contract_id: str, price: float, delta: float = 0,
                    gamma: float = 0, theta: float = 0, vega: float = 0) -> None:
        """Update mark-to-market price and Greeks for a position."""
        if contract_id in self.positions:
            pos = self.positions[contract_id]
            pos.current_price = price
            pos.delta = delta
            pos.gamma = gamma
            pos.theta = theta
            pos.vega = vega
            pos.last_update = time.time()

    def close_position(self, contract_id: str, close_price: float) -> float:
        """Close a position and return realized P&L."""
        if contract_id not in self.positions:
            return 0.0

        pos = self.positions[contract_id]
        pnl = self._compute_close_pnl(pos, close_price, pos.quantity)
        self.realized_pnl += pnl
        self.daily_realized_pnl += pnl

        del self.positions[contract_id]
        logger.info(f"Closed position {contract_id}, PnL: ${pnl:.2f}")
        return pnl

    # ── Aggregate Metrics ──────────────────────────────

    @property
    def net_delta(self) -> float:
        return sum(p.delta * p.signed_quantity for p in self.positions.values())

    @property
    def net_gamma(self) -> float:
        return sum(p.gamma * p.signed_quantity for p in self.positions.values())

    @property
    def net_theta(self) -> float:
        return sum(p.theta * p.signed_quantity for p in self.positions.values())

    @property
    def net_vega(self) -> float:
        return sum(p.vega * p.signed_quantity for p in self.positions.values())

    @property
    def total_contracts(self) -> int:
        return sum(abs(p.quantity) for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.total_unrealized_pnl

    @property
    def daily_pnl(self) -> float:
        return self.daily_realized_pnl + self.total_unrealized_pnl

    def expiry_concentration(self) -> dict[str, float]:
        """Compute concentration by expiration (as fraction of total)."""
        total = self.total_contracts
        if total == 0:
            return {}
        expiry_counts: dict[str, int] = {}
        for p in self.positions.values():
            expiry_counts[p.expiry] = expiry_counts.get(p.expiry, 0) + abs(p.quantity)
        return {exp: count / total for exp, count in expiry_counts.items()}

    def get_positions_by_symbol(self, symbol: str) -> list[Position]:
        return [p for p in self.positions.values() if p.symbol == symbol]

    def reset_daily(self) -> None:
        """Reset daily P&L counters (call at start of each trading day)."""
        self.daily_realized_pnl = 0.0
        logger.info("Daily P&L counters reset")

    def summary(self) -> dict:
        """Portfolio summary snapshot."""
        return {
            "positions": len(self.positions),
            "total_contracts": self.total_contracts,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "net_delta": round(self.net_delta, 2),
            "net_gamma": round(self.net_gamma, 4),
            "net_theta": round(self.net_theta, 2),
            "net_vega": round(self.net_vega, 2),
            "trades_today": len([t for t in self.trade_history]),
        }

    @staticmethod
    def _compute_close_pnl(position: Position, close_price: float, qty: int) -> float:
        multiplier = 100
        if position.side == "BUY":
            return (close_price - position.entry_price) * qty * multiplier
        else:
            return (position.entry_price - close_price) * qty * multiplier
