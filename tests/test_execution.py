"""
Unit tests for the execution engine.

Tests PaperBroker, SmartRouter, and Order model behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import asyncio
import time

from src.execution.broker import (
    BrokerBase, PaperBroker, Order, OrderType, OrderStatus
)
from src.execution.router import SmartRouter


# ── Helpers ───────────────────────────────────────────


def make_order(**kwargs) -> Order:
    defaults = {
        "symbol": "SPY",
        "strike": 450.0,
        "expiry": "2025-03-21",
        "option_type": "call",
        "side": "BUY",
        "quantity": 1,
        "order_type": OrderType.LIMIT,
        "limit_price": 5.00,
    }
    defaults.update(kwargs)
    return Order(**defaults)


# ── PaperBroker Tests ─────────────────────────────────


class TestPaperBroker:
    """Tests for the paper trading broker."""

    def setup_method(self):
        self.broker = PaperBroker(slippage_bps=5.0, commission_per_contract=0.65)

    @pytest.mark.asyncio
    async def test_connect(self):
        """Should connect successfully."""
        result = await self.broker.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_place_limit_order_buy(self):
        """Should fill BUY limit orders at limit price + slippage."""
        order = make_order(limit_price=5.00, side="BUY")
        result = await self.broker.place_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 1
        assert result.order_id.startswith("PAPER-")
        # Slippage: 5.00 * 5/10000 = 0.0025. fill = 5.00 + 0.0025 = 5.0025 → rounds to 5.0
        # With bps=5, buy fills at or above limit
        assert result.filled_price >= 5.00

    @pytest.mark.asyncio
    async def test_place_limit_order_sell(self):
        """Sell order should fill at limit price - slippage."""
        order = make_order(limit_price=100.00, side="SELL")
        result = await self.broker.place_order(order)

        assert result.status == OrderStatus.FILLED
        # Slippage: 100.00 * 5/10000 = 0.05. fill = 99.95
        assert result.filled_price <= 100.00

    @pytest.mark.asyncio
    async def test_commission(self):
        """Should charge correct commission."""
        order = make_order(quantity=10)
        result = await self.broker.place_order(order)

        assert result.commission == 0.65 * 10

    @pytest.mark.asyncio
    async def test_order_ids_unique(self):
        """Each order should get a unique ID."""
        o1 = await self.broker.place_order(make_order())
        o2 = await self.broker.place_order(make_order())
        assert o1.order_id != o2.order_id

    @pytest.mark.asyncio
    async def test_order_ids_sequential(self):
        """Order IDs should be sequentially numbered."""
        o1 = await self.broker.place_order(make_order())
        o2 = await self.broker.place_order(make_order())
        assert o1.order_id == "PAPER-000001"
        assert o2.order_id == "PAPER-000002"

    @pytest.mark.asyncio
    async def test_get_order_status(self):
        """Should retrieve order status."""
        order = await self.broker.place_order(make_order())
        status = await self.broker.get_order_status(order.order_id)
        assert status is not None
        assert status.order_id == order.order_id
        assert status.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_order_status_missing(self):
        """Should return None for unknown order."""
        status = await self.broker.get_order_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_cancel_all_no_pending(self):
        """Cancel all with all-filled orders should return 0."""
        await self.broker.place_order(make_order())
        await self.broker.place_order(make_order())
        count = await self.broker.cancel_all()
        assert count == 0  # All already filled

    @pytest.mark.asyncio
    async def test_close_position(self):
        """Should create a closing order."""
        result = await self.broker.close_position("SPY_2025-03-21_450.0_call", 5)
        assert result.status == OrderStatus.FILLED
        assert result.side == "SELL"
        assert result.quantity == 5

    @pytest.mark.asyncio
    async def test_close_position_negative_qty(self):
        """Negative quantity should create BUY order (covering short)."""
        result = await self.broker.close_position("SPY_2025-03-21_450.0_call", -5)
        assert result.side == "BUY"
        assert result.quantity == 5

    @pytest.mark.asyncio
    async def test_fill_time_set(self):
        """Filled orders should have fill_time set."""
        order = await self.broker.place_order(make_order())
        assert order.fill_time is not None

    @pytest.mark.asyncio
    async def test_market_order(self):
        """Market orders should fill at 0 (no limit price)."""
        order = make_order(order_type=OrderType.MARKET, limit_price=None)
        result = await self.broker.place_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == 0  # No price simulation for market orders


# ── Order Model Tests ─────────────────────────────────


class TestOrderModel:
    """Tests for the Order dataclass."""

    def test_contract_symbol(self):
        """Should format contract symbol correctly."""
        order = make_order(symbol="AAPL", strike=150.0, expiry="2025-04-18", option_type="put")
        assert order.contract_symbol == "AAPL_2025-04-18_150.0_put"

    def test_is_terminal_filled(self):
        """Filled orders should be terminal."""
        order = make_order()
        order.status = OrderStatus.FILLED
        assert order.is_terminal is True

    def test_is_terminal_cancelled(self):
        """Cancelled orders should be terminal."""
        order = make_order()
        order.status = OrderStatus.CANCELLED
        assert order.is_terminal is True

    def test_is_terminal_rejected(self):
        """Rejected orders should be terminal."""
        order = make_order()
        order.status = OrderStatus.REJECTED
        assert order.is_terminal is True

    def test_is_terminal_error(self):
        """Error orders should be terminal."""
        order = make_order()
        order.status = OrderStatus.ERROR
        assert order.is_terminal is True

    def test_is_terminal_pending(self):
        """Pending orders should NOT be terminal."""
        order = make_order()
        order.status = OrderStatus.PENDING
        assert order.is_terminal is False

    def test_is_terminal_submitted(self):
        """Submitted orders should NOT be terminal."""
        order = make_order()
        order.status = OrderStatus.SUBMITTED
        assert order.is_terminal is False

    def test_order_type_enum(self):
        """Order types should have correct values."""
        assert OrderType.MARKET.value == "MKT"
        assert OrderType.LIMIT.value == "LMT"
        assert OrderType.IOC.value == "IOC"

    def test_order_status_enum(self):
        """Order statuses should have string values."""
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.CANCELLED.value == "CANCELLED"

    def test_submit_time_auto_set(self):
        """Submit time should be auto-set on creation."""
        order = make_order()
        assert order.submit_time > 0


# ── SmartRouter Tests ─────────────────────────────────


class TestSmartRouter:
    """Tests for the smart order router."""

    def setup_method(self):
        self.broker = PaperBroker(slippage_bps=0)
        self.router = SmartRouter(self.broker, {
            "max_retry_attempts": 3,
            "stale_order_timeout_sec": 1,
            "price_improve_bps": 50,  # 50 bps for testable improvement
        })

    @pytest.mark.asyncio
    async def test_route_fills_immediately_with_paper(self):
        """Paper broker fills immediately — no retries needed."""
        order = make_order()
        result = await self.router.route_order(order)
        assert result.status == OrderStatus.FILLED

    def test_price_improvement_buy(self):
        """BUY price improvement should raise the limit (more aggressive)."""
        order = make_order(limit_price=100.00, side="BUY")
        self.router._improve_price(order, attempt=0)
        # improvement = 100 * 50/10000 * 1 = 0.50 → new price = 100.50
        assert order.limit_price > 100.00

    def test_price_improvement_sell(self):
        """SELL price improvement should lower the limit (more aggressive)."""
        order = make_order(limit_price=100.00, side="SELL")
        self.router._improve_price(order, attempt=0)
        # improvement = 100 * 50/10000 * 1 = 0.50 → new price = 99.50
        assert order.limit_price < 100.00

    def test_price_improvement_scales_with_attempt(self):
        """Later attempts should improve price more aggressively."""
        # Create separate orders since _improve_price mutates in-place
        order1 = make_order(limit_price=100.00, side="BUY")
        self.router._improve_price(order1, attempt=0)
        price_after_attempt_0 = order1.limit_price

        order2 = make_order(limit_price=100.00, side="BUY")
        self.router._improve_price(order2, attempt=2)
        price_after_attempt_2 = order2.limit_price

        # attempt=2 scale factor is (2+1)=3 vs attempt=0 factor (0+1)=1
        assert price_after_attempt_2 > price_after_attempt_0

    def test_no_improvement_without_limit(self):
        """Should handle None limit price gracefully."""
        order = make_order(limit_price=None)
        self.router._improve_price(order, attempt=0)
        assert order.limit_price is None

    def test_router_config(self):
        """Router should store config values."""
        assert self.router.max_retries == 3
        assert self.router.stale_timeout == 1
        assert self.router.price_improve_bps == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
