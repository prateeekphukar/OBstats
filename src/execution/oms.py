"""
Order Management System (OMS).

Translates signals into orders, manages order lifecycle,
tracks fills, and updates the portfolio.
"""

import asyncio
import logging
import time
from typing import Optional

from .broker import BrokerBase, Order, OrderType, OrderStatus
from .router import SmartRouter
from ..strategy.signals import Signal, SignalType
from ..risk.manager import RiskManager
from ..risk.portfolio import Portfolio, Position, TradeRecord
from ..data.chain import OptionsChain, OptionType

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Order Management System.

    Central coordinator for:
    1. Signal → Order translation
    2. Risk approval
    3. Order submission via SmartRouter
    4. Fill tracking and portfolio updates
    """

    def __init__(
        self,
        broker: BrokerBase,
        risk_manager: RiskManager,
        portfolio: Portfolio,
        config: dict,
    ):
        self.broker = broker
        self.risk = risk_manager
        self.portfolio = portfolio
        self.config = config
        self.router = SmartRouter(broker, config.get("execution", {}))
        self._pending_orders: dict[str, Order] = {}
        self._fill_count = 0
        self._reject_count = 0

    async def process_signal(self, signal: Signal, chain: OptionsChain) -> list[Order]:
        """
        Process a trading signal end-to-end.

        1. Risk check
        2. Build order(s)
        3. Route and submit
        4. Track fills
        5. Update portfolio

        Args:
            signal: The trading signal to execute.
            chain: Current options chain for pricing.

        Returns:
            List of submitted orders.
        """
        # Step 1: Risk approval
        quantity = self._compute_quantity(signal)
        approved, reason = self.risk.approve(signal, quantity)
        if not approved:
            self._reject_count += 1
            logger.info(f"Signal rejected: {reason}")
            return []

        # Step 2: Build orders from signal
        orders = self._build_orders(signal, chain, quantity)
        if not orders:
            return []

        # Step 3: Submit via smart router
        results = []
        for order in orders:
            self.risk.record_order()
            submitted = await self.router.route_order(order)
            self._pending_orders[submitted.order_id] = submitted
            results.append(submitted)

            # Step 4: If filled, update portfolio
            if submitted.status == OrderStatus.FILLED:
                self._handle_fill(submitted, signal)

        return results

    async def process_signals(self, signals: list[Signal], chain: OptionsChain) -> list[Order]:
        """Process multiple signals sequentially."""
        all_orders = []
        for signal in signals:
            orders = await self.process_signal(signal, chain)
            all_orders.extend(orders)
        return all_orders

    async def check_pending_orders(self) -> None:
        """Check and update status of all pending orders."""
        to_remove = []
        for order_id, order in self._pending_orders.items():
            if order.is_terminal:
                to_remove.append(order_id)
                continue

            updated = await self.broker.get_order_status(order_id)
            if updated and updated.is_terminal:
                if updated.status == OrderStatus.FILLED:
                    self._handle_fill(updated, None)
                to_remove.append(order_id)

        for oid in to_remove:
            del self._pending_orders[oid]

    async def cancel_stale_orders(self, max_age_seconds: float = 5.0) -> int:
        """Cancel orders that have been pending too long."""
        now = time.time()
        cancelled = 0
        for order_id, order in list(self._pending_orders.items()):
            if not order.is_terminal and (now - order.submit_time) > max_age_seconds:
                success = await self.broker.cancel_order(order_id)
                if success:
                    cancelled += 1
                    order.status = OrderStatus.CANCELLED
                    logger.info(f"Stale order cancelled: {order_id}")
        return cancelled

    async def cancel_all(self) -> int:
        """Cancel all open orders."""
        count = await self.broker.cancel_all()
        self._pending_orders.clear()
        return count

    async def flatten_all(self) -> None:
        """Close all positions."""
        for pos in list(self.portfolio.positions.values()):
            close_side = "SELL" if pos.side == "BUY" else "BUY"
            order = Order(
                symbol=pos.symbol,
                strike=pos.strike,
                expiry=pos.expiry,
                option_type=pos.option_type.value,
                side=close_side,
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
            )
            result = await self.broker.place_order(order)
            if result.status == OrderStatus.FILLED:
                self.portfolio.close_position(pos.contract_id, result.filled_price)

    def _build_orders(self, signal: Signal, chain: OptionsChain, quantity: int) -> list[Order]:
        """Translate a signal into one or more concrete orders."""
        tick = chain.get_tick(signal.strike, signal.expiry, signal.option_type)
        if not tick:
            logger.warning(f"No tick data for {signal.strike} {signal.expiry}")
            return []

        mid_price = tick.mid
        if mid_price <= 0:
            return []

        if signal.signal_type == SignalType.SELL_VOL:
            # Sell straddle: sell call + sell put
            call_tick = chain.get_tick(signal.strike, signal.expiry, OptionType.CALL)
            put_tick = chain.get_tick(signal.strike, signal.expiry, OptionType.PUT)
            orders = []
            if call_tick and call_tick.mid > 0:
                orders.append(self._create_order(signal, "SELL", OptionType.CALL, call_tick.mid, quantity))
            if put_tick and put_tick.mid > 0:
                orders.append(self._create_order(signal, "SELL", OptionType.PUT, put_tick.mid, quantity))
            return orders

        elif signal.signal_type == SignalType.BUY_VOL:
            # Buy straddle: buy call + buy put
            call_tick = chain.get_tick(signal.strike, signal.expiry, OptionType.CALL)
            put_tick = chain.get_tick(signal.strike, signal.expiry, OptionType.PUT)
            orders = []
            if call_tick and call_tick.mid > 0:
                orders.append(self._create_order(signal, "BUY", OptionType.CALL, call_tick.mid, quantity))
            if put_tick and put_tick.mid > 0:
                orders.append(self._create_order(signal, "BUY", OptionType.PUT, put_tick.mid, quantity))
            return orders

        elif signal.signal_type == SignalType.SURFACE_ARB:
            # Single leg for now — in production, build butterfly
            side = "SELL" if signal.deviation and signal.deviation > 0 else "BUY"
            return [self._create_order(signal, side, signal.option_type, mid_price, quantity)]

        elif signal.signal_type == SignalType.SKEW_REVERT:
            # Single leg directional
            side = "BUY" if signal.option_type == OptionType.PUT else "SELL"
            return [self._create_order(signal, side, signal.option_type, mid_price, quantity)]

        return []

    def _create_order(self, signal: Signal, side: str, option_type: OptionType,
                      mid_price: float, quantity: int) -> Order:
        """Create an Order from signal parameters."""
        offset_bps = self.config.get("execution", {}).get("price_offset_bps", 5)
        offset = mid_price * offset_bps / 10000

        if side == "BUY":
            limit_price = round(mid_price - offset, 2)  # Bid below mid
        else:
            limit_price = round(mid_price + offset, 2)  # Ask above mid

        return Order(
            symbol=signal.symbol,
            strike=signal.strike,
            expiry=signal.expiry,
            option_type=option_type.value,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
        )

    def _compute_quantity(self, signal: Signal) -> int:
        """Determine order quantity based on signal strength and config."""
        base_qty = self.config.get("execution", {}).get("base_quantity", 1)
        # Scale with signal strength (1-10)
        if signal.strength > 7:
            return base_qty * 2
        return base_qty

    def _handle_fill(self, order: Order, signal: Optional[Signal]) -> None:
        """Handle a filled order — update portfolio and risk."""
        self._fill_count += 1
        self.risk.record_trade()

        # Create position
        contract_id = order.contract_symbol
        position = Position(
            contract_id=contract_id,
            symbol=order.symbol,
            strike=order.strike,
            expiry=order.expiry,
            option_type=OptionType(order.option_type),
            side=order.side,
            quantity=order.filled_qty,
            entry_price=order.filled_price,
            current_price=order.filled_price,
        )
        self.portfolio.add_position(position)

        # Record trade
        trade = TradeRecord(
            contract_id=contract_id,
            symbol=order.symbol,
            strike=order.strike,
            expiry=order.expiry,
            option_type=order.option_type,
            side=order.side,
            quantity=order.filled_qty,
            price=order.filled_price,
            commission=order.commission,
        )
        self.portfolio.record_trade(trade)

        logger.info(
            f"Fill: {order.side} {order.filled_qty}x {contract_id} "
            f"@ ${order.filled_price:.2f} (comm: ${order.commission:.2f})"
        )

    @property
    def stats(self) -> dict:
        return {
            "fills": self._fill_count,
            "rejects": self._reject_count,
            "pending_orders": len(self._pending_orders),
        }
