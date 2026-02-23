"""
Smart Order Router.

Routes orders to the broker with intelligent pricing,
retry logic, and order type selection.
"""

import asyncio
import logging
import time
from typing import Optional

from .broker import BrokerBase, Order, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class SmartRouter:
    """
    Smart order routing engine.

    Features:
    - Aggressive limit pricing (offset from mid)
    - IOC conversion for time-sensitive orders
    - Auto-retry with price improvement
    - Stale order cancellation
    """

    def __init__(self, broker: BrokerBase, config: dict):
        self.broker = broker
        self.config = config
        self.max_retries = config.get("max_retry_attempts", 3)
        self.stale_timeout = config.get("stale_order_timeout_sec", 5)
        self.price_improve_bps = config.get("price_improve_bps", 2)

    async def route_order(self, order: Order) -> Order:
        """
        Route an order with retry logic.

        1. Submit at initial limit price.
        2. If not filled within timeout, cancel and retry with price improvement.
        3. After max retries, submit as IOC at market-crossing price.
        """
        for attempt in range(self.max_retries):
            result = await self.broker.place_order(order)

            if result.status == OrderStatus.FILLED:
                return result

            if result.status in (OrderStatus.ERROR, OrderStatus.REJECTED):
                logger.warning(f"Order {result.order_id} rejected: {result.error_msg}")
                return result

            # Wait for fill
            filled = await self._wait_for_fill(result.order_id, self.stale_timeout)
            if filled:
                return filled

            # Cancel and retry with improved price
            await self.broker.cancel_order(result.order_id)
            order = self._improve_price(order, attempt)
            logger.info(f"Retry {attempt + 1}/{self.max_retries}: new price ${order.limit_price:.2f}")

        # Final attempt: aggressive IOC
        order.order_type = OrderType.IOC
        if order.limit_price:
            # Cross the spread
            cross_amount = (order.limit_price * 0.001)  # 10bps aggressive
            if order.side == "BUY":
                order.limit_price += cross_amount
            else:
                order.limit_price -= cross_amount
            order.limit_price = round(order.limit_price, 2)

        result = await self.broker.place_order(order)
        return result

    async def _wait_for_fill(self, order_id: str, timeout: float) -> Optional[Order]:
        """Wait for an order to fill within timeout."""
        start = time.time()
        while time.time() - start < timeout:
            status = await self.broker.get_order_status(order_id)
            if status and status.status == OrderStatus.FILLED:
                return status
            if status and status.is_terminal:
                return None
            await asyncio.sleep(0.1)
        return None

    def _improve_price(self, order: Order, attempt: int) -> Order:
        """Improve the limit price for retry attempts."""
        if order.limit_price is None:
            return order

        improvement = order.limit_price * self.price_improve_bps / 10000 * (attempt + 1)

        if order.side == "BUY":
            order.limit_price = round(order.limit_price + improvement, 2)
        else:
            order.limit_price = round(order.limit_price - improvement, 2)

        return order
