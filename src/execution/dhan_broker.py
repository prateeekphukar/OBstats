"""
Dhan Broker API Integration.

Implements the BrokerBase interface for Dhan (Indian broker).
Uses the `dhanhq` Python SDK for order management, positions,
and option chain data.

Dhan API Docs: https://dhanhq.co/docs/v2/
Package: pip install dhanhq
"""

import asyncio
import logging
from typing import Optional
from functools import partial
import time

from .broker import BrokerBase, Order, OrderType, OrderStatus

logger = logging.getLogger(__name__)

# ── Dhan Exchange Segment Mapping ─────────────────────

DHAN_EXCHANGE_SEGMENTS = {
    "NSE": "NSE_EQ",
    "BSE": "BSE_EQ",
    "NSE_FNO": "NSE_FNO",     # NSE Futures & Options
    "BSE_FNO": "BSE_FNO",     # BSE Futures & Options
    "MCX": "MCX_COMM",         # MCX Commodities
    "NSE_CURRENCY": "NSE_CURRENCY",
}

# ── Dhan Order Status Mapping ─────────────────────────

DHAN_STATUS_MAP = {
    "TRANSIT": OrderStatus.SUBMITTED,
    "PENDING": OrderStatus.PENDING,
    "REJECTED": OrderStatus.REJECTED,
    "CANCELLED": OrderStatus.CANCELLED,
    "TRADED": OrderStatus.FILLED,
    "EXPIRED": OrderStatus.CANCELLED,
}


class DhanBroker(BrokerBase):
    """
    Dhan broker implementation using the dhanhq Python SDK.

    Supports NSE/BSE equity and F&O segments.
    Requires a Dhan trading account with API access enabled.

    Setup:
        1. Login to https://web.dhan.co
        2. Go to My Profile → DhanHQ Trading APIs
        3. Generate your access token
        4. Enter client_id and access_token in config/settings.yaml
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Broker config with keys:
                - client_id: Dhan client ID
                - access_token: Dhan API access token
                - exchange_segment: Default segment (default: NSE_FNO)
                - product_type: Default product type (default: INTRA)
        """
        self.client_id = config.get("client_id", "")
        self.access_token = config.get("access_token", "")
        self.exchange_segment = config.get("exchange_segment", "NSE_FNO")
        self.product_type = config.get("product_type", "INTRA")
        self._dhan = None
        self._connected = False
        self._orders: dict[str, Order] = {}
        self._security_list = None  # Cached instrument list

    async def connect(self) -> bool:
        """Connect to Dhan API."""
        try:
            from dhanhq import dhanhq

            if not self.client_id or not self.access_token:
                logger.error(
                    "Dhan client_id and access_token are required. "
                    "Set them in config/settings.yaml under broker section."
                )
                return False

            # dhanhq is synchronous — run in executor
            loop = asyncio.get_event_loop()
            self._dhan = await loop.run_in_executor(
                None,
                partial(dhanhq, self.client_id, self.access_token)
            )

            self._connected = True
            logger.info(f"Connected to Dhan API (client: {self.client_id})")

            # Optionally cache security list for faster lookups
            try:
                self._security_list = await loop.run_in_executor(
                    None,
                    partial(self._dhan.fetch_security_list, "compact")
                )
                logger.info("Security list cached")
            except Exception as e:
                logger.warning(f"Could not cache security list: {e}")

            return True

        except ImportError:
            logger.error(
                "dhanhq package not installed. Install with: pip install dhanhq"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Dhan: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Dhan."""
        self._connected = False
        self._dhan = None
        logger.info("Disconnected from Dhan API")

    async def place_order(self, order: Order) -> Order:
        """
        Submit an order to Dhan.

        Dhan uses security_id for instrument identification. The order's
        `symbol` field should contain the security_id, or the `metadata`
        dict should have a `security_id` key.
        """
        if not self._connected or not self._dhan:
            order.status = OrderStatus.ERROR
            order.error_msg = "Not connected to Dhan broker"
            return order

        try:
            loop = asyncio.get_event_loop()

            # Determine order type for Dhan
            dhan_order_type = self._map_order_type(order.order_type)
            dhan_segment = self.exchange_segment

            # Determine transaction type
            transaction_type = "BUY" if order.side == "BUY" else "SELL"

            # Security ID — stored in order.symbol or constructed
            security_id = str(order.symbol)

            # Price handling
            price = order.limit_price if order.limit_price and order.order_type == OrderType.LIMIT else 0

            # Validity
            validity = "IOC" if order.order_type == OrderType.IOC else "DAY"

            # Place order via dhanhq SDK (synchronous → run in executor)
            response = await loop.run_in_executor(
                None,
                partial(
                    self._dhan.place_order,
                    security_id=security_id,
                    exchange_segment=dhan_segment,
                    transaction_type=transaction_type,
                    quantity=order.quantity,
                    order_type=dhan_order_type,
                    product_type=self.product_type,
                    price=price,
                    validity=validity,
                )
            )

            # Parse response
            if response and "orderId" in response:
                order.order_id = str(response["orderId"])
                dhan_status = response.get("orderStatus", "PENDING")
                order.status = DHAN_STATUS_MAP.get(dhan_status, OrderStatus.PENDING)
                order.submit_time = time.time()
                self._orders[order.order_id] = order
                logger.info(
                    f"Dhan order placed: {order.order_id} "
                    f"{transaction_type} {order.quantity}x {security_id} "
                    f"@ ₹{price} [{dhan_status}]"
                )
            elif response and "errorCode" in response:
                order.status = OrderStatus.REJECTED
                order.error_msg = response.get("errorMessage", str(response))
                logger.warning(f"Dhan order rejected: {order.error_msg}")
            else:
                order.status = OrderStatus.ERROR
                order.error_msg = f"Unexpected response: {response}"
                logger.error(f"Dhan unexpected response: {response}")

            return order

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_msg = str(e)
            logger.error(f"Dhan place_order error: {e}", exc_info=True)
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on Dhan."""
        if not self._connected or not self._dhan:
            return False
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(self._dhan.cancel_order, order_id)
            )

            if response and response.get("orderStatus") == "CANCELLED":
                if order_id in self._orders:
                    self._orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"Dhan order cancelled: {order_id}")
                return True
            else:
                logger.warning(f"Dhan cancel failed for {order_id}: {response}")
                return False

        except Exception as e:
            logger.error(f"Dhan cancel error: {e}")
            return False

    async def cancel_all(self) -> int:
        """Cancel all pending orders on Dhan."""
        if not self._connected or not self._dhan:
            return 0

        cancelled = 0
        try:
            loop = asyncio.get_event_loop()

            # Get order book to find pending orders
            order_list = await loop.run_in_executor(
                None, self._dhan.get_order_list
            )

            if not order_list or not isinstance(order_list, list):
                return 0

            for order_data in order_list:
                status = order_data.get("orderStatus", "")
                if status in ("PENDING", "TRANSIT"):
                    oid = str(order_data.get("orderId", ""))
                    if oid:
                        success = await self.cancel_order(oid)
                        if success:
                            cancelled += 1

            logger.info(f"Cancelled {cancelled} orders on Dhan")
            return cancelled

        except Exception as e:
            logger.error(f"Dhan cancel_all error: {e}")
            return cancelled

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status from Dhan."""
        if not self._connected or not self._dhan:
            return self._orders.get(order_id)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(self._dhan.get_order_by_id, order_id)
            )

            if response and "orderId" in response:
                order = self._orders.get(order_id, Order(order_id=order_id))
                dhan_status = response.get("orderStatus", "PENDING")
                order.status = DHAN_STATUS_MAP.get(dhan_status, OrderStatus.PENDING)

                if order.status == OrderStatus.FILLED:
                    order.filled_qty = int(response.get("filledQty", response.get("quantity", 0)))
                    order.filled_price = float(response.get("price", 0))
                    order.fill_time = time.time()

                self._orders[order_id] = order
                return order

            return self._orders.get(order_id)

        except Exception as e:
            logger.error(f"Dhan get_order_status error: {e}")
            return self._orders.get(order_id)

    async def get_positions(self) -> list[dict]:
        """Get all open positions from Dhan."""
        if not self._connected or not self._dhan:
            return []

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._dhan.get_positions
            )

            if not response or not isinstance(response, list):
                return []

            positions = []
            for pos in response:
                positions.append({
                    "security_id": pos.get("securityId", ""),
                    "symbol": pos.get("tradingSymbol", ""),
                    "exchange_segment": pos.get("exchangeSegment", ""),
                    "product_type": pos.get("productType", ""),
                    "quantity": int(pos.get("netQty", 0)),
                    "buy_avg": float(pos.get("buyAvg", 0)),
                    "sell_avg": float(pos.get("sellAvg", 0)),
                    "realized_profit": float(pos.get("realizedProfit", 0)),
                    "unrealized_profit": float(pos.get("unrealizedProfit", 0)),
                    "day_buy_qty": int(pos.get("dayBuyQty", 0)),
                    "day_sell_qty": int(pos.get("daySellQty", 0)),
                })

            return positions

        except Exception as e:
            logger.error(f"Dhan get_positions error: {e}")
            return []

    async def close_position(self, contract_symbol: str, quantity: int) -> Order:
        """Close a position with a market order on Dhan."""
        order = Order(
            symbol=contract_symbol,
            side="SELL" if quantity > 0 else "BUY",
            quantity=abs(quantity),
            order_type=OrderType.MARKET,
        )
        return await self.place_order(order)

    # ── Dhan-Specific Methods ─────────────────────────

    async def get_option_chain(
        self,
        underlying_security_id: int,
        underlying_segment: str = "IDX_I",
        expiry: str = "",
    ) -> dict:
        """
        Fetch the option chain from Dhan.

        Args:
            underlying_security_id: Security ID of the underlying (e.g., 13 for NIFTY).
            underlying_segment: Segment (IDX_I for index, NSE_EQ for equity).
            expiry: Expiry date string (e.g., "2025-03-27").

        Returns:
            Dict with option chain data including OI, Greeks, volume, prices.
        """
        if not self._connected or not self._dhan:
            return {}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self._dhan.option_chain,
                    under_security_id=underlying_security_id,
                    under_exchange_segment=underlying_segment,
                    expiry=expiry,
                )
            )
            return response if response else {}

        except Exception as e:
            logger.error(f"Dhan option_chain error: {e}")
            return {}

    async def get_expiry_list(
        self,
        underlying_security_id: int,
        underlying_segment: str = "IDX_I",
    ) -> list:
        """Get available expiry dates for an underlying."""
        if not self._connected or not self._dhan:
            return []

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(
                    self._dhan.expiry_list,
                    under_security_id=underlying_security_id,
                    under_exchange_segment=underlying_segment,
                )
            )
            return response if response else []

        except Exception as e:
            logger.error(f"Dhan expiry_list error: {e}")
            return []

    async def get_market_quote(self, securities: dict) -> dict:
        """
        Get market quotes from Dhan.

        Args:
            securities: Dict of {exchange_segment: [security_id_list]}.
                        e.g., {"NSE_FNO": ["52175", "52176"]}

        Returns:
            Dict with market quote data.
        """
        if not self._connected or not self._dhan:
            return {}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                partial(self._dhan.quote_data, securities=securities)
            )
            return response if response else {}

        except Exception as e:
            logger.error(f"Dhan market_quote error: {e}")
            return {}

    async def get_fund_limits(self) -> dict:
        """Get account fund limits and available margin."""
        if not self._connected or not self._dhan:
            return {}

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._dhan.get_fund_limits
            )
            return response if response else {}

        except Exception as e:
            logger.error(f"Dhan fund_limits error: {e}")
            return {}

    # ── Helpers ────────────────────────────────────────

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        """Map internal OrderType to Dhan order type string."""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.IOC: "LIMIT",   # IOC is a validity type in Dhan, not order type
        }
        return mapping.get(order_type, "MARKET")

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Common Underlying Security IDs (NSE India) ───────

DHAN_UNDERLYING_IDS = {
    "NIFTY": 13,
    "BANKNIFTY": 25,
    "FINNIFTY": 27,
    "MIDCPNIFTY": 442,
    "SENSEX": 51,
    "BANKEX": 52,
}
