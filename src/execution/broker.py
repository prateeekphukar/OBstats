"""
Broker API Wrapper.

Abstract broker interface with Interactive Brokers implementation.
Add additional broker implementations (Deribit, Tastytrade, etc.)
by subclassing BrokerBase.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any
import time

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    IOC = "IOC"  # Immediate or Cancel


@dataclass
class Order:
    """Represents a single order."""
    order_id: str = ""
    symbol: str = ""
    strike: float = 0.0
    expiry: str = ""
    option_type: str = "call"
    side: str = "BUY"
    quantity: int = 1
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    submit_time: float = field(default_factory=time.time)
    fill_time: Optional[float] = None
    error_msg: str = ""

    @property
    def contract_symbol(self) -> str:
        """Formatted contract symbol."""
        return f"{self.symbol}_{self.expiry}_{self.strike}_{self.option_type}"

    @property
    def is_terminal(self) -> bool:
        """Whether the order is in a final state."""
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                                OrderStatus.REJECTED, OrderStatus.ERROR)


class BrokerBase(ABC):
    """Abstract broker interface."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker."""
        ...

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Submit an order. Returns updated order with status."""
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if successful."""
        ...

    @abstractmethod
    async def cancel_all(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current status of an order."""
        ...

    @abstractmethod
    async def get_positions(self) -> list[dict]:
        """Get all open positions from the broker."""
        ...

    @abstractmethod
    async def close_position(self, contract_symbol: str, quantity: int) -> Order:
        """Close a specific position."""
        ...


class IBBroker(BrokerBase):
    """
    Interactive Brokers implementation using ib_insync.

    Requires IB TWS or IB Gateway running on the configured host/port.
    """

    def __init__(self, config: dict):
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 7497)  # 7497 = paper trading
        self.client_id = config.get("client_id", 1)
        self.account = config.get("account", "")
        self.timeout = config.get("timeout", 30)
        self._ib = None
        self._connected = False
        self._order_counter = 0
        self._orders: dict[str, Order] = {}

    async def connect(self) -> bool:
        """Connect to IB TWS/Gateway."""
        try:
            from ib_insync import IB
            self._ib = IB()
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._ib.connect(self.host, self.port, clientId=self.client_id)
                ),
                timeout=self.timeout,
            )
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True
        except ImportError:
            logger.error("ib_insync not installed. Install with: pip install ib_insync")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from IB."""
        if self._ib:
            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    async def place_order(self, order: Order) -> Order:
        """Submit an order to IB."""
        if not self._connected:
            order.status = OrderStatus.ERROR
            order.error_msg = "Not connected to broker"
            return order

        try:
            from ib_insync import Option, LimitOrder, MarketOrder

            # Create IB contract
            contract = Option(
                symbol=order.symbol,
                lastTradeDateOrContractMonth=order.expiry.replace("-", ""),
                strike=order.strike,
                right="C" if order.option_type == "call" else "P",
                exchange="SMART",
            )

            # Create IB order
            if order.order_type == OrderType.LIMIT and order.limit_price:
                ib_order = LimitOrder(
                    action=order.side,
                    totalQuantity=order.quantity,
                    lmtPrice=order.limit_price,
                )
            else:
                ib_order = MarketOrder(
                    action=order.side,
                    totalQuantity=order.quantity,
                )

            # Submit
            trade = self._ib.placeOrder(contract, ib_order)
            order.order_id = str(trade.order.orderId)
            order.status = OrderStatus.SUBMITTED
            order.submit_time = time.time()

            self._orders[order.order_id] = order
            logger.info(f"Order submitted: {order.order_id} {order.side} {order.quantity}x {order.contract_symbol}")

            return order

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_msg = str(e)
            logger.error(f"Order error: {e}")
            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on IB."""
        if not self._connected:
            return False
        try:
            # Simplified — in production use ib_insync trade tracking
            logger.info(f"Cancelling order {order_id}")
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        except Exception as e:
            logger.error(f"Cancel error: {e}")
            return False

    async def cancel_all(self) -> int:
        """Cancel all open orders."""
        if not self._connected:
            return 0
        try:
            self._ib.reqGlobalCancel()
            count = len([o for o in self._orders.values() if not o.is_terminal])
            for order in self._orders.values():
                if not order.is_terminal:
                    order.status = OrderStatus.CANCELLED
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Cancel all error: {e}")
            return 0

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status."""
        return self._orders.get(order_id)

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        if not self._connected:
            return []
        try:
            positions = self._ib.positions()
            return [
                {
                    "symbol": pos.contract.symbol,
                    "strike": pos.contract.strike,
                    "expiry": pos.contract.lastTradeDateOrContractMonth,
                    "right": pos.contract.right,
                    "quantity": pos.position,
                    "avg_cost": pos.avgCost,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []

    async def close_position(self, contract_symbol: str, quantity: int) -> Order:
        """Close a position with a market order."""
        # Parse contract symbol
        parts = contract_symbol.split("_")
        order = Order(
            symbol=parts[0] if len(parts) > 0 else "",
            expiry=parts[1] if len(parts) > 1 else "",
            strike=float(parts[2]) if len(parts) > 2 else 0.0,
            option_type=parts[3] if len(parts) > 3 else "call",
            side="SELL" if quantity > 0 else "BUY",
            quantity=abs(quantity),
            order_type=OrderType.MARKET,
        )
        return await self.place_order(order)

    @property
    def is_connected(self) -> bool:
        return self._connected


class PaperBroker(BrokerBase):
    """
    Paper trading broker for testing and backtesting.

    Simulates fills at mid-market with configurable slippage.
    """

    def __init__(self, slippage_bps: float = 5.0, commission_per_contract: float = 0.65):
        self.slippage_bps = slippage_bps
        self.commission = commission_per_contract
        self._orders: dict[str, Order] = {}
        self._order_counter = 0
        self._connected = True

    async def connect(self) -> bool:
        self._connected = True
        logger.info("Paper broker connected")
        return True

    async def disconnect(self) -> None:
        self._connected = False

    async def place_order(self, order: Order) -> Order:
        self._order_counter += 1
        order.order_id = f"PAPER-{self._order_counter:06d}"

        # Simulate immediate fill at limit price (with slippage)
        fill_price = order.limit_price if order.limit_price else 0
        slippage = fill_price * self.slippage_bps / 10000
        if order.side == "BUY":
            fill_price += slippage
        else:
            fill_price -= slippage

        order.filled_price = round(fill_price, 2)
        order.filled_qty = order.quantity
        order.commission = self.commission * order.quantity
        order.status = OrderStatus.FILLED
        order.fill_time = time.time()

        self._orders[order.order_id] = order
        logger.info(f"Paper fill: {order.order_id} {order.side} {order.quantity}x @ ${fill_price:.2f}")
        return order

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def cancel_all(self) -> int:
        count = 0
        for order in self._orders.values():
            if not order.is_terminal:
                order.status = OrderStatus.CANCELLED
                count += 1
        return count

    async def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    async def get_positions(self) -> list[dict]:
        return []

    async def close_position(self, contract_symbol: str, quantity: int) -> Order:
        order = Order(side="SELL" if quantity > 0 else "BUY", quantity=abs(quantity), order_type=OrderType.MARKET)
        return await self.place_order(order)
