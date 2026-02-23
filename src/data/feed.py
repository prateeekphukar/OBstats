"""
WebSocket Data Feed Handler.

Handles real-time streaming of options market data via WebSocket connections.
Supports automatic reconnection, heartbeating, and data normalization.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional, Any
from dataclasses import dataclass

import websockets
from websockets.exceptions import ConnectionClosed

from .chain import OptionsChain, OptionTick, UnderlyingTick, OptionType

logger = logging.getLogger(__name__)


@dataclass
class FeedConfig:
    """Configuration for the data feed connection."""
    url: str = "ws://localhost:7497"
    symbols: list[str] = None
    reconnect_delay: float = 5.0
    heartbeat_interval: float = 30.0
    max_reconnect_attempts: int = 10

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SPY"]


class DataFeed:
    """
    Async WebSocket data feed handler for options market data.

    Connects to a market data provider, normalizes incoming data,
    and updates the OptionsChain objects in real-time.

    Usage:
        feed = DataFeed(config, chains, on_tick=my_callback)
        await feed.start()
    """

    def __init__(
        self,
        config: FeedConfig,
        chains: dict[str, OptionsChain],
        on_tick: Optional[Callable[[OptionTick], Any]] = None,
        on_underlying: Optional[Callable[[UnderlyingTick], Any]] = None,
    ):
        self.config = config
        self.chains = chains
        self.on_tick = on_tick
        self.on_underlying = on_underlying
        self._ws: Optional[Any] = None
        self._running = False
        self._reconnect_count = 0
        self._stats = {
            "ticks_received": 0,
            "errors": 0,
            "reconnects": 0,
            "last_tick_time": 0,
        }

    async def start(self) -> None:
        """Start the data feed with automatic reconnection."""
        self._running = True
        logger.info(f"Starting data feed for symbols: {self.config.symbols}")

        while self._running and self._reconnect_count < self.config.max_reconnect_attempts:
            try:
                await self._connect_and_stream()
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self._stats["reconnects"] += 1
                self._reconnect_count += 1
                if self._running:
                    await asyncio.sleep(self.config.reconnect_delay)
            except Exception as e:
                logger.error(f"Feed error: {e}", exc_info=True)
                self._stats["errors"] += 1
                self._reconnect_count += 1
                if self._running:
                    await asyncio.sleep(self.config.reconnect_delay)

        if self._reconnect_count >= self.config.max_reconnect_attempts:
            logger.critical("Max reconnect attempts reached — feed stopped")

    async def stop(self) -> None:
        """Stop the data feed."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("Data feed stopped")

    async def _connect_and_stream(self) -> None:
        """Connect to WebSocket and start streaming."""
        async with websockets.connect(self.config.url) as ws:
            self._ws = ws
            self._reconnect_count = 0
            logger.info(f"Connected to {self.config.url}")

            # Subscribe to symbols
            await self._subscribe(ws)

            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))

            try:
                async for message in ws:
                    if not self._running:
                        break
                    await self._handle_message(message)
            finally:
                heartbeat_task.cancel()

    async def _subscribe(self, ws) -> None:
        """Send subscription request for configured symbols."""
        subscribe_msg = json.dumps({
            "action": "subscribe",
            "params": {
                "symbols": self.config.symbols,
                "channels": ["options", "quotes"],
            }
        })
        await ws.send(subscribe_msg)
        logger.info(f"Subscribed to: {self.config.symbols}")

    async def _heartbeat(self, ws) -> None:
        """Send periodic heartbeats to keep the connection alive."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                await ws.ping()
            except Exception:
                break

    async def _handle_message(self, raw_message: str) -> None:
        """Parse and route incoming messages."""
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type", "")

            if msg_type == "option_quote":
                tick = self._parse_option_tick(data)
                if tick:
                    self._update_chain(tick)
                    if self.on_tick:
                        await self._maybe_await(self.on_tick(tick))

            elif msg_type == "underlying_quote":
                tick = self._parse_underlying_tick(data)
                if tick:
                    symbol = tick.symbol
                    if symbol in self.chains:
                        self.chains[symbol].update_underlying(tick)
                    if self.on_underlying:
                        await self._maybe_await(self.on_underlying(tick))

            elif msg_type == "heartbeat":
                pass  # Connection alive

            elif msg_type == "error":
                logger.error(f"Feed error message: {data}")
                self._stats["errors"] += 1

            self._stats["ticks_received"] += 1
            self._stats["last_tick_time"] = time.time()

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {raw_message[:100]}")
            self._stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            self._stats["errors"] += 1

    def _parse_option_tick(self, data: dict) -> Optional[OptionTick]:
        """Parse raw data into an OptionTick."""
        try:
            return OptionTick(
                symbol=data["symbol"],
                strike=float(data["strike"]),
                expiry=data["expiry"],
                option_type=OptionType(data["option_type"]),
                bid=float(data.get("bid", 0)),
                ask=float(data.get("ask", 0)),
                last=float(data.get("last", 0)),
                bid_size=int(data.get("bid_size", 0)),
                ask_size=int(data.get("ask_size", 0)),
                volume=int(data.get("volume", 0)),
                open_interest=int(data.get("open_interest", 0)),
                timestamp=float(data.get("timestamp", time.time())),
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse option tick: {e}")
            return None

    def _parse_underlying_tick(self, data: dict) -> Optional[UnderlyingTick]:
        """Parse raw data into an UnderlyingTick."""
        try:
            return UnderlyingTick(
                symbol=data["symbol"],
                price=float(data["price"]),
                bid=float(data.get("bid", 0)),
                ask=float(data.get("ask", 0)),
                volume=int(data.get("volume", 0)),
                timestamp=float(data.get("timestamp", time.time())),
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse underlying tick: {e}")
            return None

    def _update_chain(self, tick: OptionTick) -> None:
        """Update the appropriate options chain."""
        symbol = tick.symbol
        if symbol not in self.chains:
            self.chains[symbol] = OptionsChain(symbol)
        self.chains[symbol].update_option(tick)

    @staticmethod
    async def _maybe_await(result):
        """Await coroutine if necessary."""
        if asyncio.iscoroutine(result):
            await result

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running
