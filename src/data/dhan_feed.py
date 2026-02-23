"""
Dhan Market Data Feed.

Real-time market data feed using Dhan's marketfeed WebSocket API.
Streams live prices, quotes, and depth data for options and equities
on NSE/BSE.

Requires: pip install dhanhq
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass

from ..data.chain import OptionsChain, OptionTick, UnderlyingTick, OptionType

logger = logging.getLogger(__name__)


@dataclass
class DhanFeedConfig:
    """Configuration for Dhan market data feed."""
    client_id: str = ""
    access_token: str = ""
    version: str = "v2"
    subscription_type: str = "Full"   # Ticker, Quote, or Full
    poll_interval_ms: int = 200       # How often to poll for data


# ── Dhan Exchange Segment Constants ───────────────────
# These map to the marketfeed module constants

SEGMENT_MAP = {
    "NSE": "NSE",
    "BSE": "BSE",
    "NSE_FNO": "NSE_FNO",
    "BSE_FNO": "BSE_FNO",
    "MCX": "MCX",
    "IDX": "IDX",
}


class DhanDataFeed:
    """
    Real-time market data feed using Dhan's WebSocket API.

    Uses the dhanhq marketfeed module to stream live prices.
    Converts incoming data to the bot's internal OptionTick format
    and updates the OptionsChain objects.

    Usage:
        feed = DhanDataFeed(config, chains, on_tick=my_callback)
        await feed.start()
    """

    def __init__(
        self,
        config: DhanFeedConfig,
        chains: dict[str, OptionsChain],
        instruments: list[dict] | None = None,
        on_tick: Optional[Callable[[OptionTick], Any]] = None,
        on_underlying: Optional[Callable[[UnderlyingTick], Any]] = None,
    ):
        """
        Args:
            config: Dhan feed configuration.
            chains: Dict of symbol → OptionsChain to update.
            instruments: List of instrument dicts with keys:
                         {security_id, exchange_segment, symbol, strike, expiry, option_type}
            on_tick: Callback for each option tick.
            on_underlying: Callback for each underlying tick.
        """
        self.config = config
        self.chains = chains
        self.instruments = instruments or []
        self.on_tick = on_tick
        self.on_underlying = on_underlying
        self._feed = None
        self._running = False
        self._stats = {
            "ticks_received": 0,
            "errors": 0,
            "last_tick_time": 0,
        }
        # Map security_id → instrument info for data parsing
        self._instrument_map: dict[str, dict] = {}
        for inst in self.instruments:
            self._instrument_map[str(inst.get("security_id", ""))] = inst

    async def start(self) -> None:
        """Start the Dhan market data feed."""
        self._running = True
        logger.info("Starting Dhan market data feed")

        try:
            from dhanhq import marketfeed
        except ImportError:
            logger.error("dhanhq not installed. Run: pip install dhanhq")
            return

        if not self.config.client_id or not self.config.access_token:
            logger.error("Dhan client_id and access_token required for market feed")
            return

        # Build instrument subscription list
        # Format: (exchange_segment, "security_id", subscription_type)
        sub_type_map = {
            "Ticker": marketfeed.Ticker,
            "Quote": marketfeed.Quote,
            "Full": marketfeed.Full,
        }
        sub_type = sub_type_map.get(self.config.subscription_type, marketfeed.Full)

        feed_instruments = []
        for inst in self.instruments:
            segment = self._get_segment_constant(marketfeed, inst.get("exchange_segment", "NSE_FNO"))
            sec_id = str(inst.get("security_id", ""))
            if segment is not None and sec_id:
                feed_instruments.append((segment, sec_id, sub_type))

        if not feed_instruments:
            logger.warning("No instruments to subscribe to")
            return

        logger.info(f"Subscribing to {len(feed_instruments)} instruments on Dhan")

        # Create the DhanFeed instance
        try:
            self._feed = marketfeed.DhanFeed(
                self.config.client_id,
                self.config.access_token,
                feed_instruments,
                self.config.version,
            )
        except Exception as e:
            logger.error(f"Failed to create DhanFeed: {e}")
            return

        # Run the feed in a background thread (it's synchronous/blocking)
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Run one cycle of the feed
                await loop.run_in_executor(None, self._feed.run_forever)

                # Get latest data
                data = await loop.run_in_executor(None, self._feed.get_data)

                if data:
                    await self._handle_data(data)

                # Small sleep to control polling rate
                await asyncio.sleep(self.config.poll_interval_ms / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dhan feed error: {e}", exc_info=True)
                self._stats["errors"] += 1
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the Dhan market data feed."""
        self._running = False
        if self._feed:
            try:
                self._feed.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting Dhan feed: {e}")
        logger.info("Dhan market data feed stopped")

    async def subscribe(self, instruments: list[dict]) -> None:
        """Subscribe to additional instruments while feed is running."""
        if not self._feed:
            return

        try:
            from dhanhq import marketfeed
            sub_type_map = {
                "Ticker": marketfeed.Ticker,
                "Quote": marketfeed.Quote,
                "Full": marketfeed.Full,
            }
            sub_type = sub_type_map.get(self.config.subscription_type, marketfeed.Full)

            new_subs = []
            for inst in instruments:
                segment = self._get_segment_constant(marketfeed, inst.get("exchange_segment", "NSE_FNO"))
                sec_id = str(inst.get("security_id", ""))
                if segment is not None and sec_id:
                    new_subs.append((segment, sec_id, sub_type))
                    self._instrument_map[sec_id] = inst

            if new_subs:
                self._feed.subscribe_symbols(new_subs)
                logger.info(f"Subscribed to {len(new_subs)} additional instruments")

        except Exception as e:
            logger.error(f"Error subscribing to instruments: {e}")

    async def unsubscribe(self, security_ids: list[str]) -> None:
        """Unsubscribe from instruments."""
        if not self._feed:
            return

        try:
            from dhanhq import marketfeed
            unsub = [(marketfeed.NSE_FNO, sid, marketfeed.Full) for sid in security_ids]
            self._feed.unsubscribe_symbols(unsub)
            logger.info(f"Unsubscribed from {len(security_ids)} instruments")
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")

    async def _handle_data(self, data: dict) -> None:
        """Process incoming market data from Dhan feed."""
        try:
            sec_id = str(data.get("security_id", data.get("securityId", "")))
            inst_info = self._instrument_map.get(sec_id, {})

            if not inst_info:
                # Unknown instrument — might be underlying
                return

            ltp = float(data.get("LTP", data.get("ltp", 0)))
            if ltp <= 0:
                return

            # Determine if this is an option or underlying
            option_type_val = inst_info.get("option_type", "")

            if option_type_val in ("call", "put", "CE", "PE"):
                # It's an option tick
                tick = self._parse_option_data(data, inst_info)
                if tick:
                    self._update_chain(tick)
                    if self.on_tick:
                        result = self.on_tick(tick)
                        if asyncio.iscoroutine(result):
                            await result
            else:
                # It's an underlying tick
                tick = self._parse_underlying_data(data, inst_info)
                if tick:
                    symbol = tick.symbol
                    if symbol in self.chains:
                        self.chains[symbol].update_underlying(tick)
                    if self.on_underlying:
                        result = self.on_underlying(tick)
                        if asyncio.iscoroutine(result):
                            await result

            self._stats["ticks_received"] += 1
            self._stats["last_tick_time"] = time.time()

        except Exception as e:
            logger.error(f"Error handling Dhan data: {e}", exc_info=True)
            self._stats["errors"] += 1

    def _parse_option_data(self, data: dict, inst_info: dict) -> Optional[OptionTick]:
        """Parse Dhan feed data into an OptionTick."""
        try:
            ltp = float(data.get("LTP", data.get("ltp", 0)))
            # Dhan Full packet includes: LTP, open, high, low, close, volume,
            # avg_price, oi, total_buy_qty, total_sell_qty, bid, ask, etc.

            # Map option type
            opt_type_raw = inst_info.get("option_type", "call")
            if opt_type_raw in ("CE", "call"):
                opt_type = OptionType.CALL
            else:
                opt_type = OptionType.PUT

            bid = float(data.get("bid_price", data.get("best_bid_price", ltp * 0.99)))
            ask = float(data.get("ask_price", data.get("best_ask_price", ltp * 1.01)))

            return OptionTick(
                symbol=str(inst_info.get("symbol", "")),
                strike=float(inst_info.get("strike", 0)),
                expiry=str(inst_info.get("expiry", "")),
                option_type=opt_type,
                bid=round(bid, 2),
                ask=round(ask, 2),
                last=round(ltp, 2),
                bid_size=int(data.get("total_buy_qty", 0)),
                ask_size=int(data.get("total_sell_qty", 0)),
                volume=int(data.get("volume", data.get("vol", 0))),
                open_interest=int(data.get("oi", data.get("OI", 0))),
                timestamp=time.time(),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse Dhan option data: {e}")
            return None

    def _parse_underlying_data(self, data: dict, inst_info: dict) -> Optional[UnderlyingTick]:
        """Parse Dhan feed data into an UnderlyingTick."""
        try:
            ltp = float(data.get("LTP", data.get("ltp", 0)))
            bid = float(data.get("bid_price", data.get("best_bid_price", ltp)))
            ask = float(data.get("ask_price", data.get("best_ask_price", ltp)))

            return UnderlyingTick(
                symbol=str(inst_info.get("symbol", "")),
                price=round(ltp, 2),
                bid=round(bid, 2),
                ask=round(ask, 2),
                volume=int(data.get("volume", data.get("vol", 0))),
                timestamp=time.time(),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse Dhan underlying data: {e}")
            return None

    def _update_chain(self, tick: OptionTick) -> None:
        """Update the appropriate options chain."""
        symbol = tick.symbol
        if symbol not in self.chains:
            self.chains[symbol] = OptionsChain(symbol)
        self.chains[symbol].update_option(tick)

    @staticmethod
    def _get_segment_constant(marketfeed_module, segment: str):
        """Get the marketfeed segment constant from string name."""
        segment_attr_map = {
            "NSE": "NSE",
            "NSE_EQ": "NSE",
            "BSE": "BSE",
            "BSE_EQ": "BSE",
            "NSE_FNO": "NSE_FNO",
            "BSE_FNO": "BSE_FNO",
            "MCX": "MCX",
            "IDX_I": "IDX",
        }
        attr_name = segment_attr_map.get(segment, segment)
        return getattr(marketfeed_module, attr_name, None)

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    @property
    def is_connected(self) -> bool:
        return self._running and self._feed is not None
