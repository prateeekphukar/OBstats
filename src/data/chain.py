"""
Options Chain Data Model.

Defines the core data structures for representing options ticks,
chains, and related market data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OptionTick:
    """A single options market data tick."""
    symbol: str              # Underlying symbol (e.g., "SPY")
    strike: float
    expiry: str              # ISO date string (e.g., "2025-03-21")
    option_type: OptionType
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    open_interest: int = 0
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        mid = self.mid
        return (self.spread / mid * 100) if mid > 0 else float("inf")

    @property
    def contract_id(self) -> str:
        """Unique identifier for this contract."""
        return f"{self.symbol}_{self.expiry}_{self.strike}_{self.option_type.value}"


@dataclass
class UnderlyingTick:
    """A single underlying asset tick."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int = 0
    timestamp: float = field(default_factory=time.time)


class OptionsChain:
    """
    Manages an options chain for a single underlying symbol.

    Stores the latest ticks for all strikes and expirations,
    and provides methods for querying and filtering.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.underlying: Optional[UnderlyingTick] = None
        self._chain: dict[str, OptionTick] = {}  # contract_id -> OptionTick
        self._last_update: float = 0

    def update_underlying(self, tick: UnderlyingTick) -> None:
        """Update the underlying price."""
        self.underlying = tick
        self._last_update = tick.timestamp

    def update_option(self, tick: OptionTick) -> None:
        """Add or update an option tick in the chain."""
        self._chain[tick.contract_id] = tick
        self._last_update = tick.timestamp

    def update_options_batch(self, ticks: list[OptionTick]) -> None:
        """Update multiple option ticks."""
        for tick in ticks:
            self.update_option(tick)

    def get_tick(self, strike: float, expiry: str, option_type: OptionType) -> Optional[OptionTick]:
        """Get a specific option tick."""
        key = f"{self.symbol}_{expiry}_{strike}_{option_type.value}"
        return self._chain.get(key)

    def get_calls(self, expiry: Optional[str] = None) -> list[OptionTick]:
        """Get all call options, optionally filtered by expiry."""
        ticks = [t for t in self._chain.values() if t.option_type == OptionType.CALL]
        if expiry:
            ticks = [t for t in ticks if t.expiry == expiry]
        return sorted(ticks, key=lambda t: t.strike)

    def get_puts(self, expiry: Optional[str] = None) -> list[OptionTick]:
        """Get all put options, optionally filtered by expiry."""
        ticks = [t for t in self._chain.values() if t.option_type == OptionType.PUT]
        if expiry:
            ticks = [t for t in ticks if t.expiry == expiry]
        return sorted(ticks, key=lambda t: t.strike)

    def get_by_expiry(self, expiry: str) -> list[OptionTick]:
        """Get all options for a specific expiration."""
        return sorted(
            [t for t in self._chain.values() if t.expiry == expiry],
            key=lambda t: (t.strike, t.option_type.value),
        )

    def get_atm_strike(self) -> Optional[float]:
        """Find the at-the-money strike (closest to underlying price)."""
        if not self.underlying or not self._chain:
            return None
        strikes = set(t.strike for t in self._chain.values())
        return min(strikes, key=lambda s: abs(s - self.underlying.price))

    def get_expirations(self) -> list[str]:
        """Get all available expirations, sorted."""
        return sorted(set(t.expiry for t in self._chain.values()))

    def get_strikes(self, expiry: Optional[str] = None) -> list[float]:
        """Get all available strikes, optionally for a specific expiry."""
        ticks = self._chain.values()
        if expiry:
            ticks = [t for t in ticks if t.expiry == expiry]
        return sorted(set(t.strike for t in ticks))

    def filter_liquid(self, min_volume: int = 100, min_oi: int = 500) -> list[OptionTick]:
        """Filter options by liquidity thresholds."""
        return [
            t for t in self._chain.values()
            if t.volume >= min_volume and t.open_interest >= min_oi
        ]

    @property
    def size(self) -> int:
        """Number of option contracts in the chain."""
        return len(self._chain)

    @property
    def all_ticks(self) -> list[OptionTick]:
        """All option ticks."""
        return list(self._chain.values())

    @property
    def last_update(self) -> float:
        """Timestamp of last update."""
        return self._last_update
