"""
Signal Filters.

Applies liquidity, strength, time-of-day, and correlation filters
to raw signals before they are passed to the execution engine.
"""

import logging
from datetime import datetime
from typing import Optional

from .signals import Signal, SignalType

logger = logging.getLogger(__name__)


class SignalFilter:
    """
    Multi-stage signal filter pipeline.

    Filters raw signals for quality, liquidity, and timing
    before execution.
    """

    def __init__(self, config: dict):
        self.config = config
        self.min_volume = config.get("min_option_volume", 100)
        self.min_oi = config.get("min_open_interest", 500)
        self.min_strength = config.get("min_signal_strength", 1.2)
        self.max_spread_pct = config.get("max_spread_pct", 5.0)  # Max 5% spread
        self._recent_signals: list[Signal] = []

    def apply(self, signals: list[Signal], chain=None) -> list[Signal]:
        """
        Apply all filters sequentially.

        Args:
            signals: Raw signals from the generator.
            chain: Optional OptionsChain for liquidity checks.

        Returns:
            Filtered and ranked signals.
        """
        filtered = signals

        # Stage 1: Strength filter
        filtered = self._filter_strength(filtered)

        # Stage 2: Liquidity filter
        if chain:
            filtered = self._filter_liquidity(filtered, chain)

        # Stage 3: Time-of-day filter
        filtered = self._filter_time_of_day(filtered)

        # Stage 4: Dedup / correlation filter
        filtered = self._filter_correlated(filtered)

        # Stage 5: Spread filter
        if chain:
            filtered = self._filter_spread(filtered, chain)

        # Update recent signals for correlation tracking
        self._recent_signals = filtered[:20]

        logger.debug(f"Filtered {len(signals)} signals → {len(filtered)} remaining")
        return filtered

    def _filter_strength(self, signals: list[Signal]) -> list[Signal]:
        """Keep only signals above the minimum strength threshold."""
        return [s for s in signals if s.strength >= self.min_strength]

    def _filter_liquidity(self, signals: list[Signal], chain) -> list[Signal]:
        """Filter out signals on illiquid contracts."""
        filtered = []
        for signal in signals:
            tick = chain.get_tick(signal.strike, signal.expiry, signal.option_type)
            if tick is None:
                continue
            if tick.volume >= self.min_volume and tick.open_interest >= self.min_oi:
                filtered.append(signal)
            else:
                logger.debug(
                    f"Filtered {signal.signal_type.value} at {signal.strike} "
                    f"— low liquidity (vol={tick.volume}, oi={tick.open_interest})"
                )
        return filtered

    def _filter_time_of_day(self, signals: list[Signal]) -> list[Signal]:
        """
        Avoid trading during noisy periods.

        Uses the signal's own timestamp (not real clock) so backtesting
        works correctly regardless of when the backtest is run.

        - Skip first 5 minutes after market open (9:30-9:35)
        - Skip last 5 minutes before close (15:55-16:00)
        """
        if not signals:
            return signals

        # Use the first signal's timestamp as the simulated clock
        sig_time = signals[0].timestamp
        if sig_time and sig_time > 0:
            now = datetime.fromtimestamp(sig_time)
        else:
            now = datetime.now()

        hour, minute = now.hour, now.minute

        # Market open noise (9:30 - 9:35 ET)
        if hour == 9 and minute < 35:
            logger.debug("Filtered all signals — market open noise period")
            return []

        # Market close noise (15:55 - 16:00 ET)
        if hour == 15 and minute >= 55:
            logger.debug("Filtered all signals — market close noise period")
            return []

        # Outside market hours — only filter if using real time
        # During backtesting, signal timestamps reflect the simulated market hours
        if hour < 9 or hour >= 16:
            # Check if this looks like backtesting data (timestamps from data bars)
            if sig_time and sig_time > 0:
                # Backtesting — allow through (data timestamps drive the clock)
                pass
            else:
                return []

        return signals

    def _filter_correlated(self, signals: list[Signal]) -> list[Signal]:
        """
        Remove redundant signals on highly correlated strikes.

        If two signals are on the same expiry and within $2 of each other,
        keep only the stronger one.
        """
        if not signals:
            return signals

        strike_distance = self.config.get("dedup_strike_distance", 2.0)
        seen: list[Signal] = []

        for signal in sorted(signals, key=lambda s: s.strength, reverse=True):
            is_dup = False
            for existing in seen:
                if (
                    signal.expiry == existing.expiry
                    and signal.signal_type == existing.signal_type
                    and abs(signal.strike - existing.strike) <= strike_distance
                ):
                    is_dup = True
                    break
            if not is_dup:
                seen.append(signal)

        return seen

    def _filter_spread(self, signals: list[Signal], chain) -> list[Signal]:
        """Filter out signals on contracts with excessively wide spreads."""
        filtered = []
        for signal in signals:
            tick = chain.get_tick(signal.strike, signal.expiry, signal.option_type)
            if tick and tick.spread_pct <= self.max_spread_pct:
                filtered.append(signal)
            elif tick:
                logger.debug(
                    f"Filtered {signal.signal_type.value} at {signal.strike} "
                    f"— spread too wide ({tick.spread_pct:.1f}%)"
                )
        return filtered
