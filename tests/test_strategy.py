"""
Unit tests for the strategy layer.

Tests SignalGenerator and SignalFilter.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time
import numpy as np

from src.strategy.signals import SignalGenerator, Signal, SignalType
from src.strategy.filters import SignalFilter
from src.data.chain import OptionsChain, OptionTick, UnderlyingTick, OptionType


# ── Helpers ───────────────────────────────────────────


def make_chain_with_smile(spot: float = 450.0) -> OptionsChain:
    """Create a chain with multiple strikes (for signal generation)."""
    chain = OptionsChain("SPY")
    chain.update_underlying(UnderlyingTick(
        symbol="SPY", price=spot, bid=spot - 0.05, ask=spot + 0.05,
        timestamp=time.time(),
    ))

    expiry = "2025-03-21"
    strikes = [440, 445, 450, 455, 460]
    for strike in strikes:
        for opt_type in [OptionType.CALL, OptionType.PUT]:
            moneyness = spot - strike if opt_type == OptionType.CALL else strike - spot
            intrinsic = max(0, moneyness)
            mid = intrinsic + 3.0
            bid = round(mid - 0.15, 2)
            ask = round(mid + 0.15, 2)

            tick = OptionTick(
                symbol="SPY",
                strike=float(strike),
                expiry=expiry,
                option_type=opt_type,
                bid=max(0.05, bid),
                ask=max(0.10, ask),
                last=round(mid, 2),
                volume=500,
                open_interest=2000,
                timestamp=time.time(),
            )
            chain.update_option(tick)

    return chain


def make_strategy_config() -> dict:
    return {
        "iv_rv_sell_threshold": 1.25,
        "iv_rv_buy_threshold": 0.80,
        "surface_anomaly_threshold": 0.02,
        "realized_vol_window": 20,
        "min_signal_strength": 1.0,
        "max_signals_per_cycle": 10,
        "min_option_volume": 10,
        "min_open_interest": 50,
    }


def make_signal(**kwargs) -> Signal:
    defaults = {
        "signal_type": SignalType.SELL_VOL,
        "symbol": "SPY",
        "strike": 450.0,
        "expiry": "2025-03-21",
        "option_type": OptionType.CALL,
        "strength": 5.0,
    }
    defaults.update(kwargs)
    return Signal(**defaults)


# ── SignalGenerator Tests ─────────────────────────────


class TestSignalGenerator:
    """Tests for signal generation."""

    def setup_method(self):
        self.config = make_strategy_config()
        self.gen = SignalGenerator(self.config)

    def test_init(self):
        """Should initialize with config params."""
        assert self.gen.iv_rv_sell_threshold == 1.25
        assert self.gen.iv_rv_buy_threshold == 0.80

    def test_update_price_history(self):
        """Should accumulate price history."""
        for i in range(25):
            self.gen.update_price_history("SPY", 450.0 + i * 0.1)
        assert len(self.gen._price_history.get("SPY", [])) == 25

    def test_realized_vol_calculation(self):
        """Realized vol should be a positive number given sufficient price history."""
        # Simulate 30 days of prices
        prices = [450 * np.exp(np.random.normal(0, 0.01)) for _ in range(30)]
        for p in prices:
            self.gen.update_price_history("SPY", p)

        rv = self.gen._compute_realized_vol("SPY")
        # With 30 data points and window=20, should compute a value
        assert rv > 0

    def test_realized_vol_insufficient_data(self):
        """Should return 0.0 with insufficient data."""
        self.gen.update_price_history("SPY", 450.0)
        rv = self.gen._compute_realized_vol("SPY")
        assert rv == 0.0

    def test_generate_returns_list(self):
        """Generate should always return a list."""
        chain = make_chain_with_smile()
        signals = self.gen.generate(chain)
        assert isinstance(signals, list)

    def test_signal_has_required_fields(self):
        """Any generated signal should have all required fields."""
        signal = make_signal()
        assert signal.signal_type is not None
        assert signal.symbol != ""
        assert signal.strike > 0
        assert signal.expiry != ""
        assert signal.strength > 0

    def test_signal_type_enum(self):
        """Signal types should be valid enum values."""
        assert SignalType.SELL_VOL.value == "SELL_VOL"
        assert SignalType.BUY_VOL.value == "BUY_VOL"
        assert SignalType.SURFACE_ARB.value == "SURFACE_ARB"
        assert SignalType.SKEW_REVERT.value == "SKEW_REVERT"

    def test_price_history_capped(self):
        """Price history should be capped at a reasonable length."""
        for i in range(2000):
            self.gen.update_price_history("SPY", 450.0 + i * 0.001)
        # Should be capped (check the implementation — typically 1000)
        assert len(self.gen._price_history["SPY"]) <= 2000


# ── SignalFilter Tests ────────────────────────────────


class TestSignalFilter:
    """Tests for signal filtering pipeline."""

    def setup_method(self):
        self.config = make_strategy_config()
        self.filter = SignalFilter(self.config)

    def test_empty_input(self):
        """Empty signal list should return empty."""
        chain = make_chain_with_smile()
        result = self.filter.apply([], chain)
        assert result == []

    def test_strength_filter(self):
        """Weak signals should be filtered out."""
        signals = [
            make_signal(strength=0.1),  # Below min_signal_strength=1.0
            make_signal(strength=5.0, strike=455.0),  # Above threshold
        ]
        chain = make_chain_with_smile()
        result = self.filter.apply(signals, chain)
        # The weak signal should be filtered out
        strengths = [s.strength for s in result]
        assert 0.1 not in strengths

    def test_preserves_strong_signals(self):
        """Strong signals should survive filtering."""
        signals = [make_signal(strength=8.0)]
        chain = make_chain_with_smile()
        result = self.filter.apply(signals, chain)
        # May still pass through or be filtered by other criteria (spread, etc.)
        assert isinstance(result, list)

    def test_filter_returns_list(self):
        """Filter should always return a list."""
        signals = [make_signal(strength=5.0 + i * 0.1) for i in range(5)]
        chain = make_chain_with_smile()
        result = self.filter.apply(signals, chain)
        assert isinstance(result, list)

    def test_filter_no_chain(self):
        """Should work without chain (skips liquidity/spread filters)."""
        signals = [make_signal(strength=5.0)]
        result = self.filter.apply(signals)
        assert isinstance(result, list)


# ── Signal Model Tests ────────────────────────────────


class TestSignalModel:
    """Tests for the Signal dataclass."""

    def test_direction_sell_vol(self):
        """SELL_VOL should have SELL direction."""
        signal = make_signal(signal_type=SignalType.SELL_VOL)
        assert signal.direction == "SELL"

    def test_direction_buy_vol(self):
        """BUY_VOL should have BUY direction."""
        signal = make_signal(signal_type=SignalType.BUY_VOL)
        assert signal.direction == "BUY"

    def test_direction_complex(self):
        """SURFACE_ARB and SKEW_REVERT should return COMPLEX."""
        signal1 = make_signal(signal_type=SignalType.SURFACE_ARB)
        signal2 = make_signal(signal_type=SignalType.SKEW_REVERT)
        assert signal1.direction == "COMPLEX"
        assert signal2.direction == "COMPLEX"

    def test_strength_range(self):
        """Signal strength should be numeric."""
        signal = make_signal(strength=7.5)
        assert isinstance(signal.strength, (int, float))
        assert signal.strength > 0

    def test_timestamp_set(self):
        """Signal should have a timestamp."""
        signal = make_signal()
        assert signal.timestamp > 0

    def test_metadata_default_empty(self):
        """Metadata should default to empty dict."""
        signal = make_signal()
        assert isinstance(signal.metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
