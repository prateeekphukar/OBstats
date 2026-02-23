"""
Unit tests for the data layer.

Tests OptionsChain, OptionTick models, and data operations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import time

from src.data.chain import OptionsChain, OptionTick, UnderlyingTick, OptionType


# ── Helpers ───────────────────────────────────────────


def make_tick(**kwargs) -> OptionTick:
    defaults = {
        "symbol": "SPY",
        "strike": 450.0,
        "expiry": "2025-03-21",
        "option_type": OptionType.CALL,
        "bid": 4.80,
        "ask": 5.20,
        "last": 5.00,
        "volume": 500,
        "open_interest": 1000,
        "timestamp": time.time(),
    }
    defaults.update(kwargs)
    return OptionTick(**defaults)


def make_underlying(**kwargs) -> UnderlyingTick:
    defaults = {
        "symbol": "SPY",
        "price": 450.0,
        "bid": 449.95,
        "ask": 450.05,
        "timestamp": time.time(),
    }
    defaults.update(kwargs)
    return UnderlyingTick(**defaults)


# ── OptionTick Model Tests ────────────────────────────


class TestOptionTick:
    """Tests for the OptionTick dataclass."""

    def test_mid_price(self):
        """Mid should be average of bid and ask."""
        tick = make_tick(bid=4.80, ask=5.20)
        assert tick.mid == pytest.approx(5.00, abs=0.01)

    def test_mid_price_zero_bid(self):
        """Should handle zero bid gracefully."""
        tick = make_tick(bid=0, ask=1.00)
        assert tick.mid == pytest.approx(0.50, abs=0.01)

    def test_spread(self):
        """Spread should be ask - bid."""
        tick = make_tick(bid=4.80, ask=5.20)
        assert tick.spread == pytest.approx(0.40, abs=0.01)

    def test_spread_pct(self):
        """Spread percent should be spread / mid * 100."""
        tick = make_tick(bid=4.80, ask=5.20)
        expected_pct = (0.40 / 5.00) * 100
        assert tick.spread_pct == pytest.approx(expected_pct, abs=0.1)

    def test_contract_id(self):
        """Contract ID should combine symbol, expiry, strike, type."""
        tick = make_tick(symbol="AAPL", strike=150.0, expiry="2025-04-18", option_type=OptionType.PUT)
        assert "AAPL" in tick.contract_id
        assert "150" in tick.contract_id
        assert "2025-04-18" in tick.contract_id

    def test_contract_id_format(self):
        """Contract ID should be symbol_expiry_strike_type."""
        tick = make_tick(symbol="SPY", strike=450.0, expiry="2025-03-21", option_type=OptionType.CALL)
        assert tick.contract_id == "SPY_2025-03-21_450.0_call"


# ── UnderlyingTick Tests ──────────────────────────────


class TestUnderlyingTick:
    """Tests for the UnderlyingTick dataclass."""

    def test_fields(self):
        """Should store all fields correctly."""
        tick = make_underlying(symbol="AAPL", price=175.50)
        assert tick.symbol == "AAPL"
        assert tick.price == 175.50

    def test_bid_ask(self):
        """Bid and ask should be stored."""
        tick = make_underlying(bid=449.95, ask=450.05)
        assert tick.bid == 449.95
        assert tick.ask == 450.05


# ── OptionsChain Tests ────────────────────────────────


class TestOptionsChain:
    """Tests for the OptionsChain manager."""

    def setup_method(self):
        self.chain = OptionsChain("SPY")

    def test_init(self):
        """Should initialize with symbol."""
        assert self.chain.symbol == "SPY"

    def test_update_option(self):
        """Should store option ticks."""
        tick = make_tick()
        self.chain.update_option(tick)
        result = self.chain.get_tick(450.0, "2025-03-21", OptionType.CALL)
        assert result is not None
        assert result.bid == 4.80
        assert result.ask == 5.20

    def test_update_underlying(self):
        """Should store underlying price."""
        tick = make_underlying(price=451.50)
        self.chain.update_underlying(tick)
        assert self.chain.underlying is not None
        assert self.chain.underlying.price == 451.50

    def test_get_tick_missing(self):
        """Should return None for missing ticks."""
        result = self.chain.get_tick(999.0, "2025-03-21", OptionType.CALL)
        assert result is None

    def test_get_tick_by_type(self):
        """Should distinguish calls from puts at same strike."""
        call = make_tick(option_type=OptionType.CALL, bid=5.00, ask=5.50)
        put = make_tick(option_type=OptionType.PUT, bid=3.00, ask=3.50)
        self.chain.update_option(call)
        self.chain.update_option(put)

        call_result = self.chain.get_tick(450.0, "2025-03-21", OptionType.CALL)
        put_result = self.chain.get_tick(450.0, "2025-03-21", OptionType.PUT)
        assert call_result.bid == 5.00
        assert put_result.bid == 3.00

    def test_multiple_strikes(self):
        """Should handle multiple strikes."""
        for strike in [440, 445, 450, 455, 460]:
            tick = make_tick(strike=float(strike))
            self.chain.update_option(tick)

        assert self.chain.get_tick(440.0, "2025-03-21", OptionType.CALL) is not None
        assert self.chain.get_tick(460.0, "2025-03-21", OptionType.CALL) is not None

    def test_multiple_expiries(self):
        """Should handle multiple expiry dates."""
        t1 = make_tick(expiry="2025-03-21")
        t2 = make_tick(expiry="2025-04-18")
        self.chain.update_option(t1)
        self.chain.update_option(t2)

        assert self.chain.get_tick(450.0, "2025-03-21", OptionType.CALL) is not None
        assert self.chain.get_tick(450.0, "2025-04-18", OptionType.CALL) is not None

    def test_update_replaces_old_tick(self):
        """Updating same contract should replace the old tick."""
        tick1 = make_tick(bid=4.80, ask=5.20)
        self.chain.update_option(tick1)

        tick2 = make_tick(bid=5.00, ask=5.40)
        self.chain.update_option(tick2)

        result = self.chain.get_tick(450.0, "2025-03-21", OptionType.CALL)
        assert result.bid == 5.00  # Updated value

    def test_filter_liquid(self):
        """Should filter for liquid ticks."""
        liquid = make_tick(volume=1000, open_interest=5000)
        illiquid = make_tick(strike=440.0, volume=5, open_interest=10)
        self.chain.update_option(liquid)
        self.chain.update_option(illiquid)

        liquid_ticks = self.chain.filter_liquid(min_volume=100, min_oi=500)
        assert len(liquid_ticks) >= 1
        # The illiquid tick should not be in the result
        for t in liquid_ticks:
            assert t.volume >= 100

    def test_size(self):
        """Size should reflect total ticks."""
        assert self.chain.size == 0
        self.chain.update_option(make_tick())
        assert self.chain.size == 1
        self.chain.update_option(make_tick(strike=455.0))
        assert self.chain.size == 2

    def test_get_calls(self):
        """Should return only call options."""
        self.chain.update_option(make_tick(option_type=OptionType.CALL, strike=450.0))
        self.chain.update_option(make_tick(option_type=OptionType.PUT, strike=450.0))
        calls = self.chain.get_calls()
        assert len(calls) == 1
        assert calls[0].option_type == OptionType.CALL

    def test_get_puts(self):
        """Should return only put options."""
        self.chain.update_option(make_tick(option_type=OptionType.CALL, strike=450.0))
        self.chain.update_option(make_tick(option_type=OptionType.PUT, strike=450.0))
        puts = self.chain.get_puts()
        assert len(puts) == 1
        assert puts[0].option_type == OptionType.PUT

    def test_get_expirations(self):
        """Should return sorted list of expirations."""
        self.chain.update_option(make_tick(expiry="2025-04-18"))
        self.chain.update_option(make_tick(expiry="2025-03-21", strike=445.0))
        exps = self.chain.get_expirations()
        assert exps == ["2025-03-21", "2025-04-18"]

    def test_get_strikes(self):
        """Should return sorted strikes."""
        for s in [455, 445, 450]:
            self.chain.update_option(make_tick(strike=float(s)))
        strikes = self.chain.get_strikes()
        assert strikes == [445.0, 450.0, 455.0]

    def test_all_ticks(self):
        """Should return all ticks."""
        self.chain.update_option(make_tick(strike=450.0))
        self.chain.update_option(make_tick(strike=455.0))
        assert len(self.chain.all_ticks) == 2

    def test_update_options_batch(self):
        """Should update multiple ticks at once."""
        ticks = [make_tick(strike=float(s)) for s in [440, 445, 450]]
        self.chain.update_options_batch(ticks)
        assert self.chain.size == 3

    def test_get_atm_strike(self):
        """Should find ATM strike closest to underlying price."""
        self.chain.update_underlying(make_underlying(price=448.0))
        for s in [440, 445, 450, 455]:
            self.chain.update_option(make_tick(strike=float(s)))
        atm = self.chain.get_atm_strike()
        assert atm == 450.0  # Closest to 448

    def test_last_update(self):
        """Should track timestamp of last update."""
        assert self.chain.last_update == 0
        self.chain.update_option(make_tick())
        assert self.chain.last_update > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
