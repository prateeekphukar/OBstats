"""
Unit tests for the risk manager.

Tests all pre-trade risk checks, portfolio tracking, and kill switch.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.risk.portfolio import Portfolio, Position
from src.risk.manager import RiskManager
from src.risk.kill_switch import KillSwitch
from src.strategy.signals import Signal, SignalType
from src.data.chain import OptionType


# ── Fixtures ──────────────────────────────────────────


def make_signal(**kwargs) -> Signal:
    """Helper to create test signals."""
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


def make_position(**kwargs) -> Position:
    """Helper to create test positions."""
    defaults = {
        "contract_id": "SPY_2025-03-21_450.0_call",
        "symbol": "SPY",
        "strike": 450.0,
        "expiry": "2025-03-21",
        "option_type": OptionType.CALL,
        "side": "BUY",
        "quantity": 5,
        "entry_price": 3.50,
        "current_price": 3.50,
    }
    defaults.update(kwargs)
    return Position(**defaults)


def make_risk_config() -> dict:
    """Create a test risk config."""
    return {
        "position_limits": {
            "max_total_contracts": 50,
            "max_contracts_per_symbol": 20,
            "max_contracts_per_strike": 10,
            "max_single_expiry_pct": 0.40,
        },
        "pnl_limits": {
            "max_daily_loss": 5000.0,
            "max_unrealized_loss": 3000.0,
        },
        "greeks_limits": {
            "max_portfolio_delta": 100,
            "max_portfolio_gamma": 50,
            "max_portfolio_vega": 500,
        },
        "rate_limits": {
            "max_trades_per_minute": 30,
            "max_orders_per_second": 5,
        },
    }


# ── Portfolio Tests ───────────────────────────────────


class TestPortfolio:
    """Tests for portfolio tracking."""

    def setup_method(self):
        self.portfolio = Portfolio()

    def test_add_position(self):
        """Should add positions correctly."""
        pos = make_position()
        self.portfolio.add_position(pos)
        assert self.portfolio.total_contracts == 5
        assert len(self.portfolio.positions) == 1

    def test_unrealized_pnl(self):
        """Should compute unrealized P&L correctly."""
        pos = make_position(quantity=1, entry_price=3.00, current_price=4.00)
        self.portfolio.add_position(pos)
        # BUY 1 contract: (4.00 - 3.00) * 1 * 100 = $100
        self.portfolio.update_mark(pos.contract_id, 4.00)
        assert self.portfolio.total_unrealized_pnl == 100.0

    def test_close_position_pnl(self):
        """Should compute realized P&L on close."""
        pos = make_position(quantity=1, entry_price=3.00)
        self.portfolio.add_position(pos)
        pnl = self.portfolio.close_position(pos.contract_id, 5.00)
        # BUY 1: (5.00 - 3.00) * 1 * 100 = $200
        assert pnl == 200.0
        assert pos.contract_id not in self.portfolio.positions

    def test_net_delta(self):
        """Should compute net portfolio delta."""
        pos = make_position(quantity=1, side="BUY")
        self.portfolio.add_position(pos)
        self.portfolio.update_mark(pos.contract_id, 3.50, delta=0.5)
        assert self.portfolio.net_delta == 0.5  # 0.5 * 1

    def test_expiry_concentration(self):
        """Should compute expiry concentration."""
        self.portfolio.add_position(
            make_position(contract_id="c1", expiry="2025-03-21", quantity=3)
        )
        self.portfolio.add_position(
            make_position(contract_id="c2", expiry="2025-04-18", quantity=7)
        )
        conc = self.portfolio.expiry_concentration()
        assert abs(conc["2025-03-21"] - 0.30) < 0.01
        assert abs(conc["2025-04-18"] - 0.70) < 0.01


# ── Risk Manager Tests ────────────────────────────────


class TestRiskManager:
    """Tests for pre-trade risk checks."""

    def setup_method(self):
        self.portfolio = Portfolio()
        self.config = make_risk_config()
        self.risk = RiskManager(self.config, self.portfolio)

    def test_approve_empty_portfolio(self):
        """Should approve signals when portfolio is empty."""
        signal = make_signal()
        approved, reason = self.risk.approve(signal)
        assert approved
        assert reason == "approved"

    def test_reject_position_limit(self):
        """Should reject when position limit would be exceeded."""
        # Fill up to limit
        for i in range(50):
            self.portfolio.add_position(
                make_position(contract_id=f"c{i}", quantity=1)
            )

        signal = make_signal()
        approved, reason = self.risk.approve(signal)
        assert not approved
        assert "contracts" in reason.lower() or "limit" in reason.lower()

    def test_reject_daily_loss_limit(self):
        """Should reject when daily loss limit is exceeded."""
        self.portfolio.daily_realized_pnl = -6000.0
        signal = make_signal()
        approved, reason = self.risk.approve(signal)
        assert not approved
        assert "daily loss" in reason.lower()

    def test_reject_greeks_limit(self):
        """Should reject when Greek limits are exceeded."""
        pos = make_position(quantity=10)
        self.portfolio.add_position(pos)
        self.portfolio.update_mark(pos.contract_id, 3.50, delta=15.0)
        # net_delta = 15.0 * 10 = 150 > max 100

        # Use a different expiry so concentration check doesn't interfere
        signal = make_signal(expiry="2025-04-18")
        approved, reason = self.risk.approve(signal)
        assert not approved
        assert "delta" in reason.lower()

    def test_approve_within_limits(self):
        """Should approve when all checks pass."""
        # Add positions across two expiries so no single expiry exceeds 40% limit
        pos1 = make_position(contract_id="c1", quantity=3, expiry="2025-03-21")
        pos2 = make_position(contract_id="c2", quantity=5, expiry="2025-04-18")
        self.portfolio.add_position(pos1)
        self.portfolio.add_position(pos2)
        self.portfolio.update_mark("c1", 3.50, delta=0.3)
        self.portfolio.update_mark("c2", 3.50, delta=0.3)

        # Signal on a new expiry so it doesn't push concentration above 40%
        signal = make_signal(expiry="2025-05-16")
        approved, reason = self.risk.approve(signal)
        assert approved


# ── Kill Switch Tests ─────────────────────────────────


class TestKillSwitch:
    """Tests for the kill switch."""

    def setup_method(self):
        self.portfolio = Portfolio()
        self.config = {
            "kill_switch": {
                "enabled": True,
                "max_daily_loss": 5000.0,
                "max_daily_trades": 500,
                "max_consecutive_losses": 10,
                "cool_down_minutes": 15,
            }
        }
        self.kill_switch = KillSwitch(self.config, self.portfolio)

    def test_not_triggered_initially(self):
        """Should not be triggered initially."""
        assert not self.kill_switch.is_triggered

    def test_consecutive_losses_tracking(self):
        """Should track consecutive losses."""
        for _ in range(5):
            self.kill_switch.record_trade_result(-100.0)
        assert self.kill_switch._consecutive_losses == 5

    def test_consecutive_losses_reset_on_win(self):
        """Should reset consecutive losses on a winning trade."""
        for _ in range(5):
            self.kill_switch.record_trade_result(-100.0)
        self.kill_switch.record_trade_result(200.0)
        assert self.kill_switch._consecutive_losses == 0

    def test_daily_reset(self):
        """Should reset daily counters."""
        self.kill_switch._trade_count_today = 100
        self.kill_switch._consecutive_losses = 5
        self.kill_switch.reset_daily()
        assert self.kill_switch._trade_count_today == 0
        assert self.kill_switch._consecutive_losses == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
