"""
Unit tests for the backtesting engine.

Tests PerformanceMetrics calculations and HistoricalDataLoader.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from src.backtest.metrics import PerformanceMetrics
from src.backtest.data_loader import HistoricalDataLoader


# ── PerformanceMetrics Tests ──────────────────────────


class TestPerformanceMetrics:
    """Tests for the performance metrics calculator."""

    def test_basic_metrics_flat(self):
        """Should compute basic metrics from flat equity curve."""
        equity = [100_000] * 100
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["total_pnl"] == 0
        assert metrics["total_return_pct"] == 0
        assert metrics["max_drawdown_pct"] == 0

    def test_profitable_equity_curve(self):
        """Should show positive return for upward curve."""
        equity = [100_000 + i * 100 for i in range(100)]
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["total_pnl"] > 0
        assert metrics["total_return_pct"] > 0
        assert metrics["final_equity"] == equity[-1]

    def test_losing_equity_curve(self):
        """Should show negative return for downward curve."""
        equity = [100_000 - i * 100 for i in range(50)]
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["total_pnl"] < 0
        assert metrics["total_return_pct"] < 0

    def test_max_drawdown(self):
        """Should compute max drawdown correctly."""
        # Peak at 110k, valley at 90k → 18.18% drawdown
        equity = [100_000, 105_000, 110_000, 100_000, 90_000, 95_000, 100_000]
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert abs(metrics["max_drawdown_pct"] - 18.18) < 0.5

    def test_sharpe_ratio_positive(self):
        """Steady positive returns should have positive Sharpe."""
        equity = [100_000 + i * 50 for i in range(252)]
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["sharpe_ratio"] > 0

    def test_sharpe_ratio_volatile(self):
        """Highly volatile returns should produce a numeric Sharpe."""
        np.random.seed(42)
        equity = [100_000]
        for i in range(251):
            equity.append(equity[-1] * (1 + np.random.normal(0, 0.03)))
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert isinstance(metrics["sharpe_ratio"], float)

    def test_sortino_ratio(self):
        """Sortino should be computed."""
        equity = [100_000 + i * 10 for i in range(100)]
        trades = []
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert "sortino_ratio" in metrics

    def test_win_rate(self):
        """Should compute win rate from trade P&Ls."""
        trades = [
            {"pnl": 100}, {"pnl": 200}, {"pnl": -50},
            {"pnl": 150}, {"pnl": -30},
        ]
        equity = [100_000] * 10
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["win_rate_pct"] == 60.0

    def test_profit_factor(self):
        """Should compute profit factor correctly."""
        trades = [
            {"pnl": 100}, {"pnl": 200}, {"pnl": -50},
        ]
        equity = [100_000] * 10
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["profit_factor"] == 6.0

    def test_profit_factor_no_losses(self):
        """Profit factor with no losses should be infinite."""
        trades = [{"pnl": 100}, {"pnl": 200}]
        equity = [100_000] * 10
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["profit_factor"] == "inf"

    def test_insufficient_data(self):
        """Should return error dict with insufficient data."""
        metrics = PerformanceMetrics.compute([], [])
        assert "error" in metrics

    def test_single_point(self):
        """Should return error with single data point."""
        metrics = PerformanceMetrics.compute([100_000], [])
        assert "error" in metrics

    def test_total_trades_count(self):
        """Should count total trades."""
        trades = [{"side": "BUY"}, {"side": "BUY"}, {"side": "SELL"}]
        equity = [100_000] * 10
        metrics = PerformanceMetrics.compute(equity, trades, initial_capital=100_000)
        assert metrics["total_trades"] == 3

    def test_format_report(self):
        """Should produce a formatted text report."""
        metrics = {
            "total_pnl": 5000,
            "total_return_pct": 5.0,
            "final_equity": 105_000,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown_pct": 3.5,
            "annual_volatility_pct": 15.0,
            "total_trades": 150,
            "win_rate_pct": 55.0,
            "avg_win": 100.0,
            "avg_loss": 50.0,
            "profit_factor": 2.2,
        }
        report = PerformanceMetrics.format_report(metrics)
        assert "BACKTEST PERFORMANCE REPORT" in report
        assert "5,000" in report
        assert "5.00%" in report

    def test_annual_volatility(self):
        """Should compute annual volatility."""
        np.random.seed(123)
        equity = [100_000]
        for _ in range(251):
            equity.append(equity[-1] * (1 + np.random.normal(0.0003, 0.01)))
        metrics = PerformanceMetrics.compute(equity, [], initial_capital=100_000)
        assert metrics["annual_volatility_pct"] > 0


# ── HistoricalDataLoader Tests ────────────────────────


class TestHistoricalDataLoader:
    """Tests for the historical data loader."""

    def setup_method(self):
        self.loader = HistoricalDataLoader()

    def test_generate_sample_data(self):
        """Should generate a valid DataFrame."""
        df = self.loader.generate_sample_data(
            symbol="NIFTY", spot=22000, num_bars=100,
            num_strikes=5, num_expiries=1,
        )
        assert len(df) > 0
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "strike" in df.columns
        assert "expiry" in df.columns
        assert "option_type" in df.columns
        assert "bid" in df.columns
        assert "ask" in df.columns

    def test_sample_data_has_correct_symbol(self):
        """Generated data should use the specified symbol."""
        df = self.loader.generate_sample_data(symbol="BANKNIFTY", num_bars=10, num_strikes=3)
        assert (df["symbol"] == "BANKNIFTY").all()

    def test_sample_data_bids_below_asks(self):
        """Bid should always be below ask in generated data."""
        df = self.loader.generate_sample_data(num_bars=50, num_strikes=3)
        assert (df["bid"] <= df["ask"]).all()

    def test_sample_data_positive_prices(self):
        """All prices should be positive."""
        df = self.loader.generate_sample_data(num_bars=50, num_strikes=3)
        assert (df["bid"] > 0).all()
        assert (df["ask"] > 0).all()

    def test_sample_data_has_option_types(self):
        """Should generate both calls and puts."""
        df = self.loader.generate_sample_data(num_bars=50, num_strikes=5)
        assert "call" in df["option_type"].values
        assert "put" in df["option_type"].values

    def test_sample_data_sorted(self):
        """Generated data should be sorted by timestamp."""
        df = self.loader.generate_sample_data(num_bars=50, num_strikes=3)
        timestamps = df["timestamp"].values
        assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))

    def test_sample_data_has_underlying(self):
        """Should include underlying price."""
        df = self.loader.generate_sample_data(num_bars=50, num_strikes=3)
        assert "underlying_price" in df.columns
        assert (df["underlying_price"] > 0).all()

    def test_sample_data_num_rows(self):
        """Should generate expected number of rows."""
        df = self.loader.generate_sample_data(
            num_bars=10, num_strikes=5, num_expiries=2,
        )
        # 10 bars * 5 strikes * 2 expiries * 2 option types = 200
        assert len(df) == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
