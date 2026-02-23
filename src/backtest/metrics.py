"""
Performance Metrics Calculator.

Computes key strategy metrics from backtesting results:
Sharpe, Sortino, max drawdown, win rate, profit factor, etc.
"""

import numpy as np
from typing import Optional


class PerformanceMetrics:
    """
    Computes and reports strategy performance metrics.
    """

    @staticmethod
    def compute(
        equity_curve: list[float],
        trades: list[dict],
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> dict:
        """
        Compute all performance metrics.

        Args:
            equity_curve: List of portfolio values over time.
            trades: List of trade dicts from backtesting.
            initial_capital: Starting capital.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            periods_per_year: Trading periods per year (252 for daily).

        Returns:
            Dict of computed metrics.
        """
        if not equity_curve or len(equity_curve) < 2:
            return {"error": "Insufficient data for metrics"}

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic P&L
        total_pnl = equity[-1] - initial_capital
        total_return_pct = (equity[-1] / initial_capital - 1) * 100

        # Risk metrics
        sharpe = PerformanceMetrics._sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = PerformanceMetrics._sortino_ratio(returns, risk_free_rate, periods_per_year)
        max_dd, max_dd_duration = PerformanceMetrics._max_drawdown(equity)
        volatility = float(np.std(returns) * np.sqrt(periods_per_year)) if len(returns) > 0 else 0

        # Trade-level metrics
        trade_pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]

        win_rate = len(winning_trades) / len(trade_pnls) * 100 if trade_pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float("inf")

        # Duration metrics
        total_trades = len(trades)

        return {
            "total_pnl": round(float(total_pnl), 2),
            "total_return_pct": round(float(total_return_pct), 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "max_drawdown_pct": round(float(max_dd * 100), 2),
            "max_drawdown_duration": int(max_dd_duration),
            "annual_volatility_pct": round(float(volatility * 100), 2),
            "total_trades": total_trades,
            "win_rate_pct": round(float(win_rate), 2),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
            "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else "inf",
            "final_equity": round(float(equity[-1]), 2),
        }

    @staticmethod
    def _sharpe_ratio(
        returns: np.ndarray, risk_free_rate: float, periods_per_year: int
    ) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year))

    @staticmethod
    def _sortino_ratio(
        returns: np.ndarray, risk_free_rate: float, periods_per_year: int
    ) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(returns) == 0:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        excess_returns = returns - rf_per_period
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0
        if downside_std == 0:
            return 0.0
        return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
        """
        Compute maximum drawdown and its duration.

        Returns:
            (max_drawdown_fraction, duration_in_periods)
        """
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown))

        # Compute duration of max drawdown
        max_dd_idx = int(np.argmax(drawdown))
        peak_idx = int(np.argmax(equity[:max_dd_idx + 1]))
        duration = max_dd_idx - peak_idx

        return max_dd, duration

    @staticmethod
    def format_report(metrics: dict) -> str:
        """Format metrics as a human-readable report."""
        lines = [
            "=" * 50,
            "  BACKTEST PERFORMANCE REPORT",
            "=" * 50,
            f"  Total P&L:           ${metrics.get('total_pnl', 0):>12,.2f}",
            f"  Total Return:        {metrics.get('total_return_pct', 0):>11.2f}%",
            f"  Final Equity:        ${metrics.get('final_equity', 0):>12,.2f}",
            "-" * 50,
            f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):>12.4f}",
            f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):>12.4f}",
            f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):>11.2f}%",
            f"  Annual Volatility:   {metrics.get('annual_volatility_pct', 0):>11.2f}%",
            "-" * 50,
            f"  Total Trades:        {metrics.get('total_trades', 0):>12}",
            f"  Win Rate:            {metrics.get('win_rate_pct', 0):>11.2f}%",
            f"  Avg Win:             ${metrics.get('avg_win', 0):>12,.2f}",
            f"  Avg Loss:            ${metrics.get('avg_loss', 0):>12,.2f}",
            f"  Profit Factor:       {metrics.get('profit_factor', 0):>12}",
            "=" * 50,
        ]
        return "\n".join(lines)
