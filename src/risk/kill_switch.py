"""
Kill Switch.

Emergency shutdown system that monitors portfolio health and
triggers immediate position flatten + order cancellation if
thresholds are breached.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Any

from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class KillSwitch:
    """
    Emergency kill switch for the trading bot.

    Continuously monitors P&L and trade activity.
    When triggered, it:
    1. Cancels ALL open orders
    2. Flattens ALL positions
    3. Pauses trading for a cool-down period
    4. Sends alerts via configured channels
    """

    def __init__(
        self,
        config: dict,
        portfolio: Portfolio,
        cancel_all_fn: Optional[Callable] = None,
        flatten_all_fn: Optional[Callable] = None,
        alert_fn: Optional[Callable[[str], Any]] = None,
    ):
        """
        Args:
            config: Kill switch config from risk_limits.yaml.
            portfolio: Portfolio tracker.
            cancel_all_fn: Async function to cancel all open orders.
            flatten_all_fn: Async function to flatten all positions.
            alert_fn: Function to send alerts (Telegram, Discord, email, etc.).
        """
        ks_config = config.get("kill_switch", {})
        self.enabled = ks_config.get("enabled", True)
        self.max_daily_loss = ks_config.get("max_daily_loss", 5000.0)
        self.max_daily_trades = ks_config.get("max_daily_trades", 500)
        self.max_consecutive_losses = ks_config.get("max_consecutive_losses", 10)
        self.cool_down_minutes = ks_config.get("cool_down_minutes", 15)

        self.portfolio = portfolio
        self.cancel_all_fn = cancel_all_fn
        self.flatten_all_fn = flatten_all_fn
        self.alert_fn = alert_fn

        self._triggered = False
        self._trigger_time: Optional[float] = None
        self._consecutive_losses = 0
        self._trade_count_today = 0
        self._monitor_interval = 0.5  # Check every 500ms

    @property
    def is_triggered(self) -> bool:
        """Whether the kill switch is currently active."""
        return self._triggered

    @property
    def is_in_cooldown(self) -> bool:
        """Whether we're in the cool-down period after a trigger."""
        if not self._trigger_time:
            return False
        elapsed = time.time() - self._trigger_time
        return elapsed < self.cool_down_minutes * 60

    @property
    def cooldown_remaining_seconds(self) -> float:
        """Seconds remaining in cool-down period."""
        if not self._trigger_time:
            return 0
        elapsed = time.time() - self._trigger_time
        remaining = self.cool_down_minutes * 60 - elapsed
        return max(0, remaining)

    def record_trade_result(self, pnl: float) -> None:
        """Record the result of a completed trade for monitoring."""
        self._trade_count_today += 1
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._trade_count_today = 0
        self._consecutive_losses = 0
        self._triggered = False
        self._trigger_time = None
        logger.info("Kill switch daily counters reset")

    async def monitor(self) -> None:
        """
        Continuous monitoring loop.

        Run this as an asyncio task alongside the main trading loop.
        """
        if not self.enabled:
            logger.info("Kill switch is disabled")
            return

        logger.info("Kill switch monitoring started")

        while True:
            try:
                if self.is_in_cooldown:
                    remaining = self.cooldown_remaining_seconds
                    if remaining <= 0:
                        logger.info("Cool-down period ended — resuming monitoring")
                        self._triggered = False
                    await asyncio.sleep(self._monitor_interval)
                    continue

                if not self._triggered and self._should_trigger():
                    await self._execute_shutdown()

                await asyncio.sleep(self._monitor_interval)

            except asyncio.CancelledError:
                logger.info("Kill switch monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Kill switch monitor error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _should_trigger(self) -> bool:
        """Evaluate all trigger conditions."""
        reasons = []

        # Check 1: Daily loss exceeded
        daily_pnl = self.portfolio.daily_pnl
        if daily_pnl < -self.max_daily_loss:
            reasons.append(f"Daily loss: ${daily_pnl:.2f} < -${self.max_daily_loss}")

        # Check 2: Too many trades (possible runaway)
        if self._trade_count_today > self.max_daily_trades:
            reasons.append(f"Trade count: {self._trade_count_today} > {self.max_daily_trades}")

        # Check 3: Consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            reasons.append(f"Consecutive losses: {self._consecutive_losses}")

        if reasons:
            logger.critical(f"🚨 KILL SWITCH TRIGGERED: {'; '.join(reasons)}")
            return True

        return False

    async def _execute_shutdown(self) -> None:
        """Execute emergency shutdown sequence."""
        self._triggered = True
        self._trigger_time = time.time()

        msg = (
            f"🚨 KILL SWITCH ACTIVATED\n"
            f"Daily P&L: ${self.portfolio.daily_pnl:.2f}\n"
            f"Trades today: {self._trade_count_today}\n"
            f"Consecutive losses: {self._consecutive_losses}\n"
            f"Cool-down: {self.cool_down_minutes} minutes"
        )

        logger.critical(msg)

        # Step 1: Cancel all open orders
        if self.cancel_all_fn:
            try:
                await self._maybe_await(self.cancel_all_fn())
                logger.info("All open orders cancelled")
            except Exception as e:
                logger.error(f"Failed to cancel orders: {e}")

        # Step 2: Flatten all positions
        if self.flatten_all_fn:
            try:
                await self._maybe_await(self.flatten_all_fn())
                logger.info("All positions flattened")
            except Exception as e:
                logger.error(f"Failed to flatten positions: {e}")

        # Step 3: Send alert
        if self.alert_fn:
            try:
                await self._maybe_await(self.alert_fn(msg))
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    @staticmethod
    async def _maybe_await(result):
        """Await coroutine if necessary."""
        if asyncio.iscoroutine(result):
            await result

    def status(self) -> dict:
        """Kill switch status."""
        return {
            "enabled": self.enabled,
            "triggered": self._triggered,
            "in_cooldown": self.is_in_cooldown,
            "cooldown_remaining_sec": round(self.cooldown_remaining_seconds, 0),
            "consecutive_losses": self._consecutive_losses,
            "trades_today": self._trade_count_today,
        }
