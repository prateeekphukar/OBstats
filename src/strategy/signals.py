"""
Signal Generator.

Produces trading signals based on volatility analysis:
1. IV vs Realized Vol divergence
2. Vol surface anomalies
3. Skew mean-reversion
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

from ..data.chain import OptionsChain, OptionTick, OptionType
from ..pricing.vol_surface import VolSurface, SurfacePoint
from ..pricing.iv_solver import ImpliedVolSolver
from ..pricing.black_scholes import BlackScholesPricer

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    SELL_VOL = "SELL_VOL"         # IV is high relative to RV — sell volatility
    BUY_VOL = "BUY_VOL"           # IV is low relative to RV — buy volatility
    SURFACE_ARB = "SURFACE_ARB"   # Vol surface anomaly — butterfly/spread
    SKEW_REVERT = "SKEW_REVERT"   # Skew is extreme — mean-reversion trade


@dataclass
class Signal:
    """A trading signal produced by the strategy."""
    signal_type: SignalType
    symbol: str
    strike: float
    expiry: str
    option_type: OptionType
    strength: float            # 0-10 scale, higher = stronger conviction
    iv: Optional[float] = None
    realized_vol: Optional[float] = None
    expected_iv: Optional[float] = None
    deviation: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def direction(self) -> str:
        """Whether the signal implies buying or selling."""
        if self.signal_type in (SignalType.SELL_VOL,):
            return "SELL"
        elif self.signal_type in (SignalType.BUY_VOL,):
            return "BUY"
        else:
            return "COMPLEX"  # Multi-leg strategies


class SignalGenerator:
    """
    Core signal generation engine.

    Analyzes the options chain, vol surface, and historical data
    to produce actionable trading signals.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Strategy configuration dict from settings.yaml.
        """
        self.config = config
        self.pricer = BlackScholesPricer(
            risk_free_rate=config.get("risk_free_rate", 0.05),
            dividend_yield=config.get("dividend_yield", 0.013),
        )
        self.iv_solver = ImpliedVolSolver(self.pricer)
        self.vol_surface = VolSurface(
            anomaly_threshold=config.get("surface_anomaly_threshold", 0.02)
        )

        # Thresholds
        self.iv_rv_sell_threshold = config.get("iv_rv_sell_threshold", 1.25)
        self.iv_rv_buy_threshold = config.get("iv_rv_buy_threshold", 0.80)
        self.min_signal_strength = config.get("min_signal_strength", 1.2)
        self.max_signals = config.get("max_signals_per_cycle", 5)

        # Historical prices for realized vol calculation
        self._price_history: dict[str, list[float]] = {}  # symbol -> prices
        self._realized_vol_cache: dict[str, float] = {}

    def update_price_history(self, symbol: str, price: float) -> None:
        """Add a price observation for realized vol computation."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)

        # Keep a rolling window
        window = self.config.get("realized_vol_window", 20)
        max_history = window * 10  # Keep extra for safety
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]

        # Recompute realized vol
        self._realized_vol_cache[symbol] = self._compute_realized_vol(symbol)

    def generate(self, chain: OptionsChain) -> list[Signal]:
        """
        Generate signals from the current options chain state.

        Args:
            chain: Current options chain with live data.

        Returns:
            List of Signal objects, sorted by strength (descending).
        """
        if not chain.underlying:
            return []

        signals: list[Signal] = []
        symbol = chain.symbol
        spot = chain.underlying.price

        # Update vol surface with current chain data
        self._update_vol_surface(chain, spot)

        # Strategy 1: IV vs Realized Vol divergence
        realized_vol = self._realized_vol_cache.get(symbol)
        if realized_vol and realized_vol > 0:
            signals.extend(self._iv_rv_signals(chain, realized_vol, spot))

        # Strategy 2: Vol surface anomalies
        signals.extend(self._surface_anomaly_signals(chain))

        # Strategy 3: Skew mean-reversion
        signals.extend(self._skew_signals(chain, spot))

        # Filter weak signals and sort by strength
        signals = [s for s in signals if s.strength >= self.min_signal_strength]
        signals.sort(key=lambda s: s.strength, reverse=True)

        return signals[: self.max_signals]

    def _iv_rv_signals(
        self, chain: OptionsChain, realized_vol: float, spot: float
    ) -> list[Signal]:
        """Generate signals from IV vs realized vol divergence."""
        signals = []

        for tick in chain.all_ticks:
            if tick.iv is None or tick.iv <= 0:
                continue

            ratio = tick.iv / realized_vol

            if ratio > self.iv_rv_sell_threshold:
                # IV is too high — sell vol
                strength = min((ratio - 1.0) * 5, 10.0)
                signals.append(Signal(
                    signal_type=SignalType.SELL_VOL,
                    symbol=chain.symbol,
                    strike=tick.strike,
                    expiry=tick.expiry,
                    option_type=tick.option_type,
                    strength=round(strength, 2),
                    iv=tick.iv,
                    realized_vol=realized_vol,
                    metadata={"iv_rv_ratio": round(ratio, 4)},
                ))

            elif ratio < self.iv_rv_buy_threshold:
                # IV is too low — buy vol
                strength = min((1.0 / ratio - 1.0) * 5, 10.0)
                signals.append(Signal(
                    signal_type=SignalType.BUY_VOL,
                    symbol=chain.symbol,
                    strike=tick.strike,
                    expiry=tick.expiry,
                    option_type=tick.option_type,
                    strength=round(strength, 2),
                    iv=tick.iv,
                    realized_vol=realized_vol,
                    metadata={"iv_rv_ratio": round(ratio, 4)},
                ))

        return signals

    def _surface_anomaly_signals(self, chain: OptionsChain) -> list[Signal]:
        """Generate signals from vol surface anomalies."""
        min_vol = self.config.get("min_option_volume", 100)
        anomalies = self.vol_surface.detect_anomalies(min_volume=min_vol)
        signals = []

        for anomaly in anomalies:
            strength = min(anomaly.confidence * 8, 10.0)
            option_type = OptionType.CALL  # Will be refined by execution

            signals.append(Signal(
                signal_type=SignalType.SURFACE_ARB,
                symbol=chain.symbol,
                strike=anomaly.strike,
                expiry=anomaly.expiry,
                option_type=option_type,
                strength=round(strength, 2),
                iv=anomaly.iv,
                expected_iv=anomaly.expected_iv,
                deviation=anomaly.deviation,
                metadata={
                    "deviation_pct": anomaly.deviation_pct,
                    "confidence": anomaly.confidence,
                },
            ))

        return signals

    def _skew_signals(self, chain: OptionsChain, spot: float) -> list[Signal]:
        """Generate signals from extreme skew levels."""
        signals = []
        atm_strike = chain.get_atm_strike()
        if not atm_strike:
            return signals

        for expiry in chain.get_expirations():
            skew_info = self.vol_surface.get_skew(expiry, atm_strike)
            if not skew_info:
                continue

            skew_val = skew_info.get("skew", 0)
            # If skew is extreme (> 2 standard deviations from typical)
            # This is a simplified check — in production, use rolling z-scores
            if abs(skew_val) > 0.05:  # 5% skew is significant
                direction = SignalType.SKEW_REVERT
                strength = min(abs(skew_val) * 50, 10.0)

                signals.append(Signal(
                    signal_type=direction,
                    symbol=chain.symbol,
                    strike=atm_strike,
                    expiry=expiry,
                    option_type=OptionType.PUT if skew_val > 0 else OptionType.CALL,
                    strength=round(strength, 2),
                    metadata={"skew": round(skew_val, 4), "atm_iv": skew_info.get("atm_iv")},
                ))

        return signals

    def _update_vol_surface(self, chain: OptionsChain, spot: float) -> None:
        """Recompute IVs and update the vol surface from the chain."""
        for tick in chain.all_ticks:
            if tick.mid <= 0:
                continue

            # Compute time to expiry using the tick's own timestamp
            try:
                from datetime import datetime
                exp_date = datetime.strptime(tick.expiry, "%Y-%m-%d")
                # Use the tick's timestamp instead of now() for backtesting
                if tick.timestamp and tick.timestamp > 0:
                    now = datetime.fromtimestamp(tick.timestamp)
                else:
                    now = datetime.now()
                T = max((exp_date - now).days / 365.0, 1 / 365.0)
            except (ValueError, OSError):
                continue

            iv = self.iv_solver.solve(
                market_price=tick.mid,
                spot=spot,
                strike=tick.strike,
                time_to_expiry=T,
                option_type=tick.option_type.value,
            )

            if iv is not None:
                tick.iv = iv
                self.vol_surface.update(SurfacePoint(
                    expiry=tick.expiry,
                    strike=tick.strike,
                    iv=iv,
                    volume=tick.volume,
                    open_interest=tick.open_interest,
                    timestamp=tick.timestamp,
                ))

    def _compute_realized_vol(self, symbol: str) -> float:
        """Compute annualized realized volatility from price history."""
        prices = self._price_history.get(symbol, [])
        window = self.config.get("realized_vol_window", 20)

        if len(prices) < window + 1:
            return 0.0

        recent = prices[-(window + 1):]
        log_returns = np.diff(np.log(recent))
        return float(np.std(log_returns) * np.sqrt(252))  # Annualized
