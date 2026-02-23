"""
Unit tests for the pricing engine.

Tests Black-Scholes pricing, Greeks, and IV solver against known values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from src.pricing.black_scholes import BlackScholesPricer, PricingResult
from src.pricing.iv_solver import ImpliedVolSolver
from src.pricing.vol_surface import VolSurface, SurfacePoint


# ── Black-Scholes Tests ───────────────────────────────


class TestBlackScholesPricer:
    """Tests for Black-Scholes pricing model."""

    def setup_method(self):
        self.pricer = BlackScholesPricer(risk_free_rate=0.05, dividend_yield=0.0)

    def test_call_price_basic(self):
        """ATM call option should have a reasonable price."""
        result = self.pricer.price(
            spot=100, strike=100, time_to_expiry=1.0, volatility=0.20
        )
        assert result.price > 0
        assert result.price < 100  # Can't exceed spot
        # Known value: ~$10.45 for 20% vol, 1yr ATM call
        assert 9.0 < result.price < 12.0

    def test_put_price_basic(self):
        """ATM put option should have a reasonable price."""
        result = self.pricer.price(
            spot=100, strike=100, time_to_expiry=1.0, volatility=0.20,
            option_type="put"
        )
        assert result.price > 0
        assert result.price < 100

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*e^(-rT)."""
        S, K, T, r, vol = 100, 100, 1.0, 0.05, 0.30
        call = self.pricer.price(S, K, T, vol, "call")
        put = self.pricer.price(S, K, T, vol, "put")

        expected = S - K * np.exp(-r * T)
        actual = call.price - put.price
        assert abs(actual - expected) < 0.01

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call should approach intrinsic value."""
        result = self.pricer.price(
            spot=200, strike=100, time_to_expiry=0.01, volatility=0.20
        )
        intrinsic = 200 - 100
        assert abs(result.price - intrinsic) < 2.0

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call should be nearly worthless."""
        result = self.pricer.price(
            spot=50, strike=100, time_to_expiry=0.01, volatility=0.20
        )
        assert result.price < 0.01

    def test_delta_call_range(self):
        """Call delta should be between 0 and 1."""
        result = self.pricer.price(spot=100, strike=100, time_to_expiry=0.5, volatility=0.25)
        assert 0 < result.delta < 1

    def test_delta_put_range(self):
        """Put delta should be between -1 and 0."""
        result = self.pricer.price(
            spot=100, strike=100, time_to_expiry=0.5, volatility=0.25,
            option_type="put"
        )
        assert -1 < result.delta < 0

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        for opt_type in ["call", "put"]:
            result = self.pricer.price(
                spot=100, strike=100, time_to_expiry=0.5, volatility=0.25,
                option_type=opt_type
            )
            assert result.gamma > 0

    def test_vega_positive(self):
        """Vega should always be positive (for both calls and puts)."""
        for opt_type in ["call", "put"]:
            result = self.pricer.price(
                spot=100, strike=100, time_to_expiry=0.5, volatility=0.25,
                option_type=opt_type
            )
            assert result.vega > 0

    def test_higher_vol_higher_price(self):
        """Higher volatility should produce higher prices."""
        low_vol = self.pricer.price(spot=100, strike=100, time_to_expiry=0.5, volatility=0.10)
        high_vol = self.pricer.price(spot=100, strike=100, time_to_expiry=0.5, volatility=0.50)
        assert high_vol.price > low_vol.price

    def test_longer_time_higher_price(self):
        """Longer time to expiry should produce higher call prices."""
        short = self.pricer.price(spot=100, strike=100, time_to_expiry=0.1, volatility=0.25)
        long = self.pricer.price(spot=100, strike=100, time_to_expiry=1.0, volatility=0.25)
        assert long.price > short.price

    def test_invalid_inputs_raise(self):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError):
            self.pricer.price(spot=-100, strike=100, time_to_expiry=0.5, volatility=0.25)
        with pytest.raises(ValueError):
            self.pricer.price(spot=100, strike=-100, time_to_expiry=0.5, volatility=0.25)
        with pytest.raises(ValueError):
            self.pricer.price(spot=100, strike=100, time_to_expiry=-0.5, volatility=0.25)
        with pytest.raises(ValueError):
            self.pricer.price(spot=100, strike=100, time_to_expiry=0.5, volatility=-0.25)

    def test_pricing_result_fields(self):
        """PricingResult should have all expected fields."""
        result = self.pricer.price(spot=100, strike=100, time_to_expiry=0.5, volatility=0.25)
        assert isinstance(result, PricingResult)
        assert hasattr(result, "price")
        assert hasattr(result, "delta")
        assert hasattr(result, "gamma")
        assert hasattr(result, "theta")
        assert hasattr(result, "vega")
        assert hasattr(result, "rho")


# ── Implied Volatility Tests ──────────────────────────


class TestImpliedVolSolver:
    """Tests for the implied volatility solver."""

    def setup_method(self):
        self.pricer = BlackScholesPricer(risk_free_rate=0.05, dividend_yield=0.0)
        self.solver = ImpliedVolSolver(self.pricer)

    def test_round_trip_call(self):
        """Price with known vol → solve IV → should recover the vol."""
        known_vol = 0.25
        result = self.pricer.price(
            spot=100, strike=100, time_to_expiry=0.5, volatility=known_vol
        )
        solved_iv = self.solver.solve(
            market_price=result.price, spot=100, strike=100,
            time_to_expiry=0.5, option_type="call"
        )
        assert solved_iv is not None
        assert abs(solved_iv - known_vol) < 0.001

    def test_round_trip_put(self):
        """IV round-trip for puts."""
        known_vol = 0.30
        result = self.pricer.price(
            spot=100, strike=100, time_to_expiry=0.5,
            volatility=known_vol, option_type="put"
        )
        solved_iv = self.solver.solve(
            market_price=result.price, spot=100, strike=100,
            time_to_expiry=0.5, option_type="put"
        )
        assert solved_iv is not None
        assert abs(solved_iv - known_vol) < 0.001

    def test_otm_option(self):
        """IV solver should work for OTM options."""
        result = self.pricer.price(
            spot=100, strike=110, time_to_expiry=0.25, volatility=0.20
        )
        solved_iv = self.solver.solve(
            market_price=result.price, spot=100, strike=110,
            time_to_expiry=0.25
        )
        assert solved_iv is not None
        assert abs(solved_iv - 0.20) < 0.001

    def test_negative_price_returns_none(self):
        """Negative market price should return None."""
        result = self.solver.solve(
            market_price=-1.0, spot=100, strike=100, time_to_expiry=0.5
        )
        assert result is None

    def test_zero_time_returns_none(self):
        """Zero time to expiry should return None."""
        result = self.solver.solve(
            market_price=5.0, spot=100, strike=100, time_to_expiry=0
        )
        assert result is None

    def test_high_vol_option(self):
        """Should solve for high-vol options (e.g., meme stocks)."""
        known_vol = 2.0  # 200% vol
        result = self.pricer.price(
            spot=100, strike=100, time_to_expiry=0.25, volatility=known_vol
        )
        solved_iv = self.solver.solve(
            market_price=result.price, spot=100, strike=100,
            time_to_expiry=0.25
        )
        assert solved_iv is not None
        assert abs(solved_iv - known_vol) < 0.01


# ── Volatility Surface Tests ─────────────────────────


class TestVolSurface:
    """Tests for the volatility surface."""

    def setup_method(self):
        self.surface = VolSurface(anomaly_threshold=0.02)

    def test_update_and_retrieve(self):
        """Should store and retrieve IV points."""
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=100, iv=0.25))
        result = self.surface.get_iv("2025-03-21", 100)
        assert result == 0.25

    def test_missing_point_returns_none(self):
        """Missing points should return None."""
        result = self.surface.get_iv("2025-03-21", 999)
        assert result is None

    def test_get_smile(self):
        """Should return sorted smile for an expiry."""
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=105, iv=0.22))
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=100, iv=0.25))
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=95, iv=0.28))

        smile = self.surface.get_smile("2025-03-21")
        assert len(smile) == 3
        assert smile[0].strike == 95
        assert smile[1].strike == 100
        assert smile[2].strike == 105

    def test_detect_anomaly(self):
        """Should detect a strike with anomalous IV."""
        # Normal smile: 0.30, 0.25, 0.22, 0.20, 0.22
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=90, iv=0.30))
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=95, iv=0.25))
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=100, iv=0.35))  # ← Anomaly!
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=105, iv=0.20))
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=110, iv=0.22))

        anomalies = self.surface.detect_anomalies()
        assert len(anomalies) > 0
        # The anomaly at strike 100 should be detected
        anomaly_strikes = [a.strike for a in anomalies]
        assert 100 in anomaly_strikes

    def test_no_anomaly_smooth_smile(self):
        """Smooth smile should produce no anomalies."""
        ivs = [0.30, 0.27, 0.25, 0.24, 0.25, 0.27, 0.30]
        strikes = [85, 90, 95, 100, 105, 110, 115]
        for s, iv in zip(strikes, ivs):
            self.surface.update(SurfacePoint(expiry="2025-03-21", strike=s, iv=iv))

        anomalies = self.surface.detect_anomalies()
        assert len(anomalies) == 0

    def test_expirations(self):
        """Should track all expirations."""
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=100, iv=0.25))
        self.surface.update(SurfacePoint(expiry="2025-04-18", strike=100, iv=0.24))

        exps = self.surface.expirations
        assert len(exps) == 2
        assert "2025-03-21" in exps
        assert "2025-04-18" in exps

    def test_size(self):
        """Size should reflect total points."""
        assert self.surface.size == 0
        self.surface.update(SurfacePoint(expiry="2025-03-21", strike=100, iv=0.25))
        assert self.surface.size == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
