"""
Implied Volatility Solver.

Computes implied volatility from market prices using a combination of
Newton-Raphson (fast convergence) with Brent's method fallback (guaranteed convergence).
"""

import numpy as np
from scipy.optimize import brentq
from typing import Literal

from .black_scholes import BlackScholesPricer


class ImpliedVolSolver:
    """
    Solve for implied volatility given a market price.

    Uses Newton-Raphson with vega as the derivative for fast convergence,
    with Brent's method as a robust fallback.
    """

    def __init__(
        self,
        pricer: BlackScholesPricer,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        vol_bounds: tuple[float, float] = (0.001, 10.0),
    ):
        self.pricer = pricer
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.vol_bounds = vol_bounds

    def solve(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: Literal["call", "put"] = "call",
        rate: float | None = None,
    ) -> float | None:
        """
        Compute implied volatility for a given market price.

        Args:
            market_price: Observed market price of the option.
            spot: Current underlying price.
            strike: Strike price.
            time_to_expiry: Time to expiry in years.
            option_type: 'call' or 'put'.
            rate: Risk-free rate override.

        Returns:
            Implied volatility as a decimal (e.g., 0.25 = 25%), or None if no solution.
        """
        # Quick sanity checks
        if market_price <= 0:
            return None
        if time_to_expiry <= 0:
            return None

        # Check intrinsic value bounds
        intrinsic = self._intrinsic_value(spot, strike, option_type, rate, time_to_expiry)
        if market_price < intrinsic - 0.01:
            return None  # Below intrinsic — no valid IV

        # Try Newton-Raphson first (faster)
        iv = self._newton_raphson(market_price, spot, strike, time_to_expiry, option_type, rate)
        if iv is not None:
            return iv

        # Fall back to Brent's method (guaranteed convergence)
        return self._brent_method(market_price, spot, strike, time_to_expiry, option_type, rate)

    def solve_chain(
        self,
        market_prices: list[float],
        spot: float,
        strikes: list[float],
        time_to_expiry: float,
        option_types: list[str],
        rate: float | None = None,
    ) -> list[float | None]:
        """Solve IV for an entire options chain."""
        return [
            self.solve(price, spot, strike, time_to_expiry, ot, rate)
            for price, strike, ot in zip(market_prices, strikes, option_types)
        ]

    def _newton_raphson(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: str,
        rate: float | None,
    ) -> float | None:
        """Newton-Raphson solver using vega as derivative."""
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / time_to_expiry) * market_price / spot

        # Clamp initial guess
        sigma = max(self.vol_bounds[0], min(sigma, self.vol_bounds[1]))

        for _ in range(self.max_iterations):
            try:
                result = self.pricer.price(spot, strike, time_to_expiry, sigma, option_type, rate)
            except ValueError:
                return None

            diff = result.price - market_price

            if abs(diff) < self.tolerance:
                return sigma

            # Vega in raw units (undo the /100 scaling from pricer)
            vega_raw = result.vega * 100
            if abs(vega_raw) < 1e-12:
                return None  # Vega too small — Newton won't converge

            sigma -= diff / vega_raw

            # Enforce bounds
            if sigma < self.vol_bounds[0] or sigma > self.vol_bounds[1]:
                return None  # Out of range — let Brent handle it

        return None  # Didn't converge

    def _brent_method(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: str,
        rate: float | None,
    ) -> float | None:
        """Brent's method — guaranteed convergence within bounds."""

        def objective(sigma: float) -> float:
            try:
                result = self.pricer.price(spot, strike, time_to_expiry, sigma, option_type, rate)
                return result.price - market_price
            except ValueError:
                return float("inf")

        try:
            return brentq(objective, self.vol_bounds[0], self.vol_bounds[1], xtol=self.tolerance)
        except (ValueError, RuntimeError):
            return None

    @staticmethod
    def _intrinsic_value(
        spot: float, strike: float, option_type: str, rate: float | None, T: float
    ) -> float:
        """Compute intrinsic value of the option."""
        r = rate if rate is not None else 0.05
        df = np.exp(-r * T)
        if option_type == "call":
            return max(0, spot - strike * df)
        else:
            return max(0, strike * df - spot)
