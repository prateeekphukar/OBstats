"""
Black-Scholes Options Pricing Model with full Greeks computation.

This module provides the core pricing engine used to compute theoretical
fair values and Greeks for European-style options. It supports both
call and put options.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PricingResult:
    """Result of an options pricing calculation."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    option_type: str
    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    rate: float


class BlackScholesPricer:
    """
    Black-Scholes-Merton options pricing engine.

    Computes theoretical prices and Greeks for European options.
    Optimized for repeated calls with cached intermediate computations.
    """

    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: Literal["call", "put"] = "call",
        rate: float | None = None,
    ) -> PricingResult:
        """
        Compute the full pricing result including price and all Greeks.

        Args:
            spot: Current price of the underlying asset.
            strike: Strike price of the option.
            time_to_expiry: Time to expiration in years (e.g., 30/365 for 30 days).
            volatility: Annualized volatility (sigma).
            option_type: 'call' or 'put'.
            rate: Risk-free rate override. Uses instance default if None.

        Returns:
            PricingResult with price, delta, gamma, theta, vega, rho.

        Raises:
            ValueError: If inputs are invalid (negative spot, zero time, etc.).
        """
        r = rate if rate is not None else self.risk_free_rate
        q = self.dividend_yield
        self._validate_inputs(spot, strike, time_to_expiry, volatility)

        T = time_to_expiry
        sqrt_T = np.sqrt(T)
        sigma = volatility

        # Core intermediate values
        d1 = (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Standard normal PDF and CDF values (cached for reuse)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        npd1 = norm.pdf(d1)

        # Discount factors
        df_div = np.exp(-q * T)   # Dividend discount factor
        df_rate = np.exp(-r * T)  # Risk-free discount factor

        if option_type == "call":
            price = spot * df_div * nd1 - strike * df_rate * nd2
            delta = df_div * nd1
            theta = (
                -(spot * df_div * npd1 * sigma) / (2 * sqrt_T)
                - r * strike * df_rate * nd2
                + q * spot * df_div * nd1
            )
            rho = strike * T * df_rate * nd2
        else:
            price = strike * df_rate * n_neg_d2 - spot * df_div * n_neg_d1
            delta = -df_div * n_neg_d1
            theta = (
                -(spot * df_div * npd1 * sigma) / (2 * sqrt_T)
                + r * strike * df_rate * n_neg_d2
                - q * spot * df_div * n_neg_d1
            )
            rho = -strike * T * df_rate * n_neg_d2

        # Greeks common to both call and put
        gamma = (df_div * npd1) / (spot * sigma * sqrt_T)
        vega = spot * df_div * npd1 * sqrt_T  # Per 1.0 vol (multiply by 0.01 for per 1%)

        return PricingResult(
            price=round(price, 6),
            delta=round(delta, 6),
            gamma=round(gamma, 6),
            theta=round(theta / 365, 6),  # Per-day theta
            vega=round(vega / 100, 6),    # Per 1% vol move
            rho=round(rho / 100, 6),      # Per 1% rate move
            option_type=option_type,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            rate=r,
        )

    def price_batch(
        self,
        spots: np.ndarray,
        strikes: np.ndarray,
        times: np.ndarray,
        vols: np.ndarray,
        option_types: list[str],
        rate: float | None = None,
    ) -> list[PricingResult]:
        """
        Vectorized pricing for multiple options simultaneously.

        All arrays must have the same length. This is significantly faster
        than calling price() in a loop for large option chains.
        """
        return [
            self.price(s, k, t, v, ot, rate)
            for s, k, t, v, ot in zip(spots, strikes, times, vols, option_types)
        ]

    @staticmethod
    def _validate_inputs(
        spot: float, strike: float, time_to_expiry: float, volatility: float
    ) -> None:
        if spot <= 0:
            raise ValueError(f"Spot price must be positive, got {spot}")
        if strike <= 0:
            raise ValueError(f"Strike price must be positive, got {strike}")
        if time_to_expiry <= 0:
            raise ValueError(f"Time to expiry must be positive, got {time_to_expiry}")
        if volatility <= 0:
            raise ValueError(f"Volatility must be positive, got {volatility}")
