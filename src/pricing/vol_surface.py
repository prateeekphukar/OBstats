"""
Volatility Surface Construction and Anomaly Detection.

Builds and maintains an implied volatility surface across strikes and
expirations. Detects anomalies (mispriced strikes) that represent
potential trading opportunities.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SurfacePoint:
    """A single point on the volatility surface."""
    expiry: str          # Expiration date string (e.g., '2025-03-21')
    strike: float
    iv: float
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    volume: int = 0
    open_interest: int = 0
    timestamp: float = 0.0


@dataclass
class SurfaceAnomaly:
    """A detected anomaly in the volatility surface."""
    expiry: str
    strike: float
    iv: float
    expected_iv: float
    deviation: float
    deviation_pct: float
    confidence: float    # 0-1, based on liquidity and deviation magnitude


class VolSurface:
    """
    Implied Volatility Surface manager.

    Maintains a 2D grid of IV values indexed by (expiry, strike).
    Provides methods for querying, interpolation, and anomaly detection.
    """

    def __init__(self, anomaly_threshold: float = 0.02):
        """
        Args:
            anomaly_threshold: Minimum absolute IV deviation from neighbors
                               to flag as an anomaly.
        """
        self.anomaly_threshold = anomaly_threshold
        self._surface: dict[tuple[str, float], SurfacePoint] = {}
        self._expirations: set[str] = set()
        self._history: list[dict] = []  # Track surface snapshots for analysis

    def update(self, point: SurfacePoint) -> None:
        """Add or update a point on the surface."""
        key = (point.expiry, point.strike)
        self._surface[key] = point
        self._expirations.add(point.expiry)

    def update_batch(self, points: list[SurfacePoint]) -> None:
        """Update multiple points at once."""
        for point in points:
            self.update(point)

    def get_iv(self, expiry: str, strike: float) -> Optional[float]:
        """Get IV for a specific (expiry, strike) pair."""
        point = self._surface.get((expiry, strike))
        return point.iv if point else None

    def get_smile(self, expiry: str) -> list[SurfacePoint]:
        """
        Get the volatility smile for a specific expiration.
        Returns points sorted by strike.
        """
        points = [
            point for (exp, _), point in self._surface.items()
            if exp == expiry
        ]
        return sorted(points, key=lambda p: p.strike)

    def get_term_structure(self, strike: float) -> list[SurfacePoint]:
        """
        Get the term structure (IV across expirations) for a specific strike.
        Returns points sorted by expiry.
        """
        points = [
            point for (_, k), point in self._surface.items()
            if k == strike
        ]
        return sorted(points, key=lambda p: p.expiry)

    def get_skew(self, expiry: str, atm_strike: float) -> dict:
        """
        Compute skew metrics for a given expiration.

        Returns:
            Dict with 25-delta skew, skew slope, and curvature (smile).
        """
        smile = self.get_smile(expiry)
        if len(smile) < 3:
            return {}

        strikes = [p.strike for p in smile]
        ivs = [p.iv for p in smile]

        # Find ATM index
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm_strike))
        atm_iv = ivs[atm_idx]

        # OTM put IV (lower strikes) vs OTM call IV (higher strikes)
        put_ivs = [iv for s, iv in zip(strikes, ivs) if s < atm_strike]
        call_ivs = [iv for s, iv in zip(strikes, ivs) if s > atm_strike]

        skew = {}
        if put_ivs and call_ivs:
            skew["put_wing_avg"] = np.mean(put_ivs)
            skew["call_wing_avg"] = np.mean(call_ivs)
            skew["skew"] = np.mean(put_ivs) - np.mean(call_ivs)  # Positive = normal skew
            skew["smile"] = (np.mean(put_ivs) + np.mean(call_ivs)) / 2 - atm_iv  # Curvature
            skew["atm_iv"] = atm_iv

        return skew

    def detect_anomalies(self, min_volume: int = 0) -> list[SurfaceAnomaly]:
        """
        Detect anomalies in the volatility surface.

        An anomaly is a strike whose IV deviates significantly from its
        neighbors' interpolated value (i.e., the vol smile is locally broken).

        Args:
            min_volume: Minimum volume for a point to be considered.

        Returns:
            List of SurfaceAnomaly objects, sorted by confidence (descending).
        """
        anomalies = []

        for expiry in self._expirations:
            smile = self.get_smile(expiry)
            # Filter by volume
            if min_volume > 0:
                smile = [p for p in smile if p.volume >= min_volume]

            if len(smile) < 3:
                continue

            for i in range(1, len(smile) - 1):
                prev_iv = smile[i - 1].iv
                curr_iv = smile[i].iv
                next_iv = smile[i + 1].iv

                # Linear interpolation between neighbors
                strike_range = smile[i + 1].strike - smile[i - 1].strike
                if strike_range <= 0:
                    continue

                weight = (smile[i].strike - smile[i - 1].strike) / strike_range
                expected_iv = prev_iv + weight * (next_iv - prev_iv)

                deviation = curr_iv - expected_iv

                if abs(deviation) > self.anomaly_threshold:
                    # Confidence based on deviation magnitude and liquidity
                    deviation_confidence = min(abs(deviation) / (self.anomaly_threshold * 3), 1.0)
                    liquidity_confidence = min(smile[i].volume / 500, 1.0) if smile[i].volume > 0 else 0.3
                    confidence = 0.6 * deviation_confidence + 0.4 * liquidity_confidence

                    anomalies.append(SurfaceAnomaly(
                        expiry=expiry,
                        strike=smile[i].strike,
                        iv=curr_iv,
                        expected_iv=round(expected_iv, 6),
                        deviation=round(deviation, 6),
                        deviation_pct=round(deviation / expected_iv * 100, 2) if expected_iv > 0 else 0,
                        confidence=round(confidence, 4),
                    ))

        return sorted(anomalies, key=lambda a: a.confidence, reverse=True)

    def snapshot(self) -> dict[str, list[dict]]:
        """Take a snapshot of the current surface for logging/analysis."""
        result = {}
        for expiry in sorted(self._expirations):
            smile = self.get_smile(expiry)
            result[expiry] = [
                {"strike": p.strike, "iv": p.iv, "volume": p.volume, "oi": p.open_interest}
                for p in smile
            ]
        return result

    @property
    def expirations(self) -> list[str]:
        """All tracked expirations, sorted."""
        return sorted(self._expirations)

    @property
    def size(self) -> int:
        """Total number of points on the surface."""
        return len(self._surface)

    def clear(self) -> None:
        """Reset the surface."""
        self._surface.clear()
        self._expirations.clear()
