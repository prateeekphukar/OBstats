"""
Technical Indicator Engine — Computes all indicators on OHLCV data.

Calculates RSI, MACD, VWAP, Supertrend, ADX, Bollinger Bands, EMA crossovers,
and volume spikes. Each indicator returns a normalized score (0-100) for
the AI signal aggregator.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def _safe(val):
    """Convert numpy/pandas types to native Python, handle NaN."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


class TechnicalIndicators:
    """Compute all technical indicators and return normalized scores."""

    def __init__(self, config: dict):
        ind = config.get("indicators", {})
        self.rsi_period = ind.get("rsi_period", 14)
        self.macd_fast = ind.get("macd_fast", 12)
        self.macd_slow = ind.get("macd_slow", 26)
        self.macd_signal = ind.get("macd_signal", 9)
        self.st_period = ind.get("supertrend_period", 10)
        self.st_mult = ind.get("supertrend_multiplier", 3.0)
        self.adx_period = ind.get("adx_period", 14)
        self.bb_period = ind.get("bb_period", 20)
        self.bb_std = ind.get("bb_std", 2.0)
        self.ema_fast = ind.get("ema_fast", 20)
        self.ema_slow = ind.get("ema_slow", 50)
        self.vol_avg_period = ind.get("volume_avg_period", 20)
        self.vol_spike_thresh = ind.get("volume_spike_threshold", 1.5)

    # ── Core indicator calculations ──

    def compute_rsi(self, close: pd.Series) -> pd.Series:
        """Relative Strength Index (Wilder's method)."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, and histogram."""
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (session-reset)."""
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vol = df["Volume"].replace(0, np.nan)
        cum_tp_vol = (tp * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap = cum_tp_vol / cum_vol
        return vwap

    def compute_supertrend(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Supertrend indicator. Returns (supertrend_value, direction: 1=up/-1=down)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # ATR
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": (high - close.shift(1)).abs(),
            "lc": (low - close.shift(1)).abs(),
        }).max(axis=1)
        atr = tr.ewm(span=self.st_period, adjust=False).mean()

        hl2 = (high + low) / 2
        upper_band = hl2 + self.st_mult * atr
        lower_band = hl2 - self.st_mult * atr

        supertrend = pd.Series(np.nan, index=df.index)
        direction = pd.Series(1, index=df.index)

        for i in range(1, len(df)):
            # Upper band
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                pass
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            # Lower band
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                pass
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]

            if i == 1:
                direction.iloc[i] = 1
            elif supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                direction.iloc[i] = -1 if close.iloc[i] > upper_band.iloc[i] else 1 if close.iloc[i] < lower_band.iloc[i] else direction.iloc[i-1] if direction.iloc[i-1] == -1 else -1
            else:
                direction.iloc[i] = 1 if close.iloc[i] < lower_band.iloc[i] else -1 if close.iloc[i] > upper_band.iloc[i] else direction.iloc[i-1] if direction.iloc[i-1] == 1 else 1

            # Simplified logic
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                supertrend.iloc[i] = lower_band.iloc[i]

        # Simplified supertrend: 1 = bullish (price > ST), -1 = bearish
        final_dir = pd.Series(np.where(close > supertrend, 1, -1), index=df.index)
        return supertrend, final_dir

    def compute_adx(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX, +DI, -DI."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.DataFrame({
            "hl": high - low,
            "hc": (high - close.shift(1)).abs(),
            "lc": (low - close.shift(1)).abs(),
        }).max(axis=1)

        atr = tr.ewm(span=self.adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=self.adx_period, adjust=False).mean()

        return adx, plus_di, minus_di

    def compute_bollinger(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands — upper, middle (SMA), lower."""
        middle = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        return upper, middle, lower

    def compute_ema(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Fast and slow EMAs."""
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
        return ema_f, ema_s

    def compute_volume_spike(self, volume: pd.Series) -> pd.Series:
        """Volume relative to 20-period average."""
        avg_vol = volume.rolling(window=self.vol_avg_period).mean()
        ratio = volume / avg_vol.replace(0, np.nan)
        return ratio

    # ── Scoring functions (each returns 0-100) ──

    def score_rsi(self, rsi_val: float) -> float:
        """RSI score: 0=extreme sell, 50=neutral, 100=extreme buy zone."""
        if rsi_val is None:
            return 50.0
        # Bullish: RSI 40-65 is ideal buying zone
        # Bearish: RSI > 80 or < 20
        if 55 <= rsi_val <= 70:
            return 80.0  # Strong buy zone
        elif 40 <= rsi_val < 55:
            return 60.0  # Mild buy
        elif 30 <= rsi_val < 40:
            return 40.0  # Neutral-weak
        elif rsi_val < 30:
            return 25.0  # Oversold — potential reversal buy
        elif 70 < rsi_val <= 80:
            return 35.0  # Overbought — caution
        else:
            return 15.0  # Extreme overbought — sell signal

    def score_macd(self, macd_line: float, signal_line: float, histogram: float) -> float:
        """MACD score based on crossover and histogram."""
        if any(v is None for v in [macd_line, signal_line, histogram]):
            return 50.0
        score = 50.0
        if macd_line > signal_line:
            score += 25  # Bullish crossover
        else:
            score -= 25

        if histogram > 0 and histogram > abs(macd_line) * 0.05:
            score += 15  # Expanding bullish histogram
        elif histogram < 0 and abs(histogram) > abs(macd_line) * 0.05:
            score -= 15
        return max(0, min(100, score))

    def score_vwap(self, price: float, vwap: float) -> float:
        """Price vs VWAP — above = bullish."""
        if vwap is None or vwap == 0:
            return 50.0
        pct_diff = ((price - vwap) / vwap) * 100
        if pct_diff > 0.5:
            return 80.0
        elif pct_diff > 0:
            return 65.0
        elif pct_diff > -0.5:
            return 35.0
        else:
            return 20.0

    def score_supertrend(self, direction: float) -> float:
        """Supertrend direction — 1=bullish, -1=bearish."""
        if direction is None:
            return 50.0
        return 85.0 if direction > 0 else 15.0

    def score_adx(self, adx_val: float, plus_di: float, minus_di: float) -> float:
        """ADX trend strength + DI direction."""
        if any(v is None for v in [adx_val, plus_di, minus_di]):
            return 50.0
        score = 50.0
        # Trend strength
        if adx_val > 25:
            # Trending — direction matters
            if plus_di > minus_di:
                score = 70.0 + min(20, (adx_val - 25))  # up to 90
            else:
                score = 30.0 - min(20, (adx_val - 25))  # down to 10
        else:
            score = 50.0  # Ranging — neutral
        return max(0, min(100, score))

    def score_volume(self, vol_ratio: float) -> float:
        """Volume spike score — higher volume = stronger signal."""
        if vol_ratio is None:
            return 50.0
        if vol_ratio > 2.0:
            return 90.0
        elif vol_ratio > 1.5:
            return 75.0
        elif vol_ratio > 1.0:
            return 55.0
        elif vol_ratio > 0.5:
            return 40.0
        else:
            return 25.0

    def score_ema(self, price: float, ema_fast: float, ema_slow: float) -> float:
        """EMA alignment — price above both EMAs + golden cross."""
        if any(v is None for v in [price, ema_fast, ema_slow]):
            return 50.0
        score = 50.0
        if price > ema_fast > ema_slow:
            score = 85.0  # Perfect bullish alignment
        elif price > ema_fast:
            score = 65.0
        elif price < ema_fast < ema_slow:
            score = 15.0  # Perfect bearish alignment
        elif price < ema_fast:
            score = 35.0
        return score

    def score_bollinger(self, price: float, upper: float, middle: float, lower: float) -> float:
        """Bollinger Band position — breakout or squeeze."""
        if any(v is None for v in [price, upper, middle, lower]):
            return 50.0
        if price > upper:
            return 80.0  # Breakout (bullish momentum)
        elif price > middle:
            band_pct = (price - middle) / (upper - middle) if upper != middle else 0
            return 50 + band_pct * 25
        elif price > lower:
            band_pct = (middle - price) / (middle - lower) if middle != lower else 0
            return 50 - band_pct * 25
        else:
            return 20.0  # Below lower band (bearish or oversold)

    # ── Master computation ──

    def compute_all(self, df: pd.DataFrame) -> dict:
        """
        Compute all indicators and scores for a stock's OHLCV DataFrame.

        Returns dict with raw indicator values and normalized scores (0-100).
        """
        if df is None or len(df) < self.ema_slow + 5:
            return {"error": "Insufficient data", "scores": {}, "raw": {}}

        close = df["Close"]
        latest_price = float(close.iloc[-1])

        # Compute raw indicators
        rsi = self.compute_rsi(close)
        macd_line, signal_line, histogram = self.compute_macd(close)
        vwap = self.compute_vwap(df)
        st_val, st_dir = self.compute_supertrend(df)
        adx, plus_di, minus_di = self.compute_adx(df)
        bb_upper, bb_middle, bb_lower = self.compute_bollinger(close)
        ema_f, ema_s = self.compute_ema(close)
        vol_ratio = self.compute_volume_spike(df["Volume"])

        # Get latest values
        rsi_val = _safe(rsi.iloc[-1])
        macd_val = _safe(macd_line.iloc[-1])
        signal_val = _safe(signal_line.iloc[-1])
        hist_val = _safe(histogram.iloc[-1])
        vwap_val = _safe(vwap.iloc[-1])
        st_dir_val = _safe(st_dir.iloc[-1])
        adx_val = _safe(adx.iloc[-1])
        pdi_val = _safe(plus_di.iloc[-1])
        mdi_val = _safe(minus_di.iloc[-1])
        bb_u = _safe(bb_upper.iloc[-1])
        bb_m = _safe(bb_middle.iloc[-1])
        bb_l = _safe(bb_lower.iloc[-1])
        ema_f_val = _safe(ema_f.iloc[-1])
        ema_s_val = _safe(ema_s.iloc[-1])
        vol_ratio_val = _safe(vol_ratio.iloc[-1])

        # Compute scores
        scores = {
            "rsi": self.score_rsi(rsi_val),
            "macd": self.score_macd(macd_val, signal_val, hist_val),
            "vwap": self.score_vwap(latest_price, vwap_val),
            "supertrend": self.score_supertrend(st_dir_val),
            "adx": self.score_adx(adx_val, pdi_val, mdi_val),
            "volume": self.score_volume(vol_ratio_val),
            "ema": self.score_ema(latest_price, ema_f_val, ema_s_val),
            "bollinger": self.score_bollinger(latest_price, bb_u, bb_m, bb_l),
        }

        raw = {
            "price": latest_price,
            "rsi": round(rsi_val, 2) if rsi_val else None,
            "macd": round(macd_val, 3) if macd_val else None,
            "macd_signal": round(signal_val, 3) if signal_val else None,
            "macd_histogram": round(hist_val, 3) if hist_val else None,
            "vwap": round(vwap_val, 2) if vwap_val else None,
            "supertrend_dir": "BULLISH" if st_dir_val and st_dir_val > 0 else "BEARISH",
            "adx": round(adx_val, 2) if adx_val else None,
            "plus_di": round(pdi_val, 2) if pdi_val else None,
            "minus_di": round(mdi_val, 2) if mdi_val else None,
            "bb_upper": round(bb_u, 2) if bb_u else None,
            "bb_middle": round(bb_m, 2) if bb_m else None,
            "bb_lower": round(bb_l, 2) if bb_l else None,
            "ema_20": round(ema_f_val, 2) if ema_f_val else None,
            "ema_50": round(ema_s_val, 2) if ema_s_val else None,
            "volume_ratio": round(vol_ratio_val, 2) if vol_ratio_val else None,
        }

        return {"scores": scores, "raw": raw}
