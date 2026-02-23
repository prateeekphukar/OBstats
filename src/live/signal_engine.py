"""
AI Signal Scoring Engine — Aggregates indicators into trade signals.

Takes individual indicator scores, applies configurable weights,
and produces final BUY/SELL/NEUTRAL signals with confidence levels,
entry prices, stop losses, and targets.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# Sector mapping for context
SECTOR_MAP = {
    "COALINDIA": "Mining / Coal / PSU",
    "CANBK": "Banking / PSU Bank",
    "HINDALCO": "Metals / Aluminium",
    "LT": "Capital Goods / Infra",
    "NTPC": "Power / Utilities / PSU",
    "ABB": "Capital Goods / Electrical / MNC",
    "TATAPOWER": "Power / Green Energy",
    "INDUSINDBK": "Private Banking",
    "JSWSTEEL": "Metals / Steel",
    "PNB": "Banking / PSU Bank",
    "RELIANCE": "Oil & Gas / Conglomerate",
    "HDFCBANK": "Private Banking",
    "ICICIBANK": "Private Banking",
    "SBIN": "Banking / PSU Bank",
    "TCS": "IT / Services",
    "INFY": "IT / Services",
    "BHARTIARTL": "Telecom",
    "ITC": "FMCG / Hotels",
    "KOTAKBANK": "Private Banking",
    "TATAMOTORS": "Automobiles",
    "BAJFINANCE": "NBFC / Financial",
    "MARUTI": "Automobiles",
    "WIPRO": "IT / Services",
    "AXISBANK": "Private Banking",
}


class SignalType:
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    NO_DATA = "NO DATA"


class TradeSignal:
    """A complete trade signal with all details."""

    def __init__(self, symbol: str, score: float, config: dict,
                 indicator_scores: dict, raw_indicators: dict, stock_info: dict):
        self.symbol = symbol
        self.score = round(score, 1)
        self.sector = SECTOR_MAP.get(symbol, "Unknown")
        self.indicator_scores = indicator_scores
        self.raw = raw_indicators
        self.stock_info = stock_info
        self.timestamp = datetime.now(IST).strftime("%H:%M:%S")

        thresholds = config.get("signals", {})
        self.strong_buy = thresholds.get("strong_buy_threshold", 70)
        self.buy_thresh = thresholds.get("buy_threshold", 55)
        self.neutral_low = thresholds.get("neutral_low", 40)
        self.sell_thresh = thresholds.get("sell_threshold", 25)

        risk = config.get("risk", {})
        self.sl_pct = risk.get("default_sl_pct", 1.0)
        self.target_pct = risk.get("default_target_pct", 2.0)

        self._compute()

    def _compute(self):
        """Derive signal type, levels, and recommendation."""
        price = self.raw.get("price", 0)

        # Signal type
        if self.score >= self.strong_buy:
            self.signal_type = SignalType.STRONG_BUY
            self.bias = "BULLISH"
            self.color = "#00e676"
            self.bg_color = "rgba(0, 230, 118, 0.1)"
        elif self.score >= self.buy_thresh:
            self.signal_type = SignalType.BUY
            self.bias = "BULLISH"
            self.color = "#69f0ae"
            self.bg_color = "rgba(105, 240, 174, 0.08)"
        elif self.score >= self.neutral_low:
            self.signal_type = SignalType.NEUTRAL
            self.bias = "NEUTRAL"
            self.color = "#ffd740"
            self.bg_color = "rgba(255, 215, 64, 0.08)"
        elif self.score >= self.sell_thresh:
            self.signal_type = SignalType.SELL
            self.bias = "BEARISH"
            self.color = "#ff6e40"
            self.bg_color = "rgba(255, 110, 64, 0.08)"
        else:
            self.signal_type = SignalType.STRONG_SELL
            self.bias = "BEARISH"
            self.color = "#ff1744"
            self.bg_color = "rgba(255, 23, 68, 0.1)"

        # Entry / SL / Target
        if self.bias == "BULLISH":
            self.entry = round(price * 1.001, 2)  # Slight above CMP
            self.stop_loss = round(price * (1 - self.sl_pct / 100), 2)
            self.target_1 = round(price * (1 + self.target_pct / 100), 2)
            self.target_2 = round(price * (1 + self.target_pct * 1.5 / 100), 2)
            self.option_type = "CE"
        elif self.bias == "BEARISH":
            self.entry = round(price * 0.999, 2)
            self.stop_loss = round(price * (1 + self.sl_pct / 100), 2)
            self.target_1 = round(price * (1 - self.target_pct / 100), 2)
            self.target_2 = round(price * (1 - self.target_pct * 1.5 / 100), 2)
            self.option_type = "PE"
        else:
            self.entry = price
            self.stop_loss = 0
            self.target_1 = 0
            self.target_2 = 0
            self.option_type = "—"

        # Generate reasons
        self.reasons = self._generate_reasons()

        # Cautions
        self.cautions = self._generate_cautions()

    def _generate_reasons(self) -> List[str]:
        """Generate human-readable reasons for the signal."""
        reasons = []
        raw = self.raw

        # RSI
        rsi = raw.get("rsi")
        if rsi:
            if 55 <= rsi <= 70:
                reasons.append(f"RSI at {rsi} — bullish momentum zone")
            elif rsi > 75:
                reasons.append(f"RSI at {rsi} — overbought, reversal risk")
            elif rsi < 30:
                reasons.append(f"RSI at {rsi} — oversold, bounce expected")

        # MACD
        if raw.get("macd") and raw.get("macd_signal"):
            if raw["macd"] > raw["macd_signal"]:
                reasons.append("MACD bullish crossover active")
            else:
                reasons.append("MACD bearish crossover active")

        # Supertrend
        if raw.get("supertrend_dir") == "BULLISH":
            reasons.append("Supertrend is BULLISH (buy signal)")
        elif raw.get("supertrend_dir") == "BEARISH":
            reasons.append("Supertrend is BEARISH (sell signal)")

        # VWAP
        vwap = raw.get("vwap")
        price = raw.get("price", 0)
        if vwap and price:
            if price > vwap:
                reasons.append(f"Trading ABOVE VWAP (₹{vwap})")
            else:
                reasons.append(f"Trading BELOW VWAP (₹{vwap})")

        # ADX
        adx = raw.get("adx")
        if adx:
            if adx > 25:
                reasons.append(f"ADX at {adx} — strong trend in play")
            else:
                reasons.append(f"ADX at {adx} — weak/ranging market")

        # Volume
        vol = raw.get("volume_ratio")
        if vol and vol > 1.5:
            reasons.append(f"Volume spike {vol}x above average")

        # EMA alignment
        ema_20 = raw.get("ema_20")
        ema_50 = raw.get("ema_50")
        if ema_20 and ema_50 and price:
            if price > ema_20 > ema_50:
                reasons.append("Price above EMA 20 & 50 — bullish alignment")
            elif price < ema_20 < ema_50:
                reasons.append("Price below EMA 20 & 50 — bearish alignment")

        # Options Data (PCR)
        options = self.stock_info.get("options", {})
        pcr = options.get("pcr")
        if pcr:
            if pcr > 1.2:
                reasons.append(f"Strong Put writing (PCR: {pcr:.2f}) — highly bullish setup")
            elif pcr > 1.0:
                reasons.append(f"Put writing exceeds Calls (PCR: {pcr:.2f}) — options data supports bulls")
            elif pcr < 0.6:
                reasons.append(f"Heavy Call writing (PCR: {pcr:.2f}) — highly bearish setup")
            elif pcr < 0.8:
                reasons.append(f"Call writing exceeds Puts (PCR: {pcr:.2f}) — options data supports bears")

        if not reasons:
            reasons.append("Mixed signals — no strong bias")

        return reasons

    def _generate_cautions(self) -> List[str]:
        """Generate caution messages."""
        cautions = []
        rsi = self.raw.get("rsi")
        adx = self.raw.get("adx")

        if rsi and rsi > 75:
            cautions.append("⚠️ Overbought — don't chase, wait for pullback")
        if rsi and rsi < 25:
            cautions.append("⚠️ Heavily oversold — may bounce but risky")
        if adx and adx < 20:
            cautions.append("⚠️ Weak trend — range-bound, avoid trending strategies")
        if self.signal_type == SignalType.NEUTRAL:
            cautions.append("🚫 No clear direction — CASH IS KING")

        # Options divergences
        options = self.stock_info.get("options", {})
        pcr = options.get("pcr")
        if pcr:
            if self.bias == "BULLISH" and pcr < 0.8:
                cautions.append(f"⚠️ Fakeout Risk: Price is bullish but Options OI is Bearish (PCR {pcr:.2f})")
            elif self.bias == "BEARISH" and pcr > 1.1:
                cautions.append(f"⚠️ Fakeout Risk: Price is bearish but Options OI is Bullish (PCR {pcr:.2f})")

        cautions.append("⚠️ Verify prices in pre-open session before trading")
        return cautions

    def to_dict(self) -> dict:
        """Serialize to dict for JSON API."""
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "score": self.score,
            "signal_type": self.signal_type,
            "bias": self.bias,
            "color": self.color,
            "bg_color": self.bg_color,
            "price": self.raw.get("price", 0),
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "option_type": self.option_type,
            "reasons": self.reasons,
            "cautions": self.cautions,
            "indicators": self.raw,
            "indicator_scores": self.indicator_scores,
            "stock_info": self.stock_info,
            "timestamp": self.timestamp,
        }


class AISignalEngine:
    """
    Aggregates indicator scores using weighted AI scoring.
    Produces actionable trade signals for each stock.
    """

    def __init__(self, config: dict):
        self.config = config
        weights = config.get("weights", {})
        self.weights = {
            "vwap": weights.get("vwap", 0.15),
            "rsi": weights.get("rsi", 0.15),
            "macd": weights.get("macd", 0.15),
            "supertrend": weights.get("supertrend", 0.15),
            "adx": weights.get("adx", 0.10),
            "volume": weights.get("volume", 0.10),
            "ema": weights.get("ema", 0.10),
            "bollinger": weights.get("bollinger", 0.10),
            "pcr": weights.get("pcr", 0.10),
        }
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

    def generate_signal(self, symbol: str, indicator_result: dict,
                        stock_info: dict) -> Optional[TradeSignal]:
        """
        Generate a single TradeSignal from indicator results.

        Args:
            symbol: Stock ticker
            indicator_result: Output from TechnicalIndicators.compute_all()
            stock_info: Output from LiveDataFetcher.get_stock_info()

        Returns:
            TradeSignal or None if insufficient data
        """
        if "error" in indicator_result:
            logger.warning(f"{symbol}: {indicator_result['error']}")
            return None

        scores = indicator_result.get("scores", {})
        raw = indicator_result.get("raw", {})

        if not scores:
            return None
            
        # Add PCR score from stock_info options data
        options_data = stock_info.get("options", {})
        pcr = options_data.get("pcr", 1.0)
        
        # Calculate PCR score mapping (0-100)
        # > 1.2 is strongly bullish (100)
        # 0.9 - 1.1 is neutral (50)
        # < 0.6 is strongly bearish (0)
        if pcr >= 1.2:
            scores["pcr"] = 90.0
        elif pcr >= 1.0:
            scores["pcr"] = 70.0
        elif pcr <= 0.6:
            scores["pcr"] = 10.0
        elif pcr <= 0.8:
            scores["pcr"] = 30.0
        else:
            scores["pcr"] = 50.0

        # Weighted aggregate score
        total_score = 0.0
        for indicator, weight in self.weights.items():
            ind_score = scores.get(indicator, 50.0)
            total_score += ind_score * weight

        signal = TradeSignal(
            symbol=symbol,
            score=total_score,
            config=self.config,
            indicator_scores=scores,
            raw_indicators=raw,
            stock_info=stock_info,
        )

        logger.info(f"{symbol}: Score={total_score:.1f} → {signal.signal_type}")
        return signal

    def generate_all(self, indicator_results: Dict[str, dict],
                     stock_infos: Dict[str, dict]) -> List[TradeSignal]:
        """
        Generate signals for all stocks.
        Returns list sorted by score (highest first).
        """
        signals = []
        for symbol, result in indicator_results.items():
            info = stock_infos.get(symbol, {})
            signal = self.generate_signal(symbol, result, info)
            if signal:
                signals.append(signal)

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
