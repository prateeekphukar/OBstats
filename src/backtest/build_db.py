"""
Backtesting Database Builder (Vectorized).

Generates a comprehensive SQLite database of synthetic options data
for NIFTY, NIFTY50, BANKNIFTY indices and 10 major F&O stocks,
covering the last 3 months with 5-minute bars.

Usage:
    python -m src.backtest.build_db [--output data/backtest.db]
"""

import sqlite3
import os
import sys
import math
import argparse
import logging
from datetime import datetime, timedelta, date
from typing import NamedTuple
from pathlib import Path

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Instrument Definitions
# ──────────────────────────────────────────────

class Instrument(NamedTuple):
    symbol: str
    start_price: float
    annual_vol: float
    beta: float
    strike_step: float
    strike_range_pct: float
    is_index: bool
    lot_size: int


INSTRUMENTS = [
    # Indices
    Instrument("NIFTY",      22500, 0.16, 1.00,  50,  0.05, True,   25),
    Instrument("NIFTY50",    22500, 0.16, 1.00,  50,  0.05, True,   25),
    Instrument("BANKNIFTY",  48000, 0.21, 0.85, 100,  0.05, True,   15),
    # F&O Stocks
    Instrument("RELIANCE",    2450, 0.22, 1.10,  20,  0.08, False, 250),
    Instrument("TCS",         3800, 0.18, 0.70,  25,  0.08, False, 150),
    Instrument("INFY",        1850, 0.20, 0.75,  25,  0.08, False, 300),
    Instrument("HDFCBANK",    1700, 0.19, 1.05,  10,  0.08, False, 550),
    Instrument("ICICIBANK",   1250, 0.21, 1.15,  10,  0.08, False, 700),
    Instrument("SBIN",         780, 0.25, 1.25,  10,  0.08, False, 750),
    Instrument("BHARTIARTL",  1650, 0.23, 0.65,  10,  0.08, False, 300),
    Instrument("ITC",          465, 0.16, 0.50,  5,   0.08, False, 1600),
    Instrument("KOTAKBANK",   1800, 0.20, 0.95,  25,  0.08, False, 400),
    Instrument("LT",          3500, 0.19, 0.90,  25,  0.08, False, 150),
]

# ──────────────────────────────────────────────
# Trading Calendar
# ──────────────────────────────────────────────

NSE_HOLIDAYS = {
    date(2025, 11, 5), date(2025, 11, 14),
    date(2025, 12, 25), date(2026, 1, 26),
}


def generate_trading_days(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in NSE_HOLIDAYS:
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_timestamps(trading_day: date, bars_per_day: int = 75) -> np.ndarray:
    market_open = datetime(trading_day.year, trading_day.month, trading_day.day, 9, 15)
    interval = 375.0 / bars_per_day
    return np.array([
        (market_open + timedelta(minutes=i * interval)).timestamp()
        for i in range(bars_per_day)
    ])


# ──────────────────────────────────────────────
# Expiry Calendar
# ──────────────────────────────────────────────

def get_thursdays(start: date, end: date) -> list[date]:
    current = start
    while current.weekday() != 3:
        current += timedelta(days=1)
    thursdays = []
    while current <= end:
        thursdays.append(current)
        current += timedelta(days=7)
    return thursdays


def get_monthly_expiries(start: date, end: date) -> list[date]:
    monthly = []
    current_month = start.replace(day=1)
    while current_month <= end:
        if current_month.month == 12:
            next_month = current_month.replace(year=current_month.year + 1, month=1)
        else:
            next_month = current_month.replace(month=current_month.month + 1)
        last_day = next_month - timedelta(days=1)
        while last_day.weekday() != 3:
            last_day -= timedelta(days=1)
        if start <= last_day <= end:
            monthly.append(last_day)
        current_month = next_month
    return monthly


def get_active_expiries(current_date: date, all_expiries: list[date], max_n: int = 4) -> list[str]:
    future = [e for e in all_expiries if e >= current_date]
    return [e.isoformat() for e in future[:max_n]]


# ──────────────────────────────────────────────
# Vectorized Black-Scholes
# ──────────────────────────────────────────────

def bs_vectorized(S, K, T, r, sigma, is_call_mask):
    """
    Fully vectorized Black-Scholes pricing + Greeks.
    All inputs are numpy arrays of the same shape.
    is_call_mask: boolean array (True=call, False=put)
    Returns dict of arrays: price, delta, gamma, theta, vega, iv
    """
    T = np.maximum(T, 0.001)
    sigma = np.maximum(sigma, 0.01)
    S = np.maximum(S, 0.01)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)

    # Prices
    call_price = S * nd1 - K * np.exp(-r * T) * nd2
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    price = np.where(is_call_mask, call_price, put_price)
    price = np.maximum(price, 0.05)

    # Delta
    call_delta = nd1
    put_delta = nd1 - 1.0
    delta = np.where(is_call_mask, call_delta, put_delta)

    # Gamma
    gamma = npd1 / (S * sigma * sqrt_T)

    # Theta (per day)
    theta = -(S * npd1 * sigma) / (2 * sqrt_T) / 365.0

    # Vega (per 1% vol move)
    vega = S * npd1 * sqrt_T / 100.0

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "iv": sigma,
    }


# ──────────────────────────────────────────────
# Strike & Vol Helpers
# ──────────────────────────────────────────────

def generate_strikes(spot: float, step: float, range_pct: float) -> np.ndarray:
    low = math.floor(spot * (1 - range_pct) / step) * step
    high = math.ceil(spot * (1 + range_pct) / step) * step
    return np.arange(low, high + step, step)


def get_implied_vol_vec(base_vol, spot, strikes, T):
    """Vectorized vol smile: skew + smile + term structure."""
    moneyness = np.log(strikes / spot)
    skew = -0.12 * moneyness
    smile = 0.8 * moneyness**2
    term_adj = 1.0 / max(0.1, math.sqrt(T))
    vol = base_vol + (skew + smile) * term_adj * 0.5
    return np.clip(vol, 0.05, 1.5)


# ──────────────────────────────────────────────
# GBM Spot Simulation
# ──────────────────────────────────────────────

def simulate_spots(instruments, num_steps, dt, rng):
    n = len(instruments)
    base_corr = 0.5
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho = min(0.90, max(0.05, instruments[i].beta * instruments[j].beta * base_corr))
            corr[i, j] = rho
            corr[j, i] = rho
    # Ensure PD
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)

    Z = rng.standard_normal((num_steps, n))
    Z_corr = Z @ L.T

    spots = {}
    for idx, inst in enumerate(instruments):
        prices = np.empty(num_steps)
        prices[0] = inst.start_price
        vol_base = inst.annual_vol
        vol_mult = 1.0
        for t in range(1, num_steps):
            if t % 1125 == 0:
                vol_mult = rng.uniform(0.7, 1.5)
            vol = vol_base * vol_mult
            prices[t] = prices[t - 1] * math.exp(
                -0.5 * vol**2 * dt + vol * math.sqrt(dt) * Z_corr[t, idx]
            )
        spots[inst.symbol] = prices

    if "NIFTY" in spots and "NIFTY50" in spots:
        spots["NIFTY50"] = spots["NIFTY"].copy()
    return spots


# ──────────────────────────────────────────────
# Database Schema
# ──────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS option_ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strike REAL NOT NULL,
    expiry TEXT NOT NULL,
    option_type TEXT NOT NULL,
    bid REAL, ask REAL, last REAL,
    volume INTEGER, open_interest INTEGER,
    iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
    underlying_price REAL,
    timestamp REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON option_ticks (symbol, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_contract ON option_ticks (symbol, expiry, strike, option_type);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_expiry ON option_ticks (expiry);",
    "CREATE INDEX IF NOT EXISTS idx_ticks_ts ON option_ticks (timestamp);",
]

INSERT_SQL = """
INSERT INTO option_ticks
    (symbol, strike, expiry, option_type, bid, ask, last,
     volume, open_interest, iv, delta, gamma, theta, vega,
     underlying_price, timestamp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

CREATE_UNDERLYING_SQL = """
CREATE TABLE IF NOT EXISTS underlying_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    timestamp REAL NOT NULL
);
"""

CREATE_UNDERLYING_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_underlying_symbol_ts ON underlying_prices (symbol, timestamp);"
)

CREATE_META_SQL = """
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


# ──────────────────────────────────────────────
# Main Builder — Vectorized per-day per-instrument
# ──────────────────────────────────────────────

def build_database(
    output_path: str = "data/backtest.db",
    bars_per_day: int = 75,
    risk_free_rate: float = 0.065,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    start_date = date(2025, 11, 22)
    end_date = date(2026, 2, 22)
    trading_days = generate_trading_days(start_date, end_date)
    num_days = len(trading_days)
    total_bars = num_days * bars_per_day
    dt = 5.0 / (252 * 375)

    print(f"{'=' * 65}")
    print(f"  OPTIONS BACKTESTING DATABASE BUILDER")
    print(f"{'=' * 65}")
    print(f"  Date range:      {start_date} -> {end_date}")
    print(f"  Trading days:    {num_days}")
    print(f"  Bars/day:        {bars_per_day} (5-min intervals)")
    print(f"  Total bars:      {total_bars:,}")
    print(f"  Instruments:     {len(INSTRUMENTS)}")
    print(f"  Output:          {output_path}")
    print(f"{'=' * 65}\n")

    # ── Simulate spots ──
    print("[1/5] Simulating spot prices (GBM)...")
    spots = simulate_spots(INSTRUMENTS, total_bars, dt, rng)
    print(f"      -> {len(spots)} instruments, {total_bars:,} bars each\n")

    # ── Expiry calendars ──
    print("[2/5] Building expiry calendars...")
    expiry_end = end_date + timedelta(days=90)
    weekly_expiries = get_thursdays(start_date, expiry_end)
    monthly_expiries = get_monthly_expiries(start_date, expiry_end)
    print(f"      -> {len(weekly_expiries)} weekly, {len(monthly_expiries)} monthly\n")

    # ── Init DB ──
    print("[3/5] Initializing SQLite database...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_UNDERLYING_SQL)
    conn.execute(CREATE_META_SQL)
    conn.commit()
    print(f"      -> Created at {output_path}\n")

    # ── Generate data (vectorized per day per instrument) ──
    print("[4/5] Generating options data...")

    grand_total = 0

    for inst_idx, inst in enumerate(INSTRUMENTS):
        expiries = weekly_expiries if inst.is_index else monthly_expiries
        max_exp = 4 if inst.is_index else 2
        bar_offset = 0
        inst_rows = 0
        batch = []
        underlying_batch = []

        print(f"      [{inst_idx + 1}/{len(INSTRUMENTS)}] {inst.symbol:<14} ", end="", flush=True)

        for day_idx, trading_day in enumerate(trading_days):
            ts_arr = generate_timestamps(trading_day, bars_per_day)
            active_expiries = get_active_expiries(trading_day, expiries, max_exp)
            if not active_expiries:
                bar_offset += bars_per_day
                continue

            spot_day = spots[inst.symbol][bar_offset:bar_offset + bars_per_day]
            bar_offset += bars_per_day

            # Store underlying prices for the day
            for bi in range(bars_per_day):
                underlying_batch.append((inst.symbol, round(float(spot_day[bi]), 2), float(ts_arr[bi])))

            # Sample every N bars to reduce volume (every 3rd bar = ~15min)
            sample_indices = np.arange(0, bars_per_day, 3)

            for exp_str in active_expiries:
                exp_date = date.fromisoformat(exp_str)
                T = max(0.001, (exp_date - trading_day).days / 365.0)

                for bi in sample_indices:
                    spot = float(spot_day[bi])
                    ts = float(ts_arr[bi])

                    strikes = generate_strikes(spot, inst.strike_step, inst.strike_range_pct)
                    n_strikes = len(strikes)

                    # Vectorized: compute for ALL strikes × 2 option types at once
                    S_arr = np.full(n_strikes * 2, spot)
                    K_arr = np.tile(strikes, 2)
                    T_arr = np.full(n_strikes * 2, T)
                    is_call = np.array([True] * n_strikes + [False] * n_strikes)

                    # Vol smile
                    vol_calls = get_implied_vol_vec(inst.annual_vol, spot, strikes, T)
                    vol_puts = get_implied_vol_vec(inst.annual_vol, spot, strikes, T)
                    sigma_arr = np.concatenate([vol_calls, vol_puts])

                    # Price everything in one shot
                    res = bs_vectorized(S_arr, K_arr, T_arr, risk_free_rate, sigma_arr, is_call)

                    prices = res["price"]
                    noise = rng.normal(0, 0.005, size=len(prices)) * prices
                    prices = np.maximum(0.05, prices + noise)

                    # Spreads
                    moneyness_frac = np.abs(S_arr - K_arr) / S_arr
                    spread_pct = 0.005 + 0.02 * moneyness_frac
                    if not inst.is_index:
                        spread_pct *= 1.5
                    half_spread = prices * spread_pct / 2

                    bids = np.round(np.maximum(0.05, prices - half_spread), 2)
                    asks = np.round(np.maximum(bids + 0.05, prices + half_spread), 2)
                    lasts = np.round(prices + rng.uniform(-0.01, 0.01, size=len(prices)) * prices, 2)

                    # Volume & OI
                    atm_factor = np.maximum(0.05, 1.0 - 3.0 * moneyness_frac)
                    base_vol_qty = 5000 if inst.is_index else 500
                    base_oi = 20000 if inst.is_index else 2000
                    volumes = rng.exponential(base_vol_qty * atm_factor).astype(int)
                    ois = rng.exponential(base_oi * atm_factor).astype(int)

                    opt_types = ["call"] * n_strikes + ["put"] * n_strikes

                    for k in range(len(prices)):
                        batch.append((
                            inst.symbol, float(K_arr[k]), exp_str, opt_types[k],
                            float(bids[k]), float(asks[k]), float(lasts[k]),
                            int(volumes[k]), int(ois[k]),
                            float(res["iv"][k]), float(res["delta"][k]),
                            float(res["gamma"][k]), float(res["theta"][k]),
                            float(res["vega"][k]),
                            round(spot, 2), ts,
                        ))
                        inst_rows += 1

                    # Flush in chunks
                    if len(batch) >= 50000:
                        conn.executemany(INSERT_SQL, batch)
                        if underlying_batch:
                            conn.executemany(
                                "INSERT INTO underlying_prices (symbol, price, timestamp) VALUES (?, ?, ?)",
                                underlying_batch,
                            )
                            underlying_batch.clear()
                        conn.commit()
                        batch.clear()

            # Progress dot every 10 days
            if (day_idx + 1) % 10 == 0:
                print(".", end="", flush=True)

        # Flush remainder for this instrument
        if batch:
            conn.executemany(INSERT_SQL, batch)
            batch.clear()
        if underlying_batch:
            conn.executemany(
                "INSERT INTO underlying_prices (symbol, price, timestamp) VALUES (?, ?, ?)",
                underlying_batch,
            )
            underlying_batch.clear()
        conn.commit()

        grand_total += inst_rows
        print(f" -> {inst_rows:>10,} rows")

    # ── Create indices ──
    print(f"\n[5/5] Creating indices and metadata...")
    for idx_sql in CREATE_INDICES_SQL:
        conn.execute(idx_sql)
    conn.execute(CREATE_UNDERLYING_IDX_SQL)

    meta = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "trading_days": str(num_days),
        "bars_per_day": str(bars_per_day),
        "instruments": ",".join(i.symbol for i in INSTRUMENTS),
        "total_rows": str(grand_total),
        "created_at": datetime.now().isoformat(),
        "seed": str(seed),
    }
    for k, v in meta.items():
        conn.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)", (k, v))
    conn.commit()

    # ── Summary ──
    cursor = conn.execute("SELECT COUNT(*) FROM option_ticks")
    actual_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(*) FROM underlying_prices")
    underlying_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM option_ticks")
    symbol_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(DISTINCT expiry) FROM option_ticks")
    expiry_count = cursor.fetchone()[0]
    cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM option_ticks")
    ts_min, ts_max = cursor.fetchone()
    db_size = os.path.getsize(output_path) / (1024 * 1024)
    conn.close()

    print(f"\n{'=' * 65}")
    print(f"  DATABASE BUILD COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Options rows:       {actual_count:>14,}")
    print(f"  Underlying rows:    {underlying_count:>14,}")
    print(f"  Symbols:            {symbol_count:>14}")
    print(f"  Expiries:           {expiry_count:>14}")
    ts_min_str = datetime.fromtimestamp(ts_min).strftime('%Y-%m-%d %H:%M') if ts_min else 'N/A'
    ts_max_str = datetime.fromtimestamp(ts_max).strftime('%Y-%m-%d %H:%M') if ts_max else 'N/A'
    print(f"  Date range:         {ts_min_str} -> {ts_max_str}")
    print(f"  Database size:      {db_size:>14.1f} MB")
    print(f"  Path:               {os.path.abspath(output_path)}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="Build backtesting options database")
    parser.add_argument("--output", "-o", default="data/backtest.db", help="Output DB path")
    parser.add_argument("--bars-per-day", type=int, default=75, help="Bars per day (75=5min)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build_database(output_path=args.output, bars_per_day=args.bars_per_day, seed=args.seed)


if __name__ == "__main__":
    main()
