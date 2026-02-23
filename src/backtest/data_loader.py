"""
Historical Data Loader.

Loads historical options data from CSV or Parquet files
for backtesting.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """
    Loads and preprocesses historical options data for backtesting.

    Supports CSV and Parquet formats. Normalizes column names
    and validates required fields.
    """

    REQUIRED_COLUMNS = [
        "timestamp", "symbol", "strike", "expiry",
        "option_type", "bid", "ask",
    ]

    OPTIONAL_COLUMNS = [
        "last", "volume", "open_interest",
        "underlying_price", "underlying_bid", "underlying_ask",
    ]

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load from a CSV file."""
        logger.info(f"Loading CSV: {filepath}")
        df = pd.read_csv(filepath, **kwargs)
        return self._normalize(df)

    def load_parquet(self, filepath: str) -> pd.DataFrame:
        """Load from a Parquet file."""
        logger.info(f"Loading Parquet: {filepath}")
        df = pd.read_parquet(filepath)
        return self._normalize(df)

    def load_directory(self, directory: Optional[str] = None, pattern: str = "*.parquet") -> pd.DataFrame:
        """Load all matching files from a directory."""
        dir_path = Path(directory) if directory else self.data_dir
        files = sorted(dir_path.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {dir_path}")

        logger.info(f"Loading {len(files)} files from {dir_path}")
        dfs = []
        for f in files:
            if f.suffix == ".parquet":
                dfs.append(pd.read_parquet(f))
            elif f.suffix == ".csv":
                dfs.append(pd.read_csv(f))

        df = pd.concat(dfs, ignore_index=True)
        return self._normalize(df)

    def generate_sample_data(
        self,
        symbol: str = "SPY",
        spot: float = 450.0,
        num_bars: int = 5000,
        num_strikes: int = 10,
        num_expiries: int = 2,
    ) -> pd.DataFrame:
        """
        Generate synthetic options data for testing.

        Creates options chain data with vol regime shifts that produce
        tradeable IV vs realized vol divergences.
        """
        import numpy as np
        from datetime import datetime, timedelta

        logger.info(f"Generating sample data: {num_bars} bars, {num_strikes} strikes, {num_expiries} expiries")

        rows = []
        base_time = datetime(2024, 1, 2, 9, 30)
        price = spot
        np.random.seed(42)

        # Generate strikes centered around spot
        strike_step = 5.0
        base_strikes = [
            spot + (i - num_strikes // 2) * strike_step
            for i in range(num_strikes)
        ]

        # Generate expiries (30 and 60 days from start)
        expiries = [
            (base_time + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d")
            for i in range(num_expiries)
        ]

        # Vol regime: alternates between high-vol and low-vol periods
        # This creates the IV/RV divergences the strategy hunts for
        base_vol = 0.20  # 20% annualized base vol
        regime_vol = base_vol
        vol_regime_shift_interval = 200  # Bars between regime shifts

        for bar in range(num_bars):
            # Shift vol regime every N bars
            if bar % vol_regime_shift_interval == 0 and bar > 0:
                if regime_vol > base_vol:
                    regime_vol = base_vol * 0.6  # Drop to low vol
                else:
                    regime_vol = base_vol * 1.8  # Spike to high vol

            # Simulate underlying price movement
            daily_vol = regime_vol / np.sqrt(252 * 390)  # Per-minute vol
            price *= np.exp(np.random.normal(0, daily_vol))
            timestamp = (base_time + timedelta(minutes=bar)).timestamp()

            for expiry_idx, expiry in enumerate(expiries):
                T = max(0.01, (30 * (expiry_idx + 1)) / 365)

                for strike in base_strikes:
                    for opt_type in ["call", "put"]:
                        moneyness = price / strike

                        if opt_type == "call":
                            intrinsic = max(0, price - strike)
                        else:
                            intrinsic = max(0, strike - price)

                        # Price the option with the CURRENT implied vol
                        # (which lags the regime shift — this creates the divergence!)
                        pricing_vol = base_vol
                        # After a regime shift, IV slowly adjusts
                        # First 50 bars after shift: IV hasn't caught up yet
                        bars_since_shift = bar % vol_regime_shift_interval
                        if bars_since_shift < 80:
                            # IV is still at the OLD level while RV has shifted
                            pricing_vol = base_vol  # IV stays at base
                        else:
                            # IV has caught up to realized vol
                            pricing_vol = regime_vol

                        # Add vol smile (OTM options have higher vol)
                        otm_adj = 0.02 * abs(moneyness - 1.0) / 0.05
                        pricing_vol_adj = pricing_vol + otm_adj

                        # BS-approximate option price
                        time_value = price * pricing_vol_adj * np.sqrt(T) * np.exp(
                            -0.5 * ((moneyness - 1) ** 2) / (pricing_vol_adj ** 2 * T)
                        )
                        mid = intrinsic + max(0.01, time_value)

                        # Add random noise to make prices more realistic
                        noise = np.random.uniform(-0.05, 0.05) * mid
                        mid = max(0.01, mid + noise)

                        spread = mid * 0.03  # 3% spread
                        bid = round(max(0.01, mid - spread / 2), 2)
                        ask = round(max(bid + 0.01, mid + spread / 2), 2)

                        rows.append({
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "strike": strike,
                            "expiry": expiry,
                            "option_type": opt_type,
                            "bid": bid,
                            "ask": ask,
                            "last": round(mid, 2),
                            "volume": int(np.random.exponential(200)),
                            "open_interest": int(np.random.exponential(1000)),
                            "underlying_price": round(price, 2),
                        })

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df)} rows of sample data")
        return df

    def load_from_db(
        self,
        db_path: str = "data/backtest.db",
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load options data from the backtesting SQLite database.

        Args:
            db_path: Path to the SQLite database file.
            symbol: Filter by underlying symbol (e.g., "NIFTY", "RELIANCE").
                    If None, loads all symbols.
            start_date: Start date filter as ISO string (e.g., "2025-12-01").
            end_date: End date filter as ISO string (e.g., "2026-02-01").
            limit: Maximum number of rows to return.

        Returns:
            DataFrame in the standard backtest format with columns:
            [timestamp, symbol, strike, expiry, option_type, bid, ask,
             last, volume, open_interest, underlying_price, iv, delta,
             gamma, theta, vega]
        """
        from datetime import datetime

        db_file = Path(db_path)
        if not db_file.exists():
            raise FileNotFoundError(
                f"Backtesting database not found at {db_path}. "
                f"Run 'python -m src.backtest.build_db' to create it."
            )

        conn = sqlite3.connect(str(db_file))
        try:
            query = """
                SELECT symbol, strike, expiry, option_type,
                       bid, ask, last, volume, open_interest,
                       iv, delta, gamma, theta, vega,
                       underlying_price, timestamp
                FROM option_ticks
                WHERE 1=1
            """
            params: list = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                ts_start = datetime.fromisoformat(f"{start_date}T00:00:00").timestamp()
                query += " AND timestamp >= ?"
                params.append(ts_start)
            if end_date:
                ts_end = datetime.fromisoformat(f"{end_date}T23:59:59").timestamp()
                query += " AND timestamp <= ?"
                params.append(ts_end)

            query += " ORDER BY timestamp ASC"

            if limit:
                query += f" LIMIT {int(limit)}"

            logger.info(f"Loading data from {db_path} (symbol={symbol}, "
                        f"start={start_date}, end={end_date})")

            df = pd.read_sql_query(query, conn, params=params)

            logger.info(f"Loaded {len(df):,} rows from database")
            return df

        finally:
            conn.close()

    def get_db_info(self, db_path: str = "data/backtest.db") -> dict:
        """Get metadata about the backtesting database."""
        db_file = Path(db_path)
        if not db_file.exists():
            return {"error": f"Database not found at {db_path}"}

        conn = sqlite3.connect(str(db_file))
        try:
            info = {}
            # Read metadata
            try:
                cursor = conn.execute("SELECT key, value FROM db_metadata")
                for key, value in cursor.fetchall():
                    info[key] = value
            except sqlite3.OperationalError:
                pass

            # Row counts
            cursor = conn.execute("SELECT COUNT(*) FROM option_ticks")
            info["option_rows"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM option_ticks")
            info["symbols"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT DISTINCT symbol FROM option_ticks ORDER BY symbol")
            info["symbol_list"] = [r[0] for r in cursor.fetchall()]

            return info
        finally:
            conn.close()

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and validate schema."""
        # Lowercase column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Check required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Fill optional columns with defaults
        for col in self.OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols, "
                     f"{df['expiry'].nunique()} expirations")
        return df
