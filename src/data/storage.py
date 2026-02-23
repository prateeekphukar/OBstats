"""
Data Storage Layer.

Provides tick data archival and retrieval using SQLite (dev) or
Parquet files (production). Swap to TimescaleDB for high-throughput production.
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path

import pandas as pd

from .chain import OptionTick, OptionType

logger = logging.getLogger(__name__)


class TickStorage:
    """
    Tick data storage engine.

    Supports SQLite for development and Parquet for efficient bulk storage.
    """

    def __init__(self, engine: str = "sqlite", sqlite_path: str = "data/ticks.db", parquet_dir: str = "data/parquet"):
        self.engine = engine
        self.sqlite_path = sqlite_path
        self.parquet_dir = parquet_dir
        self._conn: Optional[sqlite3.Connection] = None
        self._buffer: list[dict] = []
        self._buffer_size = 1000  # Flush after N ticks

        if engine == "sqlite":
            self._init_sqlite()
        elif engine == "parquet":
            Path(parquet_dir).mkdir(parents=True, exist_ok=True)

    def _init_sqlite(self) -> None:
        """Initialize SQLite database with schema."""
        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS option_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                expiry TEXT NOT NULL,
                option_type TEXT NOT NULL,
                bid REAL,
                ask REAL,
                last REAL,
                volume INTEGER,
                open_interest INTEGER,
                iv REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                timestamp REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts
            ON option_ticks (symbol, timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticks_contract
            ON option_ticks (symbol, expiry, strike, option_type)
        """)
        self._conn.commit()
        logger.info(f"SQLite storage initialized at {self.sqlite_path}")

    def store_tick(self, tick: OptionTick) -> None:
        """Store a single tick. Uses buffered writes for performance."""
        self._buffer.append(self._tick_to_dict(tick))
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def store_ticks(self, ticks: list[OptionTick]) -> None:
        """Store multiple ticks."""
        for tick in ticks:
            self.store_tick(tick)

    def flush(self) -> None:
        """Write buffered ticks to storage."""
        if not self._buffer:
            return

        if self.engine == "sqlite":
            self._flush_sqlite()
        elif self.engine == "parquet":
            self._flush_parquet()

        count = len(self._buffer)
        self._buffer.clear()
        logger.debug(f"Flushed {count} ticks to {self.engine}")

    def _flush_sqlite(self) -> None:
        """Flush buffer to SQLite."""
        if not self._conn:
            return
        self._conn.executemany(
            """INSERT INTO option_ticks
               (symbol, strike, expiry, option_type, bid, ask, last,
                volume, open_interest, iv, delta, gamma, theta, vega, timestamp)
               VALUES (:symbol, :strike, :expiry, :option_type, :bid, :ask, :last,
                :volume, :open_interest, :iv, :delta, :gamma, :theta, :vega, :timestamp)
            """,
            self._buffer,
        )
        self._conn.commit()

    def _flush_parquet(self) -> None:
        """Flush buffer to a Parquet file (one file per flush)."""
        df = pd.DataFrame(self._buffer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.parquet_dir, f"ticks_{timestamp}.parquet")
        df.to_parquet(path, index=False)

    def query(
        self,
        symbol: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        strike: Optional[float] = None,
        expiry: Optional[str] = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """
        Query stored ticks.

        Returns a pandas DataFrame of matching ticks.
        """
        if self.engine == "sqlite":
            return self._query_sqlite(symbol, start_time, end_time, strike, expiry, limit)
        elif self.engine == "parquet":
            return self._query_parquet(symbol, start_time, end_time, strike, expiry, limit)
        return pd.DataFrame()

    def _query_sqlite(
        self, symbol: str, start_time, end_time, strike, expiry, limit
    ) -> pd.DataFrame:
        """Query SQLite database."""
        query = "SELECT * FROM option_ticks WHERE symbol = ?"
        params: list = [symbol]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if strike:
            query += " AND strike = ?"
            params.append(strike)
        if expiry:
            query += " AND expiry = ?"
            params.append(expiry)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        return pd.read_sql_query(query, self._conn, params=params)

    def _query_parquet(
        self, symbol: str, start_time, end_time, strike, expiry, limit
    ) -> pd.DataFrame:
        """Query Parquet files."""
        parquet_files = sorted(Path(self.parquet_dir).glob("*.parquet"))
        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files[-50:]]  # Last 50 files
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["symbol"] == symbol]

        if start_time:
            df = df[df["timestamp"] >= start_time]
        if end_time:
            df = df[df["timestamp"] <= end_time]
        if strike:
            df = df[df["strike"] == strike]
        if expiry:
            df = df[df["expiry"] == expiry]

        return df.sort_values("timestamp", ascending=False).head(limit)

    @staticmethod
    def _tick_to_dict(tick: OptionTick) -> dict:
        """Convert OptionTick to dict for storage."""
        return {
            "symbol": tick.symbol,
            "strike": tick.strike,
            "expiry": tick.expiry,
            "option_type": tick.option_type.value,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "open_interest": tick.open_interest,
            "iv": tick.iv,
            "delta": tick.delta,
            "gamma": tick.gamma,
            "theta": tick.theta,
            "vega": tick.vega,
            "timestamp": tick.timestamp,
        }

    def close(self) -> None:
        """Close the storage connection."""
        self.flush()
        if self._conn:
            self._conn.close()
        logger.info("Storage closed")
