# 🚀 Options HFT Bot

A modular, Python-based High-Frequency Trading bot for options markets featuring volatility arbitrage, surface anomaly detection, and skew mean-reversion strategies.

**Supported Brokers:** Dhan (NSE/BSE India) • Interactive Brokers • Paper Trading

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Main Application                      │
├──────┬──────────┬───────────┬───────────┬───────────────┤
│ Data │ Pricing  │ Strategy  │   Risk    │  Execution    │
│ Feed │ Engine   │ & Signals │ Manager   │  Engine       │
├──────┴──────────┴───────────┴───────────┴───────────────┤
│                     Backtesting                          │
└─────────────────────────────────────────────────────────┘
```

## Features

- **3 Trading Strategies**: IV vs Realized Vol, Vol Surface Anomalies, Skew Mean-Reversion
- **Full Greeks Engine**: Black-Scholes with Delta, Gamma, Theta, Vega, Rho
- **Implied Volatility Solver**: Newton-Raphson + Brent's method
- **Volatility Surface**: Real-time construction and anomaly detection
- **5-Stage Signal Filtering**: Strength, liquidity, time-of-day, correlation, spread
- **Risk Management**: 8 pre-trade checks + kill switch with auto-flatten
- **Smart Order Routing**: Retry logic with progressive price improvement
- **Backtesting Engine**: Event-driven with slippage/commission simulation
- **Multi-Broker**: Dhan (India), Interactive Brokers, Paper Trading

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Broker

#### Dhan (India — NSE/BSE F&O)
1. Login to [web.dhan.co](https://web.dhan.co)
2. Go to **My Profile → DhanHQ Trading APIs & Access**
3. Generate your **Access Token**
4. Edit `config/settings.yaml`:
```yaml
broker:
  name: "dhan"
  dhan:
    client_id: "YOUR_DHAN_CLIENT_ID"
    access_token: "YOUR_DHAN_ACCESS_TOKEN"
    exchange_segment: "NSE_FNO"
    product_type: "INTRA"
```

#### Interactive Brokers
1. Install and run IB TWS or IB Gateway
2. Edit `config/settings.yaml`:
```yaml
broker:
  name: "interactive_brokers"
  ib:
    host: "127.0.0.1"
    port: 7497            # 7497=paper, 7496=live
    client_id: 1
```

### 3. Run Backtest (with sample data)
```bash
python src/main.py backtest
```

### 4. Run with Dhan (live)
```bash
python src/main.py live --broker dhan
```

### 5. Run Paper Trading
```bash
python src/main.py paper
```

### CLI Options
```bash
python src/main.py <mode> [--broker dhan|ib|paper] [--config <dir>] [--data <file>]
```

## Project Structure

```
options-hft-bot/
├── config/
│   ├── settings.yaml          # Strategy, broker (Dhan/IB) & feed config
│   └── risk_limits.yaml       # Risk thresholds
├── src/
│   ├── pricing/               # Black-Scholes, IV solver, vol surface
│   ├── data/
│   │   ├── chain.py           # Options chain model
│   │   ├── feed.py            # Generic WebSocket feed  
│   │   ├── dhan_feed.py       # Dhan live market data feed
│   │   └── storage.py         # SQLite/Parquet tick storage
│   ├── strategy/              # Signal generator, filters
│   ├── risk/                  # Portfolio, risk manager, kill switch
│   ├── execution/
│   │   ├── broker.py          # BrokerBase, IBBroker, PaperBroker
│   │   ├── dhan_broker.py     # Dhan broker (dhanhq SDK)
│   │   ├── oms.py             # Order management system
│   │   └── router.py          # Smart order router
│   ├── backtest/              # Engine, metrics, data loader
│   └── main.py                # Entry point
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

## Running Tests
```bash
python -m pytest tests/ -v
```

## ⚠️ Disclaimer

This software is for educational purposes. Options trading involves significant risk of loss. Past backtesting performance does not guarantee future results. Always paper trade extensively before using real capital.
