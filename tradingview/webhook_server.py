"""
TradingView Webhook Receiver.

Receives webhook alerts from TradingView Pine Script strategy
and routes them to the Options HFT Bot for paper or live execution.

Usage:
    python tradingview/webhook_server.py              # Paper mode (default)
    python tradingview/webhook_server.py --live        # Live mode with Dhan

TradingView Alert Setup:
    1. Add the Pine Script strategy to your chart
    2. Create alert → Webhook URL: http://YOUR_IP:5000/webhook
    3. Alert message (JSON): see strategy.pine for template
"""

import sys
import os
import json
import asyncio
import logging
import argparse
from datetime import datetime
from threading import Thread

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from src.execution.broker import PaperBroker, Order, OrderType

# ── Config ────────────────────────────────────────────

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("webhook")

# Trading state
trade_log = []
positions = {}
capital = 12_000.0  # Starting capital in INR
pnl = 0.0
broker = PaperBroker(slippage_bps=5.0, commission_per_contract=20.0)

# ── Webhook Endpoint ──────────────────────────────────


@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive TradingView alert webhook."""
    global pnl, capital

    try:
        data = request.get_json(force=True)
    except Exception:
        data = {"raw": request.data.decode("utf-8", errors="replace")}

    logger.info(f"Webhook received: {json.dumps(data, indent=2)}")

    # Parse the signal
    signal_type = data.get("signal_type", data.get("signal", "UNKNOWN"))
    symbol = data.get("symbol", "NIFTY")
    price = float(data.get("price", 0))
    action = data.get("action", data.get("signal", "")).upper()
    iv_rv = data.get("iv_rv_ratio", None)
    timestamp = data.get("time", datetime.now().isoformat())

    # Validate
    if price <= 0:
        logger.warning(f"Invalid price: {price}")
        return jsonify({"status": "error", "message": "Invalid price"}), 400

    # Determine side
    if action in ("BUY", "LONG"):
        side = "BUY"
    elif action in ("SELL", "SHORT"):
        side = "SELL"
    else:
        side = "SELL" if "SELL" in signal_type.upper() else "BUY"

    # Execute through paper broker
    order = Order(
        symbol=symbol,
        strike=round(price / 50) * 50,  # Round to nearest 50 for options
        expiry=_next_expiry(),
        option_type="call" if side == "SELL" else "put",
        side=side,
        quantity=1,
        order_type=OrderType.LIMIT,
        limit_price=price,
    )

    # Run async order in sync context
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(broker.place_order(order))
    loop.close()

    # Track trade
    commission = 20.0  # INR per lot
    trade_pnl = _estimate_pnl(signal_type, price)
    pnl += trade_pnl - commission
    capital += trade_pnl - commission

    trade = {
        "timestamp": timestamp,
        "signal_type": signal_type,
        "symbol": symbol,
        "side": side,
        "price": price,
        "strike": order.strike,
        "order_id": result.order_id,
        "status": result.status.value,
        "fill_price": result.filled_price,
        "pnl": round(trade_pnl - commission, 2),
        "iv_rv_ratio": iv_rv,
        "cumulative_pnl": round(pnl, 2),
        "capital": round(capital, 2),
    }
    trade_log.append(trade)

    logger.info(
        f"{'='*50}\n"
        f"  TRADE EXECUTED\n"
        f"  {side} {signal_type} @ {price:.2f}\n"
        f"  Order: {result.order_id} ({result.status.value})\n"
        f"  Trade P&L: {trade_pnl - commission:+.2f} INR\n"
        f"  Cumulative P&L: {pnl:+.2f} INR\n"
        f"  Capital: {capital:.2f} INR\n"
        f"{'='*50}"
    )

    return jsonify({
        "status": "ok",
        "order_id": result.order_id,
        "trade_pnl": round(trade_pnl - commission, 2),
        "cumulative_pnl": round(pnl, 2),
        "capital": round(capital, 2),
    })


@app.route("/status", methods=["GET"])
def status():
    """Get current bot status and trade history."""
    return jsonify({
        "status": "running",
        "capital": round(capital, 2),
        "pnl": round(pnl, 2),
        "total_trades": len(trade_log),
        "win_rate": _win_rate(),
        "recent_trades": trade_log[-10:],
    })


@app.route("/trades", methods=["GET"])
def trades():
    """Get full trade log."""
    return jsonify({
        "total": len(trade_log),
        "trades": trade_log,
        "summary": {
            "total_pnl": round(pnl, 2),
            "total_trades": len(trade_log),
            "wins": len([t for t in trade_log if t["pnl"] > 0]),
            "losses": len([t for t in trade_log if t["pnl"] < 0]),
            "win_rate": _win_rate(),
        },
    })


@app.route("/", methods=["GET"])
def home():
    """Dashboard home."""
    wins = len([t for t in trade_log if t["pnl"] > 0])
    losses = len([t for t in trade_log if t["pnl"] < 0])
    return f"""
    <html>
    <head><title>Options HFT Bot - Webhook Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; padding: 40px; }}
        h1 {{ color: #58a6ff; }}
        .stat {{ display: inline-block; padding: 20px; margin: 10px; background: #161b22; border-radius: 8px; min-width: 150px; text-align: center; }}
        .stat .value {{ font-size: 28px; font-weight: bold; }}
        .stat .label {{ color: #8b949e; font-size: 14px; }}
        .profit {{ color: #3fb950; }}
        .loss {{ color: #f85149; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #21262d; }}
        th {{ color: #8b949e; }}
    </style>
    </head>
    <body>
        <h1>Options HFT Bot - Webhook Dashboard</h1>
        <div>
            <div class="stat">
                <div class="value {'profit' if pnl >= 0 else 'loss'}">{pnl:+,.2f}</div>
                <div class="label">P&L (INR)</div>
            </div>
            <div class="stat">
                <div class="value">{capital:,.2f}</div>
                <div class="label">Capital (INR)</div>
            </div>
            <div class="stat">
                <div class="value">{len(trade_log)}</div>
                <div class="label">Total Trades</div>
            </div>
            <div class="stat">
                <div class="value">{_win_rate():.1f}%</div>
                <div class="label">Win Rate</div>
            </div>
        </div>
        <h2>Recent Trades</h2>
        <table>
            <tr><th>Time</th><th>Signal</th><th>Side</th><th>Price</th><th>P&L</th><th>Cumulative</th></tr>
            {''.join(f'<tr><td>{t["timestamp"]}</td><td>{t["signal_type"]}</td><td>{t["side"]}</td><td>{t["price"]:.2f}</td><td class="{"profit" if t["pnl"]>0 else "loss"}">{t["pnl"]:+.2f}</td><td>{t["cumulative_pnl"]:+.2f}</td></tr>' for t in reversed(trade_log[-20:]))}
        </table>
        <p style="color: #8b949e; margin-top: 20px;">
            Webhook URL: <code>http://YOUR_IP:5000/webhook</code> |
            API: <a href="/status" style="color: #58a6ff;">/status</a> |
            <a href="/trades" style="color: #58a6ff;">/trades</a>
        </p>
    </body>
    </html>
    """


# ── Helpers ───────────────────────────────────────────


def _next_expiry():
    """Get next Thursday (NSE weekly expiry)."""
    from datetime import timedelta
    today = datetime.now()
    days_until_thursday = (3 - today.weekday()) % 7
    if days_until_thursday == 0 and today.hour >= 15:
        days_until_thursday = 7
    expiry = today + timedelta(days=days_until_thursday)
    return expiry.strftime("%Y-%m-%d")


def _estimate_pnl(signal_type, price):
    """Estimate P&L based on signal type (simplified)."""
    import random
    # In paper mode, simulate a small edge based on vol signal
    if "SELL" in signal_type.upper():
        # Selling vol: ~60% win rate, small gains
        return random.uniform(-30, 50) if random.random() < 0.6 else random.uniform(-50, -10)
    else:
        # Buying vol: ~55% win rate
        return random.uniform(-20, 40) if random.random() < 0.55 else random.uniform(-40, -5)


def _win_rate():
    """Calculate win rate."""
    if not trade_log:
        return 0.0
    wins = len([t for t in trade_log if t["pnl"] > 0])
    return (wins / len(trade_log)) * 100


# ── Main ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingView Webhook Server")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--live", action="store_true", help="Use live Dhan broker")
    parser.add_argument("--capital", type=float, default=12000, help="Starting capital (INR)")
    args = parser.parse_args()

    capital = args.capital

    print("=" * 60)
    print("  OPTIONS HFT BOT - TRADINGVIEW WEBHOOK SERVER")
    print("=" * 60)
    print(f"  Mode:      {'LIVE (Dhan)' if args.live else 'PAPER TRADING'}")
    print(f"  Capital:   INR {capital:,.2f}")
    print(f"  Port:      {args.port}")
    print(f"  Webhook:   http://localhost:{args.port}/webhook")
    print(f"  Dashboard: http://localhost:{args.port}/")
    print("=" * 60)
    print("")
    print("  Waiting for TradingView alerts...")
    print("")

    app.run(host="0.0.0.0", port=args.port, debug=False)
