"""
Live Trading Signal Dashboard — Flask web app.

Serves a real-time dashboard that displays AI-scored trading signals.
Auto-refreshes data and provides JSON API endpoints.
Now includes user authentication, subscription management, and SEBI compliance.

Usage:
    python -m src.live.dashboard
    Then open http://127.0.0.1:5000/
"""

import sys
import os
import io
import yaml
import logging
import threading
import time as _time
import secrets
from kiteconnect import KiteConnect

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flask import Flask, render_template, jsonify, redirect, url_for, session, request
from flask_login import LoginManager, login_required, current_user
from src.live.data_fetcher import LiveDataFetcher, MarketStatus
from src.live.indicators import TechnicalIndicators
from src.live.signal_engine import AISignalEngine
from src.pricing.black_scholes import BlackScholesPricer
from src.live.models import db, User
from src.live.auth import auth_bp, init_oauth
from src.live.payments import payments_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Kite API Setup ──
KITE_API_KEY = "iog8jjp7lvduite0"
KITE_API_SECRET = "nt98exv18ogjh11z5g7ewg5tqygziixa"
kite_access_token = None
kite = KiteConnect(api_key=KITE_API_KEY)
kite_instruments = []

# ── Load config ──
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'live_signals.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# ── Initialize components ──
data_fetcher = LiveDataFetcher(CONFIG)
indicator_engine = TechnicalIndicators(CONFIG)
signal_engine = AISignalEngine(CONFIG)

# ── Shared state ──
current_signals = []
current_stock_infos = {}
index_data = {}
market_status = MarketStatus.CLOSED
last_error = None

# ── Flask app ──
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'app.db')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.abspath(db_path)}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# OAuth config
auth_cfg = CONFIG.get("auth", {})
app.config['GOOGLE_CLIENT_ID'] = auth_cfg.get("google_client_id", "")
app.config['GOOGLE_CLIENT_SECRET'] = auth_cfg.get("google_client_secret", "")
app.config['FACEBOOK_CLIENT_ID'] = auth_cfg.get("facebook_client_id", "")
app.config['FACEBOOK_CLIENT_SECRET'] = auth_cfg.get("facebook_client_secret", "")

# Razorpay config
razorpay_cfg = CONFIG.get("razorpay", {})
app.config['RAZORPAY_KEY_ID'] = razorpay_cfg.get("key_id", "")
app.config['RAZORPAY_KEY_SECRET'] = razorpay_cfg.get("key_secret", "")

# ── Init extensions ──
db.init_app(app)
init_oauth(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ── Single device enforcement middleware ──
@app.before_request
def check_single_device():
    if current_user.is_authenticated and current_user.active_session_token:
        stored_token = session.get("session_token")
        if stored_token and stored_token != current_user.active_session_token:
            from flask_login import logout_user
            logout_user()
            return redirect(url_for("auth.login"))


# ── Register blueprints ──
app.register_blueprint(auth_bp)
app.register_blueprint(payments_bp)


# ── Create tables ──
with app.app_context():
    db.create_all()


def refresh_data():
    """Fetch new data, compute indicators, generate signals."""
    global current_signals, index_data, market_status, last_error
    try:
        logger.info("=" * 50)
        logger.info("REFRESHING DATA...")

        market_status = data_fetcher.get_market_status()
        logger.info(f"Market Status: {market_status}")

        stock_data = data_fetcher.fetch_all()
        if not stock_data:
            last_error = "No data received from yfinance"
            logger.error(last_error)
            return

        index_data = data_fetcher.fetch_indices()

        indicator_results = {}
        stock_infos = {}
        for symbol, df in stock_data.items():
            indicator_results[symbol] = indicator_engine.compute_all(df)
            stock_infos[symbol] = data_fetcher.get_stock_info(symbol)

        signals = signal_engine.generate_all(indicator_results, stock_infos)
        current_signals = [s.to_dict() for s in signals]
        
        global current_stock_infos
        current_stock_infos = stock_infos
        
        last_error = None

        logger.info(f"Generated {len(current_signals)} signals")
        for s in current_signals[:3]:
            logger.info(f"  {s['symbol']}: {s['signal_type']} (Score: {s['score']})")
        logger.info("=" * 50)

    except Exception as e:
        last_error = str(e)
        logger.error(f"Refresh error: {e}", exc_info=True)


def auto_refresh_loop():
    """Background thread for periodic data refresh."""
    interval = CONFIG.get("data", {}).get("refresh_seconds", 60)
    while True:
        try:
            refresh_data()
        except Exception as e:
            logger.error(f"Auto-refresh error: {e}")
        _time.sleep(interval)


# ── Routes ──

@app.route("/")
@login_required
def dashboard():
    """Main dashboard page."""
    refresh_seconds = CONFIG.get("data", {}).get("refresh_seconds", 60)
    return render_template(
        "index.html",
        refresh_seconds=refresh_seconds,
    )


@app.route("/pricing")
def pricing_page():
    """Pricing / subscription page."""
    return render_template("pricing.html")


@app.route("/blog")
def blog_page():
    """Blog page (peedeeff)."""
    return render_template("blog.html")


@app.route("/api/signals")
@login_required
def api_signals():
    """JSON API returning all current signals."""
    return jsonify({
        "signals": current_signals,
        "indices": index_data,
        "market_status": market_status,
        "last_fetch": data_fetcher.last_fetch_time,
        "error": last_error,
        "count": len(current_signals),
    })


@app.route("/api/market-data")
@login_required
def api_market_data():
    """JSON API returning raw stock info for all watched stocks."""
    return jsonify({
        "stocks": current_stock_infos,
        "indices": index_data,
        "market_status": market_status,
        "last_fetch": data_fetcher.last_fetch_time,
        "error": last_error,
    })


@app.route("/oi-data")
@login_required
def oi_data_page():
    return render_template("oi_data.html")


@app.route("/delivery-data")
@login_required
def delivery_data_page():
    return render_template("delivery_data.html")


@app.route("/oi-change")
@login_required
def oi_change_page():
    return render_template("oi_change.html")


@app.route("/api/refresh", methods=["POST"])
@login_required
def api_refresh():
    """Force a manual data refresh."""
    refresh_data()
    return jsonify({"status": "ok", "count": len(current_signals)})

@app.route("/kite/login")
@login_required
def kite_login():
    return redirect(kite.login_url())

@app.route("/kite/callback")
@login_required
def kite_callback():
    global kite_access_token, kite, kite_instruments
    request_token = request.args.get("request_token")
    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
            kite.set_access_token(data["access_token"])
            kite_access_token = data["access_token"]
            
            logger.info("Downloading Kite NFO instruments...")
            instruments_list = kite.instruments(exchange=kite.EXCHANGE_NFO)
            kite_instruments = [i for i in instruments_list if i["segment"] == "NFO-OPT"]
            logger.info(f"Loaded {len(kite_instruments)} NFO-OPT instruments.")

            return redirect(url_for("dashboard"))
        except Exception as e:
            logger.error(f"Kite login failed: {e}")
            return f"Failed to log in to Kite: {e}"
    return "Error: no Request Token provided."



@app.route("/fii-dii")
@login_required
def fii_dii_page():
    return render_template("fii_dii.html")


@app.route("/api/fii-dii")
@login_required
def api_fii_dii():
    return jsonify({
        "data": data_fetcher.get_fii_dii_data(),
        "last_fetch": data_fetcher.last_fetch_time,
    })


@app.route("/api/option_chain/<symbol>")
@login_required
def api_option_chain(symbol):
    price = None
    for s in current_signals:
        if s["symbol"] == symbol:
            price = s["price"]
            break

    if not price:
        if symbol in current_stock_infos:
            price = current_stock_infos[symbol].get("ltp")
            
    if not price:
        return jsonify({"error": f"Symbol {symbol} not found or no data"}), 404

    global kite, kite_access_token, kite_instruments
    
    if kite_access_token and kite_instruments:
        try:
            underlying_name = "NIFTY" if symbol == "^NSEI" else ("BANKNIFTY" if symbol == "^NSEBANK" else symbol)
            symbol_instruments = [i for i in kite_instruments if i["name"] == underlying_name]
            
            if symbol_instruments:
                expiries = sorted(list(set([i["expiry"] for i in symbol_instruments])))
                nearest_expiry = expiries[0]

                nearest_instruments = [i for i in symbol_instruments if i["expiry"] == nearest_expiry]
                strikes = sorted(list(set([i["strike"] for i in nearest_instruments])))
                
                atm_strike = min(strikes, key=lambda x: abs(x - float(price)))
                
                atm_idx = strikes.index(atm_strike)
                start_idx = max(0, atm_idx - 2)
                end_idx = min(len(strikes), atm_idx + 3)
                selected_strikes = strikes[start_idx:end_idx]

                selected_instruments = [i for i in nearest_instruments if i["strike"] in selected_strikes]
                trading_symbols = [f"NFO:{i['tradingsymbol']}" for i in selected_instruments]
                
                quotes = kite.quote(trading_symbols)

                chain = []
                from datetime import datetime
                for k in selected_strikes:
                    ce_inst = next((i for i in selected_instruments if i["strike"] == k and i["instrument_type"] == "CE"), None)
                    pe_inst = next((i for i in selected_instruments if i["strike"] == k and i["instrument_type"] == "PE"), None)
                    
                    ce_quote = quotes.get(f"NFO:{ce_inst['tradingsymbol']}", {}) if ce_inst else {}
                    pe_quote = quotes.get(f"NFO:{pe_inst['tradingsymbol']}", {}) if pe_inst else {}
                    
                    chain.append({
                        "strike": k,
                        "call": {
                            "price": ce_quote.get("last_price", 0), 
                            "delta": 0,
                            "iv": round(ce_quote.get("oi", 0) / 1000, 1) if ce_quote.get("oi", 0) else 0
                        },
                        "put": {
                            "price": pe_quote.get("last_price", 0), 
                            "delta": 0, 
                            "iv": round(pe_quote.get("oi", 0) / 1000, 1) if pe_quote.get("oi", 0) else 0
                        }
                    })

                return jsonify({
                    "symbol": symbol,
                    "spot_price": price,
                    "expiry": nearest_expiry,
                    "chain": chain,
                    "is_live_kite": True
                })
        except Exception as e:
            logger.error(f"Kite option chain error for {symbol}: {e}")
            pass

    if price < 500: step = 5
    elif price < 1000: step = 10
    elif price < 5000: step = 50
    else: step = 100

    atm_strike = round(price / step) * step
    strikes = [atm_strike - 2*step, atm_strike - step, atm_strike, atm_strike + step, atm_strike + 2*step]

    from src.pricing.black_scholes import BlackScholesPricer
    pricer = BlackScholesPricer()
    chain = []

    for k in strikes:
        vol = 0.20 + abs((k - atm_strike) / atm_strike) * 0.5
        tte = 7.0 / 365.0

        call_res = pricer.price(spot=price, strike=k, time_to_expiry=tte, volatility=vol, option_type="call")
        put_res = pricer.price(spot=price, strike=k, time_to_expiry=tte, volatility=vol, option_type="put")

        chain.append({
            "strike": k,
            "call": {"price": call_res.price, "delta": call_res.delta, "iv": round(vol * 100, 1)},
            "put": {"price": put_res.price, "delta": put_res.delta, "iv": round(vol * 100, 1)}
        })

    return jsonify({
        "symbol": symbol,
        "spot_price": price,
        "expiry": "Next Expiry (SYNTH)",
        "chain": chain,
        "auth_required": not kite_access_token
    })


# ── Main ──

def main():
    """Start the dashboard server."""
    print("=" * 60)
    print("  🚀 LIVE AI TRADING SIGNAL SYSTEM")
    print("=" * 60)
    print(f"  Dashboard: http://127.0.0.1:{CONFIG['dashboard']['port']}/")
    print(f"  API:       http://127.0.0.1:{CONFIG['dashboard']['port']}/api/signals")
    print(f"  Watchlist: {', '.join(CONFIG['watchlist'])}")
    print(f"  Refresh:   Every {CONFIG['data']['refresh_seconds']}s")
    print("=" * 60)

    logger.info("Performing initial data fetch...")
    refresh_data()

    bg_thread = threading.Thread(target=auto_refresh_loop, daemon=True)
    bg_thread.start()
    logger.info("Background refresh thread started")

    app.run(
        host=CONFIG["dashboard"].get("host", "0.0.0.0"),
        port=CONFIG["dashboard"].get("port", 5000),
        debug=CONFIG["dashboard"].get("debug", False),
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
