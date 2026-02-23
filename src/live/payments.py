"""
Payments Blueprint — Razorpay integration for subscription management.

Plans:
  - Intro: ₹29 / 1 week (new users only)
  - Weekly: ₹200 / week
  - Monthly: ₹499 / month
  - Yearly: ₹2999 / year
"""

from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from src.live.models import db, Payment
import razorpay
import hmac
import hashlib

payments_bp = Blueprint("payments", __name__)

PLANS = {
    "intro": {"amount": 2900, "label": "₹29 — 1 Week Intro", "days": 7},
    "weekly": {"amount": 20000, "label": "₹200 / Week", "days": 7},
    "monthly": {"amount": 49900, "label": "₹499 / Month", "days": 30},
    "yearly": {"amount": 299900, "label": "₹2,999 / Year", "days": 365},
}


def _get_client():
    """Get Razorpay client from app config."""
    key_id = current_app.config.get("RAZORPAY_KEY_ID", "")
    key_secret = current_app.config.get("RAZORPAY_KEY_SECRET", "")
    return razorpay.Client(auth=(key_id, key_secret))


@payments_bp.route("/api/create-order", methods=["POST"])
@login_required
def create_order():
    """Create a Razorpay order for a selected plan."""
    plan_type = request.json.get("plan", "monthly")

    # Intro offer only for new users
    if plan_type == "intro" and not current_user.is_new_user:
        return jsonify({"error": "Intro offer is only for new accounts"}), 400

    plan = PLANS.get(plan_type)
    if not plan:
        return jsonify({"error": "Invalid plan"}), 400

    try:
        client = _get_client()
        order = client.order.create({
            "amount": plan["amount"],
            "currency": "INR",
            "receipt": f"order_{current_user.id}_{plan_type}",
            "notes": {"plan": plan_type, "user_id": current_user.id},
        })

        # Save payment record
        payment = Payment(
            user_id=current_user.id,
            razorpay_order_id=order["id"],
            amount=plan["amount"],
            plan_type=plan_type,
            status="created",
        )
        db.session.add(payment)
        db.session.commit()

        return jsonify({
            "order_id": order["id"],
            "amount": plan["amount"],
            "currency": "INR",
            "key_id": current_app.config.get("RAZORPAY_KEY_ID"),
            "name": "AI Signal Engine",
            "description": plan["label"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@payments_bp.route("/api/verify-payment", methods=["POST"])
@login_required
def verify_payment():
    """Verify Razorpay payment signature and activate subscription."""
    data = request.json
    order_id = data.get("razorpay_order_id")
    payment_id = data.get("razorpay_payment_id")
    signature = data.get("razorpay_signature")

    if not all([order_id, payment_id, signature]):
        return jsonify({"error": "Missing payment details"}), 400

    # Verify signature
    key_secret = current_app.config.get("RAZORPAY_KEY_SECRET", "")
    msg = f"{order_id}|{payment_id}"
    expected = hmac.new(key_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    if expected != signature:
        return jsonify({"error": "Invalid payment signature"}), 400

    # Update payment record
    payment = Payment.query.filter_by(razorpay_order_id=order_id).first()
    if not payment:
        return jsonify({"error": "Order not found"}), 404

    payment.razorpay_payment_id = payment_id
    payment.razorpay_signature = signature
    payment.status = "paid"

    # Activate subscription
    plan = PLANS.get(payment.plan_type, {})
    current_user.subscription_plan = payment.plan_type
    current_user.subscription_expires_at = datetime.utcnow() + timedelta(days=plan.get("days", 30))
    if payment.plan_type == "intro":
        current_user.is_new_user = False

    db.session.commit()

    return jsonify({
        "status": "success",
        "plan": current_user.plan_display,
        "expires": current_user.subscription_expires_at.strftime("%d %b %Y"),
    })
