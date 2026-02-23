"""
Database Models — User accounts, subscriptions, and payments.
Uses Flask-SQLAlchemy with SQLite for lightweight local storage.
"""

from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import secrets

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User account model."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=True)  # Null for OAuth-only users
    name = db.Column(db.String(100), nullable=False, default="Trader")
    profile_photo = db.Column(db.String(500), nullable=True)  # URL or base64

    # OAuth fields
    google_id = db.Column(db.String(255), unique=True, nullable=True)
    facebook_id = db.Column(db.String(255), unique=True, nullable=True)

    # Session / device enforcement
    active_session_token = db.Column(db.String(128), nullable=True)
    last_login = db.Column(db.DateTime, nullable=True)
    last_login_ip = db.Column(db.String(50), nullable=True)
    last_login_device = db.Column(db.String(300), nullable=True)

    # Subscription
    subscription_plan = db.Column(db.String(30), default="none")  # none, intro, weekly, monthly, yearly
    subscription_expires_at = db.Column(db.DateTime, nullable=True)
    is_new_user = db.Column(db.Boolean, default=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    payments = db.relationship("Payment", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def generate_session_token(self):
        self.active_session_token = secrets.token_hex(32)
        return self.active_session_token

    @property
    def has_active_subscription(self):
        if self.subscription_plan == "none":
            return False
        if self.subscription_expires_at and self.subscription_expires_at > datetime.utcnow():
            return True
        return False

    @property
    def plan_display(self):
        names = {
            "none": "No Plan",
            "intro": "₹29 Intro (1 Week)",
            "weekly": "₹200 / Week",
            "monthly": "₹499 / Month",
            "yearly": "₹2999 / Year",
        }
        return names.get(self.subscription_plan, "Unknown")


class Payment(db.Model):
    """Payment transaction log."""
    __tablename__ = "payments"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    razorpay_order_id = db.Column(db.String(100), nullable=True)
    razorpay_payment_id = db.Column(db.String(100), nullable=True)
    razorpay_signature = db.Column(db.String(255), nullable=True)
    amount = db.Column(db.Integer, nullable=False)  # in paise
    plan_type = db.Column(db.String(30), nullable=False)
    status = db.Column(db.String(20), default="created")  # created, paid, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
