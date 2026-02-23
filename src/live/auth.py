"""
Authentication Blueprint — Handles login, signup, OAuth, and session management.
"""

import os
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
from src.live.models import db, User
import pytz

IST = pytz.timezone("Asia/Kolkata")

auth_bp = Blueprint("auth", __name__)
oauth = OAuth()


def init_oauth(app):
    """Initialize OAuth providers with app config."""
    oauth.init_app(app)

    oauth.register(
        name="google",
        client_id=app.config.get("GOOGLE_CLIENT_ID"),
        client_secret=app.config.get("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

    oauth.register(
        name="facebook",
        client_id=app.config.get("FACEBOOK_CLIENT_ID", ""),
        client_secret=app.config.get("FACEBOOK_CLIENT_SECRET", ""),
        access_token_url="https://graph.facebook.com/oauth/access_token",
        authorize_url="https://www.facebook.com/dialog/oauth",
        api_base_url="https://graph.facebook.com/",
        client_kwargs={"scope": "email public_profile"},
    )


def _record_login(user, request_obj):
    """Record login metadata for single-device enforcement."""
    user.last_login = datetime.now(IST)
    user.last_login_ip = request_obj.remote_addr or "Unknown"
    user.last_login_device = request_obj.headers.get("User-Agent", "Unknown")[:300]
    token = user.generate_session_token()
    session["session_token"] = token
    db.session.commit()


# ── Routes ──

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            _record_login(user, request)
            flash("Welcome back! 🚀", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html", mode="login")


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        sebi_ack = request.form.get("sebi_ack")

        if not sebi_ack:
            flash("You must acknowledge the SEBI disclaimer to proceed.", "error")
            return render_template("login.html", mode="signup")

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("auth.login"))

        user = User(name=name, email=email, is_new_user=True)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user, remember=True)
        _record_login(user, request)
        flash("Account created! Explore our intro offer 🎉", "success")
        return redirect(url_for("pricing_page"))

    return render_template("login.html", mode="signup")


@auth_bp.route("/logout")
@login_required
def logout():
    current_user.active_session_token = None
    db.session.commit()
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("auth.login"))


# ── Google OAuth ──

@auth_bp.route("/login/google")
def google_login():
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/login/google/callback")
def google_callback():
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo") or oauth.google.userinfo()

        google_id = user_info.get("sub")
        email = user_info.get("email", "").lower()
        name = user_info.get("name", "Google User")
        picture = user_info.get("picture", "")

        # Check if user exists by google_id or email
        user = User.query.filter_by(google_id=google_id).first()
        if not user:
            user = User.query.filter_by(email=email).first()
            if user:
                user.google_id = google_id
                if picture:
                    user.profile_photo = picture
            else:
                user = User(
                    email=email, name=name, google_id=google_id,
                    profile_photo=picture, is_new_user=True
                )
                db.session.add(user)

        db.session.commit()
        login_user(user, remember=True)
        _record_login(user, request)
        flash(f"Welcome, {user.name}! 🚀", "success")

        if user.is_new_user:
            return redirect(url_for("pricing_page"))
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Google login failed: {str(e)}", "error")
        return redirect(url_for("auth.login"))


# ── Facebook OAuth ──

@auth_bp.route("/login/facebook")
def facebook_login():
    redirect_uri = url_for("auth.facebook_callback", _external=True)
    return oauth.facebook.authorize_redirect(redirect_uri)


@auth_bp.route("/login/facebook/callback")
def facebook_callback():
    try:
        token = oauth.facebook.authorize_access_token()
        resp = oauth.facebook.get("me?fields=id,name,email,picture.type(large)")
        user_info = resp.json()

        fb_id = user_info.get("id")
        email = user_info.get("email", f"{fb_id}@facebook.com").lower()
        name = user_info.get("name", "Facebook User")
        picture = user_info.get("picture", {}).get("data", {}).get("url", "")

        user = User.query.filter_by(facebook_id=fb_id).first()
        if not user:
            user = User.query.filter_by(email=email).first()
            if user:
                user.facebook_id = fb_id
                if picture:
                    user.profile_photo = picture
            else:
                user = User(
                    email=email, name=name, facebook_id=fb_id,
                    profile_photo=picture, is_new_user=True
                )
                db.session.add(user)

        db.session.commit()
        login_user(user, remember=True)
        _record_login(user, request)
        flash(f"Welcome, {user.name}! 🚀", "success")

        if user.is_new_user:
            return redirect(url_for("pricing_page"))
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Facebook login failed: {str(e)}", "error")
        return redirect(url_for("auth.login"))


# ── Profile ──

@auth_bp.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user)


@auth_bp.route("/profile/update", methods=["POST"])
@login_required
def update_profile():
    name = request.form.get("name", "").strip()
    if name:
        current_user.name = name
        db.session.commit()
        flash("Profile updated!", "success")
    return redirect(url_for("auth.profile"))
