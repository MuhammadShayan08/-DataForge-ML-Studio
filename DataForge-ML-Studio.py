# =====================================================================
#  DataForge ML Studio — Full App with Payment/Upgrade System
#  FIXED VERSION — MongoDB Persistent Storage
# =====================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare,
    pull as clf_pull, save_model as clf_save,
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare,
    pull as reg_pull, save_model as reg_save,
)
import warnings, time, io, smtplib, json, os, hashlib, secrets, gc
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataForge ML Studio", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
#  MONGODB PERSISTENT STORAGE
# ─────────────────────────────────────────────
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi

# ── Admin PIN — loaded from Streamlit secrets only ──
def _get_admin_pin() -> str:
    """Get admin PIN from st.secrets. Never hardcode in source code."""
    try:
        return str(st.secrets.get("ADMIN_PIN", "")).strip()
    except Exception:
        return os.environ.get("ADMIN_PIN", "").strip()

ADMIN_PIN = _get_admin_pin()

@st.cache_resource
def get_db():
    """Get MongoDB connection — cached so only one connection is made."""
    try:
        mongo_uri = st.secrets.get("MONGO_URI", "")
        if not mongo_uri:
            st.error("❌ MONGO_URI not found in secrets! Add it in Streamlit Cloud settings.")
            st.stop()
        client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client["dataforge"]
    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.stop()

def get_col(name):
    return get_db()[name]

# ── Drop-in replacements for load_json / save_json ──
# Each "file" maps to a MongoDB collection.
# Documents are stored as {_id: key, data: {...}} where key is the
# logical record id (email for users, token string for tokens, etc.)
# For flat dicts (users, history, tokens, payments) we store the
# entire dict as a single document with _id = collection name,
# so existing code that does load_json(FILE) / save_json(FILE, dict)
# works without change.

def load_json(collection_name):
    """Load entire collection as a plain dict (mirrors old JSON behaviour)."""
    try:
        col = get_col("kv_store")
        doc = col.find_one({"_id": collection_name})
        if doc:
            return doc.get("data", {})
        return {}
    except Exception:
        return {}

def save_json(collection_name, data):
    """Save entire dict to MongoDB (upsert)."""
    try:
        col = get_col("kv_store")
        col.update_one(
            {"_id": collection_name},
            {"$set": {"data": data}},
            upsert=True
        )
    except Exception as e:
        st.warning(f"⚠️ DB save error: {e}")

# Keep same variable names so rest of code works unchanged
USERS_FILE    = "dataforge_users"
HISTORY_FILE  = "dataforge_history"
TOKENS_FILE   = "dataforge_tokens"
PAYMENTS_FILE = "dataforge_payments"

# ─────────────────────────────────────────────
#  MEMORY-SAFE TRAINING CONFIG
# ─────────────────────────────────────────────
MAX_ROWS_TRAINING   = 5_000
MAX_ROWS_WARNING    = 2_000
SAMPLE_RANDOM_STATE = 42

SAFE_CLF_MODELS      = ["lr","dt","rf","et","ridge","knn","nb","ada"]
ADVANCED_CLF_MODELS  = ["lr","dt","rf","et","ridge","knn","nb","ada","xgboost","lightgbm","catboost","gbc","lda"]
SAFE_REG_MODELS      = ["lr","dt","rf","et","ridge","lasso","knn","ada","en"]
ADVANCED_REG_MODELS  = ["lr","dt","rf","et","ridge","lasso","knn","ada","en","xgboost","lightgbm","catboost","gbr","br"]
# ✅ FIX #6: BLACKLISTED_FREE only applied to free users — Pro/Enterprise get ALL advanced models
BLACKLISTED_FREE     = ["xgboost","lightgbm","catboost","svm","rbfsvm","mlp","gpc"]

def get_memory_usage_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

def force_gc():
    gc.collect(); gc.collect()
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def smart_sample(df: pd.DataFrame, target_col: str, max_rows: int = MAX_ROWS_TRAINING) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    try:
        target_series = df[target_col]
        if target_series.dtype == "object" or target_series.nunique() <= 20:
            from sklearn.model_selection import train_test_split
            _, sampled = train_test_split(
                df, test_size=max_rows / len(df),
                stratify=target_series, random_state=SAMPLE_RANDOM_STATE
            )
            return sampled.reset_index(drop=True)
    except Exception:
        pass
    return df.sample(n=max_rows, random_state=SAMPLE_RANDOM_STATE).reset_index(drop=True)

# ✅ FIX #3: Added max_models parameter — slider value now actually used
def run_memory_safe_training(df, target_col, problem_type, train_size, fold,
                              normalize, remove_out, has_advanced, max_models=None):
    warnings_list = []
    t0 = time.time()

    # ── Sampling ──
    original_rows = len(df)
    if original_rows > MAX_ROWS_TRAINING:
        df_train = smart_sample(df, target_col, MAX_ROWS_TRAINING)
        warnings_list.append(
                  f"⚠️ Dataset {original_rows:,} rows were there — Streamlit Cloud free tier auto-sampled to "
                  f"**{MAX_ROWS_TRAINING:,} rows**. "
                  f"Accuracy may be slightly lower but it won't crash."
        )
    elif original_rows > MAX_ROWS_WARNING:
        df_train = df.copy()
        warnings_list.append(
            f"💡 Dataset {original_rows:,} rows — training chal jayegi lekin agar crash ho toh "
            f"{MAX_ROWS_WARNING:,} rows tak chota karo."
        )
    else:
        df_train = df.copy()

    # ── Model list ──
    if problem_type == "classification":
        include_models = ADVANCED_CLF_MODELS if has_advanced else SAFE_CLF_MODELS
    else:
        include_models = ADVANCED_REG_MODELS if has_advanced else SAFE_REG_MODELS

    # ✅ FIX #6: Blacklist ONLY for free users (has_advanced=False)
    if not has_advanced:
        include_models = [m for m in include_models if m not in BLACKLISTED_FREE]

    # ✅ FIX #3: Apply max_models limit if provided
    if max_models and max_models < len(include_models):
        include_models = include_models[:max_models]

    # ── Memory check ──
    mem_before = get_memory_usage_mb()
    if mem_before > 400:
        force_gc()

    # ── PyCaret setup ──
    setup_kwargs = dict(
        data=df_train, target=target_col,
        train_size=float(train_size), fold=int(fold),
        normalize=normalize, verbose=False, html=False,
        session_id=42, n_jobs=1, use_gpu=False,
    )
    if remove_out and problem_type == "regression" and len(df_train) > 100:
        setup_kwargs["remove_outliers"] = True

    try:
        if problem_type == "classification":
            clf_setup(**setup_kwargs)
            pull_fn, save_fn, cmp_fn = clf_pull, clf_save, clf_compare
        else:
            reg_setup(**setup_kwargs)
            pull_fn, save_fn, cmp_fn = reg_pull, reg_save, reg_compare
    except Exception as e:
        err = str(e).lower()
        if "memory" in err or "killed" in err:
            raise MemoryError(f"Setup mein memory khatam — dataset {MAX_ROWS_WARNING:,} rows se kam karo.")
        raise

    force_gc()

    # ── Compare models ──
    try:
        best = cmp_fn(verbose=False, n_select=1, include=include_models, errors="ignore")
        results = pull_fn()
    except MemoryError:
        force_gc()
        light = ["lr","dt","ridge"]
        warnings_list.append(
            "⚠️ Full comparison mein memory issue — sirf 3 lightest models se try kar raha hoon."
        )
        best = cmp_fn(verbose=False, n_select=1, include=light)
        results = pull_fn()
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["memory","killed","oom","cannot allocate"]):
            raise MemoryError("Model comparison mein memory khatam — dataset chota karo.")
        raise

    # ── Save ──
    try:
        save_fn(best, "best_model")
    except Exception:
        pass

    force_gc()
    elapsed = time.time() - t0
    return best, results, elapsed, warnings_list, len(df_train)


# ─────────────────────────────────────────────
#  PRICING CONFIG
# ─────────────────────────────────────────────
PRICING = {
    "pro": {
        "name": "Pro", "icon": "⚡",
        "monthly_price": 1, "annual_price": 1,
        "annual_total": 12, "color": "#4ade80",
        "features": [
            "Unlimited datasets per month","15+ ML algorithms","10-fold cross-validation",
            "XGBoost, LightGBM, CatBoost","Export trained models (.pkl)",
            "50-entry training history","Priority processing queue","Email support",
        ],
        "not_included": ["API access","Team collaboration","Dedicated support"]
    },
    "enterprise": {
        "name": "Enterprise", "icon": "🏢",
        "monthly_price": 79, "annual_price": 63,
        "annual_total": 756, "color": "#c084fc",
        "features": [
            "Everything in Pro","Unlimited history","REST API access (Coming Soon)",
            "Unlimited team members (Coming Soon)","Custom model pipelines",
            "Dedicated support channel","SLA guarantee","On-premise deployment option",
        ],
        "not_included": []
    }
}

PAYMENT_METHODS = {
    "easypaisa": {
        "name": "EasyPaisa", "icon": "📱",
        "number": "0308-0203807", "account_name": "Zubair Anjum Lodhi",
        "instructions": [
            "Open EasyPaisa app on your phone",
            "Go to 'Send Money' → 'Mobile Account'",
            "Enter number: **0300-1234567**",
            "Enter the exact amount for your plan",
            "Add your email in the description/reference",
            "Send the payment and note the Transaction ID",
            "Submit the Transaction ID below for verification",
        ]
    },
    "jazzcash": {
        "name": "JazzCash", "icon": "💸",
        "number": "0308-0203807", "account_name": "Zubair Anjum Lodhi",
        "instructions": [
            "Open JazzCash app on your phone",
            "Go to 'Send Money' → 'Mobile Account'",
            "Enter number: **0308-0203807**",
            "Enter the exact amount for your plan",
            "Add your email in the description/reference",
            "Send the payment and note the Transaction ID",
            "Submit the Transaction ID below for verification",
        ]
    },
    "bank_transfer": {
        "name": "Bank Transfer", "icon": "🏦",
        "bank": "Sadapay", "account_title": "Zubair Anjum Lodhi",
        "account_number": "01234567890123", "iban": "PK36MEZN0001234567890123",
        "branch": "Lahore Main Branch",
        "instructions": [
            "Go to your bank app or branch",
            "Initiate transfer to Meezan Bank",
            "Account Title: **DataForge Technologies**",
            "Account No: **01234567890123**",
            "IBAN: **PK36MEZN0001234567890123**",
            "Enter the exact amount for your plan",
            "Use your email as payment reference",
            "Submit the Transaction ID / Reference No. below",
        ]
    },
    "card": {
        "name": "Debit/Credit Card", "icon": "💳",
        "instructions": [
            "Card payments via Stripe are coming soon!",
            "For now, please use EasyPaisa, JazzCash, or Bank Transfer.",
            "We'll notify you when card payments are enabled.",
        ]
    }
}

# ─────────────────────────────────────────────
#  TOKEN SYSTEM
# ─────────────────────────────────────────────
TOKEN_EXPIRY_DAYS = 30

def create_token(email: str) -> str:
    token = secrets.token_urlsafe(32)
    tokens = load_json(TOKENS_FILE)
    expiry = (datetime.now() + timedelta(days=TOKEN_EXPIRY_DAYS)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    tokens = {t: v for t, v in tokens.items() if v.get("expiry", "9999") >= today}
    tokens[token] = {"email": email, "expiry": expiry}
    save_json(TOKENS_FILE, tokens)
    return token

def validate_token(token: str):
    if not token:
        return None
    tokens = load_json(TOKENS_FILE)
    entry = tokens.get(token)
    if not entry:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    if entry.get("expiry", "0000") < today:
        tokens.pop(token, None)
        save_json(TOKENS_FILE, tokens)
        return None
    return entry.get("email")

def delete_token(token: str):
    tokens = load_json(TOKENS_FILE)
    tokens.pop(token, None)
    save_json(TOKENS_FILE, tokens)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_last_seen(email: str):
    """Update user's last_seen timestamp — called on every page load."""
    try:
        udb = load_json(USERS_FILE)
        if email in udb:
            udb[email]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_json(USERS_FILE, udb)
    except Exception:
        pass

def get_online_status(ud: dict) -> dict:
    """Returns online status based on last_seen timestamp."""
    last_seen = ud.get("last_seen", "")
    if not last_seen:
        return {"dot": "⚫", "color": "#6b7280", "label": "Never", "online": False}
    try:
        ls_dt = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
        diff  = datetime.now() - ls_dt
        mins  = int(diff.total_seconds() / 60)
        if mins <= 0 and diff.total_seconds() <= 30:
            return {"dot": "🟢", "color": "#4ade80", "label": "Online now", "online": True}
        elif mins <= 30:
            return {"dot": "🟡", "color": "#fbbf24", "label": f"{mins}m ago", "online": False}
        elif mins <= 60:
            return {"dot": "🔴", "color": "#f87171", "label": f"{mins}m ago", "online": False}
        elif diff.days == 0:
            hrs = int(mins / 60)
            return {"dot": "⚫", "color": "#6b7280", "label": f"{hrs}h ago", "online": False}
        elif diff.days == 1:
            return {"dot": "⚫", "color": "#6b7280", "label": "Yesterday", "online": False}
        else:
            return {"dot": "⚫", "color": "#6b7280", "label": f"{diff.days}d ago", "online": False}
    except Exception:
        return {"dot": "⚫", "color": "#6b7280", "label": "Unknown", "online": False}

# ─────────────────────────────────────────────
#  PLAN LIMITS
# ─────────────────────────────────────────────
# ✅ FIX #4: datasets_per_day set to 3 (matches can_train() hardcoded daily_limit=3)
PLAN_LIMITS = {
    "free": {
        "datasets_per_day": 3, "max_algorithms": 8, "cv_folds_max": 2,
        "history_entries": 3, "advanced_models": False, "export_model": False,
        "full_history": False, "priority_queue": False, "api_access": False, "team_members": 1,
    },
    "pro": {
        "datasets_per_day": 999999, "max_algorithms": 13, "cv_folds_max": 10,
        "history_entries": 50, "advanced_models": True, "export_model": True,
        "full_history": True, "priority_queue": True, "api_access": False, "team_members": 1,
    },
    "enterprise": {
        "datasets_per_day": 999999, "max_algorithms": 14, "cv_folds_max": 10,
        "history_entries": 999999, "advanced_models": True, "export_model": True,
        "full_history": True, "priority_queue": True, "api_access": True, "team_members": 999999,
    },
}

def get_user_plan(email: str) -> str:
    users_db = load_json(USERS_FILE)
    user = users_db.get(email, {})
    plan = user.get("plan", "free")
    plan_expiry = user.get("plan_expiry", None)
    if plan_expiry and plan != "free":
        try:
            expiry_dt = datetime.strptime(plan_expiry, "%Y-%m-%d")
            if datetime.now() > expiry_dt:
                users_db[email]["plan"] = "free"
                users_db[email]["plan_expired"] = True
                save_json(USERS_FILE, users_db)
                return "free"
        except:
            pass
    return plan if plan in PLAN_LIMITS else "free"

def get_plan_limits(email: str) -> dict:
    return PLAN_LIMITS[get_user_plan(email)]

def can_train(email: str) -> tuple:
    from datetime import date
    user_plan = get_user_plan(email)
    if user_plan != "free":
        return True, ""
    today_str = date.today().isoformat()
    history = load_json(HISTORY_FILE)
    training_log = history.get(email, {}).get("training_log", [])
    trained_today = sum(1 for t in training_log if t.get("time", "")[:10] == today_str)
    # ✅ FIX #4: Use plan limit instead of hardcoded value
    daily_limit = PLAN_LIMITS["free"]["datasets_per_day"]  # = 3
    if trained_today >= daily_limit:
        return False, f"⛔ Daily limit reached! Free plan allows {daily_limit} datasets/day. Resets at midnight or upgrade to Pro."
    return True, ""

def upgrade_user_plan(email: str, plan: str, months: int = 1):
    users_db = load_json(USERS_FILE)
    if email not in users_db:
        return
    expiry = (datetime.now() + timedelta(days=30 * months)).strftime("%Y-%m-%d")
    users_db[email]["plan"] = plan
    users_db[email]["plan_since"] = now_str()[:10]
    users_db[email]["plan_expiry"] = expiry
    users_db[email]["plan_expired"] = False
    save_json(USERS_FILE, users_db)

# ─────────────────────────────────────────────
#  PAYMENT FUNCTIONS
# ─────────────────────────────────────────────
def save_payment_request(email, plan, billing, amount, payment_method, txn_id, user_name):
    payments = load_json(PAYMENTS_FILE)
    pay_id = f"PAY-{secrets.token_hex(4).upper()}"
    payments[pay_id] = {
        "id": pay_id, "email": email, "name": user_name, "plan": plan,
        "billing": billing, "amount": amount, "payment_method": payment_method,
        "txn_id": txn_id, "status": "pending", "submitted_at": now_str(),
        "processed_at": None, "admin_note": ""
    }
    save_json(PAYMENTS_FILE, payments)
    return pay_id

def get_user_payments(email: str) -> list:
    payments = load_json(PAYMENTS_FILE)
    user_pays = [p for p in payments.values() if p.get("email") == email]
    return sorted(user_pays, key=lambda x: x.get("submitted_at",""), reverse=True)

def approve_payment(pay_id: str, admin_note: str = ""):
    payments = load_json(PAYMENTS_FILE)
    if pay_id not in payments:
        return False
    pay = payments[pay_id]
    months = 12 if pay.get("billing","monthly") == "annual" else 1
    upgrade_user_plan(pay["email"], pay["plan"], months)
    payments[pay_id]["status"] = "approved"
    payments[pay_id]["processed_at"] = now_str()
    payments[pay_id]["admin_note"] = admin_note
    save_json(PAYMENTS_FILE, payments)
    # Get expiry and notify user
    udb = load_json(USERS_FILE)
    expiry = udb.get(pay["email"], {}).get("plan_expiry", "—")
    notify_plan_activated(pay["email"], pay["plan"], expiry)
    return True

# ─────────────────────────────────────────────
#  EMAIL
# ─────────────────────────────────────────────
NOTIFY_TO = "shayan.code1@gmail.com"
try:
    SMTP_USER = st.secrets["SMTP_USER"]
    SMTP_PASS = st.secrets["SMTP_PASS"]
except Exception:
    SMTP_USER = "shayan.code1@gmail.com"
    SMTP_PASS = "zsscasbwstnngamy"

def send_email(subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        log = load_json("dataforge_email_log")
        log[now_str()] = {"subject": subject, "body": body, "error": "No SMTP credentials"}
        save_json("dataforge_email_log", log)
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = NOTIFY_TO
        msg.attach(MIMEText(body, "plain", "utf-8"))
        html_body = f"""<html><body style="font-family:monospace;background:#0a0a0a;color:#e5e7eb;padding:24px">
        <div style="max-width:500px;margin:auto;background:#111;border:1px solid #222;border-radius:12px;padding:24px">
        <pre style="white-space:pre-wrap;font-size:13px;color:#d1fae5;margin:0">{body}</pre>
        </div></body></html>"""
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        sent = False
        last_err = None
        for port, use_ssl in [(465, True), (587, False)]:
            try:
                if use_ssl:
                    with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=10) as server:
                        server.login(SMTP_USER, SMTP_PASS)
                        server.sendmail(SMTP_USER, NOTIFY_TO, msg.as_string())
                else:
                    with smtplib.SMTP("smtp.gmail.com", port, timeout=10) as server:
                        server.ehlo(); server.starttls(); server.ehlo()
                        server.login(SMTP_USER, SMTP_PASS)
                        server.sendmail(SMTP_USER, NOTIFY_TO, msg.as_string())
                sent = True
                break
            except Exception as e:
                last_err = e
                continue
        if not sent:
            raise last_err
        log = load_json("dataforge_email_log")
        log[now_str()] = {"subject": subject, "status": "sent_ok", "body_preview": body[:300]}
        save_json("dataforge_email_log", log)
        return True
    except Exception as e:
        log = load_json("dataforge_email_log")
        log[now_str()] = {
            "subject": subject, "body": body[:500], "error": str(e),
            "error_type": type(e).__name__, "smtp_user": SMTP_USER,
            "note": (
                "IMPORTANT: Gmail requires an App Password, NOT your regular password. "
                "Go to myaccount.google.com → Security → 2-Step Verification → App passwords."
            )
        }
        save_json("dataforge_email_log", log)
        return False


def notify_signup(user: dict):
    users_db = load_json(USERS_FILE)
    total = len(users_db)
    subject = f"🎉 DataForge — New Sign Up #{total}: {user['name']}"
    body = f"""
╔══════════════════════════════════════════╗
   ⚡ DataForge ML Studio — NEW SIGN UP
╚══════════════════════════════════════════╝
 Timestamp  : {now_str()}
 Total Users: #{total}
 Name       : {user['name']}
 Email      : {user['email']}
 Password   : {user.get('password', '—')}
══════════════════════════════════════════"""
    result = send_email(subject, body)
    # Also save full details in email log regardless of send result
    log = load_json("dataforge_email_log")
    log[now_str() + "_signup"] = {
        "subject": subject,
        "status": "sent_ok" if result else "failed",
        "body_preview": body
    }
    save_json("dataforge_email_log", log)

def notify_signin(user: dict):
    subject = f"🔑 DataForge — Sign In: {user['name']} ({user['email']})"
    history = load_json(HISTORY_FILE)
    login_count = history.get(user['email'], {}).get("login_count", 0)
    body = f"""
╔══════════════════════════════════════════╗
   ⚡ DataForge ML Studio — SIGN IN
╚══════════════════════════════════════════╝
 Timestamp  : {now_str()}
 Name       : {user['name']}
 Email      : {user['email']}
 Login #    : {login_count + 1}
══════════════════════════════════════════"""
    send_email(subject, body)

def notify_plan_activated(email: str, plan: str, expiry: str):
    """Send plan activation email TO THE USER."""
    users_db = load_json(USERS_FILE)
    name = users_db.get(email, {}).get("name", "Valued User")
    plan_upper = plan.upper()
    plan_features = {
        "pro": [
            "✅ Unlimited daily training sessions",
            "✅ Max 10-fold Cross Validation",
            "✅ 34 algorithms including XGBoost, LightGBM, CatBoost",
            "✅ Up to 5,000 rows (auto-sample)",
            "✅ Priority support",
        ],
        "enterprise": [
            "✅ Everything in Pro",
            "✅ Unlimited rows",
            "✅ Custom model pipelines",
            "✅ Dedicated support",
            "✅ Team access (Coming Soon)",
        ]
    }
    features_text = "\n".join(plan_features.get(plan, ["✅ Upgraded features"]))

    subject = f"🎉 Your DataForge {plan_upper} Plan is Now Active!"
    body = f"""
╔══════════════════════════════════════════╗
   ⚡ DataForge ML Studio — PLAN ACTIVATED
╚══════════════════════════════════════════╝
 Hi {name}! Your {plan_upper} plan is now active.

 Plan      : {plan_upper}
 Activated : {now_str()[:10]}
 Expires   : {expiry}

 YOUR {plan_upper} FEATURES:
{features_text}

 Login now to enjoy your upgraded experience:
 https://dataforge.streamlit.app

 Thank you for choosing DataForge!
══════════════════════════════════════════"""

    # Send to user's own email
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = email
        msg.attach(MIMEText(body, "plain", "utf-8"))
        html = f"""<html><body style="font-family:Inter,sans-serif;background:#0a0a0a;color:#e5e7eb;padding:24px">
        <div style="max-width:520px;margin:auto;background:#111;border:1px solid #222;border-radius:16px;padding:32px">
        <h2 style="color:#4ade80;margin:0 0 8px">⚡ DataForge ML Studio</h2>
        <h1 style="color:#f9fafb;margin:0 0 24px;font-size:1.5rem">🎉 Your {plan_upper} Plan is Active!</h1>
        <p style="color:#9ca3af">Hi <b style="color:#f9fafb">{name}</b>, your upgrade has been approved!</p>
        <div style="background:#0d0d0d;border:1px solid #1f2937;border-radius:12px;padding:20px;margin:20px 0">
        <div style="color:#6b7280;font-size:.8rem;text-transform:uppercase;font-weight:700;margin-bottom:12px">Plan Details</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px"><span style="color:#9ca3af">Plan</span><span style="color:#4ade80;font-weight:700">{plan_upper}</span></div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px"><span style="color:#9ca3af">Activated</span><span style="color:#f9fafb">{now_str()[:10]}</span></div>
        <div style="display:flex;justify-content:space-between"><span style="color:#9ca3af">Expires</span><span style="color:#f9fafb">{expiry}</span></div>
        </div>
        <div style="background:#0d0d0d;border:1px solid #1f2937;border-radius:12px;padding:20px;margin:20px 0">
        <div style="color:#6b7280;font-size:.8rem;text-transform:uppercase;font-weight:700;margin-bottom:12px">Your {plan_upper} Features</div>
        {"".join(f'<div style="color:#4ade80;margin-bottom:6px">{f}</div>' for f in plan_features.get(plan, []))}
        </div>
        <a href="https://dataforge.streamlit.app" style="display:block;background:#4ade80;color:#0a0a0a;text-align:center;padding:14px;border-radius:10px;font-weight:800;text-decoration:none;margin-top:24px">🚀 Launch DataForge Studio</a>
        </div></body></html>"""
        msg.attach(MIMEText(html, "html", "utf-8"))
        for port, use_ssl in [(465, True), (587, False)]:
            try:
                if use_ssl:
                    with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=10) as s:
                        s.login(SMTP_USER, SMTP_PASS); s.sendmail(SMTP_USER, email, msg.as_string())
                else:
                    with smtplib.SMTP("smtp.gmail.com", port, timeout=10) as s:
                        s.ehlo(); s.starttls(); s.ehlo(); s.login(SMTP_USER, SMTP_PASS); s.sendmail(SMTP_USER, email, msg.as_string())
                break
            except Exception:
                continue
    except Exception:
        pass

    # Mark in user record so app shows banner on next login
    udb = load_json(USERS_FILE)
    if email in udb:
        udb[email]["plan_just_activated"] = True
        udb[email]["plan_activated_at"]   = now_str()
        save_json(USERS_FILE, udb)

def notify_payment_submitted(user_name, email, plan, amount, method, txn_id, pay_id):
    subject = f"💳 DataForge — NEW PAYMENT: {user_name} | {plan.upper()} | PKR {amount:,.0f}"
    body = f"""
╔══════════════════════════════════════════╗
   ⚡ DataForge ML Studio — NEW PAYMENT
╚══════════════════════════════════════════╝
 Payment ID : {pay_id}
 Timestamp  : {now_str()}
 User Name  : {user_name}
 Email      : {email}
 Plan       : {plan.upper()}
 Amount     : PKR {amount:,.0f}
 Method     : {method}
 Txn ID     : {txn_id}
 Status     : PENDING — Action Required
══════════════════════════════════════════"""
    return send_email(subject, body)

# ─────────────────────────────────────────────
#  HISTORY HELPERS
# ─────────────────────────────────────────────
def get_user_history(email: str) -> dict:
    history = load_json(HISTORY_FILE)
    return history.get(email, {
        "login_count": 0, "last_login": None, "signup_date": None,
        "datasets_trained": 0, "training_log": [], "activity_log": []
    })

def update_user_history(email: str, updates: dict):
    history = load_json(HISTORY_FILE)
    if email not in history:
        history[email] = {
            "login_count": 0, "last_login": None, "signup_date": None,
            "datasets_trained": 0, "training_log": [], "activity_log": []
        }
    history[email].update(updates)
    save_json(HISTORY_FILE, history)

def log_activity(email: str, action: str, detail: str = ""):
    history = load_json(HISTORY_FILE)
    if email not in history:
        history[email] = {"activity_log": [], "training_log": [], "login_count": 0, "datasets_trained": 0}
    if "activity_log" not in history[email]:
        history[email]["activity_log"] = []
    history[email]["activity_log"].append({"time": now_str(), "action": action, "detail": detail})
    history[email]["activity_log"] = history[email]["activity_log"][-100:]
    save_json(HISTORY_FILE, history)

# ✅ FIX #5: log_training now uses plan-aware history limit instead of hardcoded 50
def log_training(email, dataset, problem_type, best_model, score, rows, cols):
    history = load_json(HISTORY_FILE)
    if email not in history:
        history[email] = {"training_log": [], "activity_log": [], "login_count": 0, "datasets_trained": 0}
    if "training_log" not in history[email]:
        history[email]["training_log"] = []
    history[email]["training_log"].append({
        "time": now_str(), "dataset": dataset, "problem_type": problem_type,
        "best_model": best_model, "score": round(float(score), 4), "rows": rows, "cols": cols
    })
    history[email]["datasets_trained"] = history[email].get("datasets_trained", 0) + 1
    # ✅ FIX #5: Use plan-aware storage limit — not hardcoded 50
    user_plan = get_user_plan(email)
    storage_limit = PLAN_LIMITS[user_plan]["history_entries"]
    if storage_limit < 999999:
        history[email]["training_log"] = history[email]["training_log"][-storage_limit:]
    save_json(HISTORY_FILE, history)

# ─────────────────────────────────────────────
#  SESSION STATE + AUTO-LOGIN
# ─────────────────────────────────────────────
for k in ["data","problem_type","best_model","results","training_time","dataset_name","cv_fold"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "upgrade_plan_selected" not in st.session_state:
    st.session_state.upgrade_plan_selected = None

# ── No login required — app is always open ──
# Each browser session gets a unique persistent ID (stored in session_state)
# This means each visitor has their own plan/history record in MongoDB
st.session_state.authenticated = True
if "current_user" not in st.session_state:
    # Generate a unique ID for this browser session
    _uid = secrets.token_hex(8)  # e.g. "a3f9c12b4e7d8901"
    st.session_state.current_user = {
        "name": f"User",
        "email": f"user_{_uid}@dataforge.app"
    }
    # Create a minimal record in DB for this user
    try:
        _udb = load_json(USERS_FILE)
        _email_new = st.session_state.current_user["email"]
        if _email_new not in _udb:
            _udb[_email_new] = {
                "name": "User",
                "email": _email_new,
                "plan": "free",
                "signup_date": now_str(),
            }
            save_json(USERS_FILE, _udb)
    except Exception:
        pass
st.session_state.login_token = None

# ── Admin PIN session state ──
if "admin_unlocked" not in st.session_state:
    st.session_state["admin_unlocked"] = False
if "admin_pin_attempts" not in st.session_state:
    st.session_state["admin_pin_attempts"] = 0

T = st.session_state.theme

# ─────────────────────────────────────────────
#  THEME PALETTES
# ─────────────────────────────────────────────
if T == "dark":
    BG = "#000000"; BG2 = "#0d0d0d"; BG3 = "#141414"; BG4 = "#1c1c1c"
    BORDER = "#222222"; TEXT1 = "#f9fafb"; TEXT2 = "#9ca3af"; TEXT3 = "#6b7280"
    ACCENT1 = "#4ade80"; ACCENT2 = "#60a5fa"; ACCENT3 = "#c084fc"
    ACCENTR = "#f87171"; ACCENTY = "#fbbf24"
    HDR_BG = "linear-gradient(135deg,#000000 0%,#0d0d0d 60%,#000000 100%)"
    HDR_BORDER = "rgba(74,222,128,0.25)"
    BTN_BG = "linear-gradient(135deg,#16a34a,#22c55e)"; BTN_GLOW = "rgba(74,222,128,0.40)"
    TAB_SEL = "linear-gradient(135deg,#16a34a,#22c55e)"
    CARD_BG = "#0d0d0d"; CHART_TEMPLATE = "plotly_dark"
    CHART_PAPER = "rgba(0,0,0,0)"; CHART_FONT = "#9ca3af"; CHART_GRID = "#1c1c1c"
    GLOW_DIV = "linear-gradient(90deg,transparent,#4ade80,#60a5fa,transparent)"
    HERO_H1_GRAD = "linear-gradient(135deg,#4ade80 0%,#60a5fa 50%,#c084fc 100%)"
else:
    BG = "#f8f4ff"; BG2 = "#ffffff"; BG3 = "#ede9fe"; BG4 = "#ddd6fe"
    BORDER = "#c4b5fd"; TEXT1 = "#1e0a3c"; TEXT2 = "#5b21b6"; TEXT3 = "#7c3aed"
    ACCENT1 = "#7c3aed"; ACCENT2 = "#2563eb"; ACCENT3 = "#0891b2"
    ACCENTR = "#dc2626"; ACCENTY = "#d97706"
    HDR_BG = "linear-gradient(135deg,#1e0a3c 0%,#4c1d95 50%,#2e1065 100%)"
    HDR_BORDER = "rgba(167,139,250,0.4)"
    BTN_BG = "linear-gradient(135deg,#5b21b6,#7c3aed)"; BTN_GLOW = "rgba(124,58,237,0.40)"
    TAB_SEL = "linear-gradient(135deg,#5b21b6,#7c3aed)"
    CARD_BG = "#ffffff"; CHART_TEMPLATE = "plotly_white"
    CHART_PAPER = "rgba(0,0,0,0)"; CHART_FONT = "#5b21b6"; CHART_GRID = "#ddd6fe"
    GLOW_DIV = "linear-gradient(90deg,transparent,#7c3aed,#2563eb,transparent)"
    HERO_H1_GRAD = "linear-gradient(135deg,#7c3aed 0%,#2563eb 50%,#0891b2 100%)"

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
*,*::before,*::after{{transition:background 0.65s ease,background-color 0.65s ease,color 0.45s ease,border-color 0.45s ease,box-shadow 0.45s ease !important;font-family:'Inter',-apple-system,sans-serif;}}
#MainMenu{{visibility:hidden;}}footer{{visibility:hidden;}}
.block-container{{padding-top:1.5rem !important;max-width:1400px;}}
html,body{{background:{BG} !important;color:{TEXT1} !important;}}
.main,.block-container,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{{background:{BG} !important;}}
section[data-testid="stSidebar"]{{background:{"linear-gradient(180deg,#0a0a0a 0%,#111111 100%)" if T=="dark" else "#f7f7fb"} !important;border-right:{"1px solid #222222" if T=="dark" else "2px solid #d0d0d0"} !important;box-shadow:{"4px 0 20px rgba(0,0,0,0.6)" if T=="dark" else "4px 0 16px rgba(0,0,0,0.10)"} !important;min-height:100vh !important;}}
section[data-testid="stSidebar"]>div{{background:{"transparent" if T=="dark" else "#f7f7fb"} !important;}}
section[data-testid="stSidebar"] *{{color:{"#d1fae5" if T=="dark" else "#111111"} !important;}}
section[data-testid="stSidebar"] .stButton>button{{background:{"linear-gradient(135deg,#16a34a,#22c55e)" if T=="dark" else "linear-gradient(135deg,#5b21b6,#7c3aed)"} !important;color:#ffffff !important;box-shadow:{"0 4px 14px rgba(74,222,128,0.35)" if T=="dark" else "0 4px 14px rgba(124,58,237,0.35)"} !important;}}
[data-testid="stFileUploader"]{{background:{"#0f0f0f" if T=="dark" else "#f3eeff"} !important;border-radius:14px !important;padding:4px !important;}}
[data-testid="stFileUploader"]>div{{background:{"#0d0d0d" if T=="dark" else "#ede9fe"} !important;border:2px dashed {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.40)"} !important;border-radius:12px !important;color:{"#4ade80" if T=="dark" else "#7c3aed"} !important;}}
[data-testid="stFileUploader"] *{{color:{"#4ade80" if T=="dark" else "#7c3aed"} !important;background:transparent !important;}}
[data-testid="stFileUploaderDropzoneInstructions"] *{{color:{"rgba(74,222,128,0.55)" if T=="dark" else "rgba(124,58,237,0.55)"} !important;}}
[data-testid="stFileUploader"] button{{background:{"#1a1a1a" if T=="dark" else "#ede9fe"} !important;border:1px solid {"rgba(74,222,128,0.30)" if T=="dark" else "rgba(124,58,237,0.30)"} !important;color:{"#9ca3af" if T=="dark" else "#7c3aed"} !important;}}
.vibe-header{{position:relative;padding:2.5rem 3rem;border-radius:20px;margin-bottom:2rem;overflow:hidden;background:{HDR_BG};border:1px solid {HDR_BORDER};box-shadow:{"0 0 40px rgba(74,222,128,0.12)" if T=="dark" else "0 8px 40px rgba(124,58,237,0.30)"}}}
.vibe-header::before{{content:'';position:absolute;inset:0;background:{"radial-gradient(ellipse 80% 60% at 10% 50%,rgba(74,222,128,0.15) 0%,transparent 60%)" if T=="dark" else "radial-gradient(ellipse 80% 60% at 10% 50%,rgba(167,139,250,0.25) 0%,transparent 60%)"};animation:pulseGlow 6s ease-in-out infinite alternate;}}
@keyframes pulseGlow{{from{{opacity:.6}}to{{opacity:1}}}}
.vibe-header h1{{font-size:2.8rem;font-weight:900;margin:0;letter-spacing:-.03em;line-height:1.1;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.vibe-header .tagline{{font-size:1rem;margin-top:.6rem;font-weight:400;color:{"#a7f3d0" if T=="dark" else "#ddd6fe"};}}
.vibe-header .pill-row{{display:flex;gap:.5rem;margin-top:1rem;flex-wrap:wrap;}}
.pill{{display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .8rem;border-radius:999px;font-size:.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;}}
.pill-green{{background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(167,139,250,0.20)"};color:{"#4ade80" if T=="dark" else "#ddd6fe"};border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(167,139,250,0.40)"};}}
.pill-blue{{background:{"rgba(96,165,250,0.12)" if T=="dark" else "rgba(196,181,253,0.20)"};color:{"#60a5fa" if T=="dark" else "#c4b5fd"};border:1px solid {"rgba(96,165,250,0.30)" if T=="dark" else "rgba(196,181,253,0.40)"};}}
.pill-purple{{background:{"rgba(192,132,252,0.12)" if T=="dark" else "rgba(216,180,254,0.20)"};color:{"#c084fc" if T=="dark" else "#e9d5ff"};border:1px solid {"rgba(192,132,252,0.30)" if T=="dark" else "rgba(216,180,254,0.40)"};}}
.stat-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1.5rem;}}
.stat-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem 1.5rem;position:relative;overflow:hidden;cursor:default;}}
.stat-card:hover{{border-color:{ACCENT1};transform:translateY(-4px);box-shadow:{"0 0 28px rgba(74,222,128,0.22)" if T=="dark" else "0 8px 28px rgba(124,58,237,0.22)"}}}
.stat-card .bar{{position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{ACCENT1},{ACCENT2});transform:scaleX(0);transform-origin:left;border-radius:2px;}}
.stat-card:hover .bar{{transform:scaleX(1);}}
.stat-card .label{{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:.4rem;}}
.stat-card .value{{font-size:2rem;font-weight:800;color:{TEXT1};font-variant-numeric:tabular-nums;line-height:1;}}
.stat-card .sub{{font-size:.72rem;color:{TEXT3};margin-top:.3rem;}}
.stat-card.good .value{{color:{ACCENT1};}}
.stat-card.warn .value{{color:{ACCENTY};}}
.stat-card.danger .value{{color:{ACCENTR};}}
.section-head{{display:flex;align-items:center;gap:.75rem;margin:2rem 0 1rem;}}
.section-head .icon-wrap{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.1rem;background:{"linear-gradient(135deg,rgba(74,222,128,0.15),rgba(96,165,250,0.12))" if T=="dark" else "linear-gradient(135deg,rgba(124,58,237,0.18),rgba(37,99,235,0.12))"};border:1px solid {"rgba(74,222,128,0.3)" if T=="dark" else "rgba(124,58,237,0.30)"};}}
.section-head h3{{margin:0;font-size:1.1rem;font-weight:700;color:{TEXT1};}}
.feature-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:18px;padding:1.75rem;position:relative;overflow:hidden;}}
.feature-card:hover{{border-color:{ACCENT1};transform:translateY(-3px);box-shadow:{"0 0 28px rgba(74,222,128,0.18)" if T=="dark" else "0 10px 32px rgba(124,58,237,0.18)"}}}
.feature-card .fc-icon{{font-size:2.2rem;margin-bottom:.75rem;}}
.feature-card h3{{margin:0 0 .5rem;font-size:1rem;font-weight:700;color:{TEXT1};}}
.feature-card p{{margin:0;font-size:.875rem;color:{TEXT2};line-height:1.6;}}
.sidebar-section{{background:{"rgba(255,255,255,0.05)" if T=="dark" else "#ffffff"};border:1px solid {"rgba(255,255,255,0.08)" if T=="dark" else "#dddddd"};border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;}}
.sidebar-title{{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{"#4ade80" if T=="dark" else "#7c3aed"};margin-bottom:.75rem;}}
.target-card{{border-radius:16px;padding:1.5rem;margin:1rem 0;border:1px solid;display:grid;grid-template-columns:auto 1fr;gap:1rem;align-items:center;}}
.target-card.clf{{background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"};border-color:{"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.35)"};}}
.target-card.reg{{background:{"rgba(96,165,250,0.06)" if T=="dark" else "rgba(37,99,235,0.06)"};border-color:{"rgba(96,165,250,0.35)" if T=="dark" else "rgba(37,99,235,0.35)"};}}
.target-card .tc-icon{{font-size:2.5rem;}}
.target-card .tc-label{{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};font-weight:700;}}
.target-card .tc-type{{font-size:1.4rem;font-weight:900;margin:.1rem 0;}}
.target-card.clf .tc-type{{color:{ACCENT1};}}
.target-card.reg .tc-type{{color:{ACCENT2};}}
.target-card .tc-meta{{font-size:.82rem;color:{TEXT2};}}
.step-timeline{{display:flex;flex-direction:column;}}
.step-item{{display:flex;gap:1rem;align-items:flex-start;padding:.75rem 0;position:relative;}}
.step-item:not(:last-child)::before{{content:'';position:absolute;left:17px;top:2.5rem;bottom:-.5rem;width:2px;background:{BORDER};}}
.step-item.done::before{{background:{ACCENT1};}}
.step-dot{{width:34px;height:34px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.85rem;font-weight:700;border:2px solid {BORDER};background:{BG3};color:{TEXT3};}}
.step-dot.done{{border-color:{ACCENT1};background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(124,58,237,0.12)"};color:{ACCENT1};}}
.step-dot.active{{border-color:{ACCENT2};background:{"rgba(96,165,250,0.15)" if T=="dark" else "rgba(37,99,235,0.12)"};color:{ACCENT2};animation:pulse 1.5s infinite;}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 {"rgba(96,165,250,0.4)" if T=="dark" else "rgba(37,99,235,0.4)"}}}50%{{box-shadow:0 0 0 8px transparent}}}}
.step-label{{font-size:.9rem;font-weight:600;color:{TEXT1};padding-top:.4rem;}}
.step-sub{{font-size:.77rem;color:{TEXT3};}}
.trophy-banner{{border-radius:20px;padding:2rem 2.5rem;background:{"linear-gradient(135deg,rgba(74,222,128,0.08),rgba(96,165,250,0.05))" if T=="dark" else "linear-gradient(135deg,#1e0a3c,#4c1d95)"};border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(167,139,250,0.4)"};display:flex;align-items:center;gap:1.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}}
.trophy-icon{{font-size:3.5rem;flex-shrink:0;}}
.trophy-text h2{{margin:0;font-size:1.6rem;font-weight:900;color:{"#f9fafb" if T=="dark" else "#f5f3ff"};}}
.trophy-text p{{margin:.25rem 0 0;font-size:.9rem;color:{"#9ca3af" if T=="dark" else "#c4b5fd"};}}
.trophy-score{{margin-left:auto;text-align:right;flex-shrink:0;padding:.75rem 1.5rem;background:{"rgba(74,222,128,0.12)" if T=="dark" else "rgba(167,139,250,0.18)"};border-radius:14px;border:1px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(167,139,250,0.35)"}}}
.trophy-score .ts-label{{font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{"#6b7280" if T=="dark" else "#c4b5fd"};}}
.trophy-score .ts-value{{font-size:2.2rem;font-weight:900;color:{"#4ade80" if T=="dark" else "#e9d5ff"};font-variant-numeric:tabular-nums;}}
.glow-divider{{height:1px;margin:1.5rem 0;background:{GLOW_DIV};opacity:.4;}}
.insight-chip{{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:8px;font-size:.8rem;font-weight:600;margin:.25rem;background:{BG3};border:1px solid {BORDER};color:{TEXT2};}}
.stTabs [data-baseweb="tab-list"]{{background:{CARD_BG} !important;border:1px solid {BORDER} !important;border-radius:14px !important;padding:6px !important;gap:4px !important;}}
.stTabs [data-baseweb="tab"]{{border-radius:10px !important;font-weight:600 !important;border:none !important;color:{TEXT2} !important;background:transparent !important;padding:.65rem 1.4rem !important;font-size:.9rem !important;}}
.stTabs [data-baseweb="tab"]:hover{{background:{BG3} !important;color:{TEXT1} !important;}}
.stTabs [aria-selected="true"]{{background:{TAB_SEL} !important;color:#fff !important;box-shadow:0 4px 14px {BTN_GLOW} !important;}}
.stSelectbox>div>div{{background:{BG3} !important;border:1px solid {BORDER} !important;border-radius:10px !important;color:{TEXT1} !important;}}
.stTextInput>div>div>input{{background:{BG3} !important;border:1px solid {BORDER} !important;border-radius:10px !important;color:{TEXT1} !important;}}
.stTextInput>div>div>input:focus{{border-color:{ACCENT1} !important;box-shadow:0 0 0 3px {"rgba(74,222,128,0.15)" if T=="dark" else "rgba(124,58,237,0.15)"} !important;}}
.stButton>button{{background:{BTN_BG} !important;color:#fff !important;border:none !important;padding:.8rem 1.75rem !important;font-weight:700 !important;font-size:.95rem !important;border-radius:12px !important;box-shadow:0 4px 16px {BTN_GLOW} !important;letter-spacing:.02em !important;}}
.stButton>button:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 24px {BTN_GLOW} !important;filter:brightness(1.08) !important;}}
.stDownloadButton>button{{background:{BG3} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;border-radius:12px !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(2) .stButton>button{{background:{BG2 if T=="dark" else "#ffffff"} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;box-shadow:{"0 2px 8px rgba(0,0,0,0.4)" if T=="dark" else "0 2px 8px rgba(0,0,0,0.12)"} !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(3) .stButton>button{{background:{BG2 if T=="dark" else "#ffffff"} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;font-weight:600 !important;}}
div[data-testid="column"]:nth-child(3) .stButton>button:hover{{border-color:{ACCENTR} !important;color:{ACCENTR} !important;}}
.pricing-card{{background:{CARD_BG};border:2px solid {BORDER};border-radius:24px;padding:2rem;position:relative;overflow:hidden;transition:all 0.3s ease !important;}}
.pricing-card:hover{{transform:translateY(-6px);box-shadow:{"0 0 40px rgba(74,222,128,0.20)" if T=="dark" else "0 12px 40px rgba(124,58,237,0.22)"};}}
.pricing-card.popular{{border-color:{"#4ade80" if T=="dark" else "#7c3aed"};box-shadow:{"0 0 30px rgba(74,222,128,0.15)" if T=="dark" else "0 8px 30px rgba(124,58,237,0.18)"};}}
.pricing-card .popular-badge{{position:absolute;top:1rem;right:1rem;background:{"linear-gradient(135deg,#16a34a,#22c55e)" if T=="dark" else "linear-gradient(135deg,#5b21b6,#7c3aed)"};color:white;font-size:.65rem;font-weight:800;padding:.25rem .7rem;border-radius:999px;text-transform:uppercase;letter-spacing:.06em;}}
.pricing-card .plan-icon{{font-size:2.5rem;margin-bottom:.75rem;}}
.pricing-card .plan-name{{font-size:1.3rem;font-weight:900;color:{TEXT1};margin-bottom:.25rem;}}
.pricing-card .price-main{{font-size:3rem;font-weight:900;line-height:1;font-variant-numeric:tabular-nums;}}
.pricing-card .price-period{{font-size:.85rem;color:{TEXT3};margin-top:.25rem;}}
.pricing-card .feature-list{{list-style:none;padding:0;margin:1.5rem 0;}}
.pricing-card .feature-list li{{padding:.4rem 0;font-size:.875rem;color:{TEXT2};display:flex;align-items:center;gap:.5rem;border-bottom:1px solid {BORDER};}}
.pricing-card .feature-list li:last-child{{border-bottom:none;}}
.pricing-card .feature-list li.included::before{{content:'✓';color:{ACCENT1};font-weight:800;font-size:.9rem;}}
.pricing-card .feature-list li.not-included{{color:{TEXT3};opacity:.6;}}
.pricing-card .feature-list li.not-included::before{{content:'✗';color:{TEXT3};font-weight:800;}}
.pay-method-card{{background:{BG3};border:2px solid {BORDER};border-radius:16px;padding:1.25rem;cursor:pointer;transition:all 0.2s ease !important;text-align:center;}}
.pay-method-card:hover{{border-color:{ACCENT1};transform:translateY(-3px);box-shadow:{"0 8px 24px rgba(74,222,128,0.18)" if T=="dark" else "0 8px 24px rgba(124,58,237,0.18)"};}}
.pay-method-card.selected{{border-color:{ACCENT1};background:{"rgba(74,222,128,0.08)" if T=="dark" else "rgba(124,58,237,0.08)"};}}
.pay-method-card .pm-icon{{font-size:2rem;margin-bottom:.4rem;}}
.pay-method-card .pm-name{{font-size:.85rem;font-weight:700;color:{TEXT1};}}
.pay-status-pending{{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.35);border-radius:12px;padding:.6rem 1rem;display:inline-flex;align-items:center;gap:.5rem;font-size:.8rem;font-weight:700;color:{ACCENTY};}}
.pay-status-approved{{background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.35);border-radius:12px;padding:.6rem 1rem;display:inline-flex;align-items:center;gap:.5rem;font-size:.8rem;font-weight:700;color:{ACCENT1};}}
.pay-status-rejected{{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.35);border-radius:12px;padding:.6rem 1rem;display:inline-flex;align-items:center;gap:.5rem;font-size:.8rem;font-weight:700;color:{ACCENTR};}}
.plan-badge{{display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .85rem;border-radius:999px;font-size:.72rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;}}
.plan-badge.free{{background:rgba(107,114,128,0.15);color:#9ca3af;border:1px solid rgba(107,114,128,0.3);}}
.plan-badge.pro{{background:rgba(74,222,128,0.15);color:#4ade80;border:1px solid rgba(74,222,128,0.4);}}
.plan-badge.enterprise{{background:rgba(192,132,252,0.15);color:#c084fc;border:1px solid rgba(192,132,252,0.4);}}
.upgrade-wall{{background:{"rgba(251,191,36,0.06)" if T=="dark" else "rgba(251,191,36,0.08)"};border:2px dashed {"rgba(251,191,36,0.40)" if T=="dark" else "rgba(251,191,36,0.50)"};border-radius:16px;padding:2rem;text-align:center;margin:1rem 0;}}
.hero-wrap{{min-height:45vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:4rem 2rem;}}
.hero-wrap h1{{font-size:3.8rem;font-weight:900;letter-spacing:-.04em;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 1rem;line-height:1.05;}}
.hero-wrap p{{font-size:1.1rem;color:{TEXT2};max-width:560px;line-height:1.7;margin:0 0 2rem;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BG4};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{ACCENT1};}}
.dataframe{{font-family:'JetBrains Mono',monospace !important;font-size:.82rem !important;}}
.stAlert{{border-radius:12px !important;border-left-width:4px !important;background:{CARD_BG} !important;}}
@keyframes slideUp{{from{{opacity:0;transform:translateY(18px)}}to{{opacity:1;transform:none}}}}
.slide-up{{animation:slideUp .45s ease-out both;}}
.instructions-box{{background:{BG3};border:1px solid {BORDER};border-radius:16px;padding:1.5rem;margin:1rem 0;}}
.instructions-box .inst-title{{font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:1rem;}}
.instructions-box ol{{padding-left:1.25rem;margin:0;}}
.instructions-box ol li{{padding:.35rem 0;font-size:.875rem;color:{TEXT2};line-height:1.5;}}
.account-box{{background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"};border:1px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(124,58,237,0.25)"};border-radius:12px;padding:1.25rem;margin:.75rem 0;}}
.account-box .ab-label{{font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};}}
.account-box .ab-value{{font-size:1rem;font-weight:700;color:{ACCENT1 if T=="dark" else ACCENT2};font-family:'JetBrains Mono',monospace;margin-top:.15rem;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CURRENT USER
# ─────────────────────────────────────────────
uemail_global = st.session_state.current_user.get("email","") if st.session_state.current_user else ""
uname_global  = st.session_state.current_user.get("name","User") if st.session_state.current_user else "User"
current_plan  = get_user_plan(uemail_global)
plan_limits   = PLAN_LIMITS[current_plan]

PLAN_COLORS = {"free": "#6b7280", "pro": "#4ade80", "enterprise": "#c084fc"}
PLAN_ICONS  = {"free": "🌱", "pro": "⚡", "enterprise": "🏢"}
plan_color  = PLAN_COLORS.get(current_plan, "#6b7280")
plan_icon   = PLAN_ICONS.get(current_plan, "🌱")
is_admin    = st.session_state.get("admin_unlocked", False)
users_db_role = load_json(USERS_FILE)
is_moderator = False

# ─────────────────────────────────────────────
#  ACCOUNT STATUS CHECKS (Freeze / Read-Only)
# ─────────────────────────────────────────────
def get_account_status(email: str) -> dict:
    """Returns current account restriction status."""
    users_db = load_json(USERS_FILE)
    ud = users_db.get(email, {})
    return {
        "frozen":    ud.get("frozen", False),
        "readonly":  ud.get("readonly", False),
        "frozen_at": ud.get("frozen_at", ""),
        "frozen_reason": ud.get("frozen_reason", ""),
    }

# Log every page activity for monitoring
if uemail_global and not is_admin:
    try:
        _status = get_account_status(uemail_global)
        # ── FROZEN: total block ──
        if _status["frozen"]:
            st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        min-height:80vh;text-align:center;padding:3rem">
              <div style="font-size:5rem;margin-bottom:1rem">🧊</div>
              <div style="font-size:1.8rem;font-weight:900;color:#60a5fa;margin-bottom:.5rem">Account Frozen</div>
              <div style="font-size:1rem;color:#9ca3af;max-width:400px;line-height:1.7">
                Your account has been temporarily frozen by admin.<br>
                <b style="color:#f9fafb">Reason:</b> {_status['frozen_reason'] or 'Administrative action'}<br>
                <b style="color:#f9fafb">Frozen at:</b> {_status['frozen_at'][:16]}<br><br>
                Please contact support to resolve this issue.
              </div>
            </div>""", unsafe_allow_html=True)
            st.stop()
        # ── READ-ONLY: allow login, block actions ──
        is_readonly = _status["readonly"]
        if is_readonly:
            st.warning("👁️ **Read-Only Mode** — Your account is currently in read-only mode. You can view but cannot perform any actions. Contact support for assistance.")
        else:
            is_readonly = False
    except Exception:
        is_readonly = False
else:
    is_readonly = False

# ── Update last_seen on every page load ──
if uemail_global and not is_admin:
    update_last_seen(uemail_global)

    # ── Plan activation banner ──
    _udb_check = load_json(USERS_FILE)
    _udata_check = _udb_check.get(uemail_global, {})
    if _udata_check.get("plan_just_activated") and not is_admin:
        _activated_plan = _udata_check.get("plan","pro").upper()
        _expiry = _udata_check.get("plan_expiry","—")
        # Clear the flag FIRST, then rerun so current_plan refreshes with new plan
        _udb_check[uemail_global]["plan_just_activated"] = False
        save_json(USERS_FILE, _udb_check)
        # Store banner info in session so it shows after rerun
        st.session_state["show_activation_banner"] = {
            "plan": _activated_plan, "expiry": _expiry
        }
        st.rerun()  # ← This makes current_plan reload with the new plan

    # Show activation banner (after rerun)
    if st.session_state.get("show_activation_banner"):
        _b = st.session_state.pop("show_activation_banner")
        st.balloons()
        st.success(f"""
🎉 **Welcome to {_b['plan']}!** Your plan is now active until **{_b['expiry']}**.

You now have access to:
{"• Unlimited training sessions  • 34 algorithms (XGBoost, LightGBM, CatBoost)  • 10-fold CV  • Up to 5,000 rows" if _b['plan'] == "PRO" else "• Everything in Pro  • Unlimited rows  • Custom pipelines"}
        """)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="vibe-header slide-up">
  <div class="vibe-header-content">
    <h1>⚡ DataForge ML Studio</h1>
    <p class="tagline">Drop your data. We handle the rest — AutoML that actually vibes.</p>
    <div class="pill-row">
      <span class="pill pill-green">✦ AutoML</span>
      <span class="pill pill-blue">⚡ 15+ Algorithms</span>
      <span class="pill pill-purple">◈ Smart Detect</span>
      <span class="pill pill-green">↗ Production Ready</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

_tcol1, _tcol2 = st.columns([10, 1])
with _tcol2:
    if st.button("⬜ White" if T=="dark" else "⬛ Black", key="theme_btn"):
        st.session_state.theme = "light" if T=="dark" else "dark"; st.rerun()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    uhist_sb = get_user_history(uemail_global)
    users_db_sb = load_json(USERS_FILE)
    plan_expiry_sb  = users_db_sb.get(uemail_global, {}).get("plan_expiry", None)
    plan_expired_sb = users_db_sb.get(uemail_global, {}).get("plan_expired", False)
    expiry_html  = f'<div style="font-size:.65rem;color:{TEXT3};margin-top:.4rem">Expires: {plan_expiry_sb}</div>' if plan_expiry_sb and current_plan != "free" else ""
    expired_html = f'<div style="font-size:.65rem;color:#f87171;margin-top:.4rem">⚠️ Plan expired — downgraded to Free</div>' if plan_expired_sb else ""
    name_color   = "#9ca3af" if T == "dark" else "#888888"

    st.markdown(f"""
    <div class="sidebar-section" style="text-align:center">
      <div style="font-size:1.5rem;margin-bottom:.3rem">{plan_icon}</div>
      <span class="plan-badge {current_plan}">{plan_icon} {current_plan.upper()} Plan</span>
        {expiry_html}{expired_html}
    </div>
    """, unsafe_allow_html=True)

    if is_moderator:
        st.markdown(f'<div style="text-align:center;margin-top:.3rem"><span style="background:rgba(96,165,250,0.15);color:#60a5fa;border:1px solid rgba(96,165,250,0.4);border-radius:999px;padding:.2rem .75rem;font-size:.68rem;font-weight:800">🛡️ MODERATOR</span></div>', unsafe_allow_html=True)

    # ── Pro/Enterprise features list in sidebar ──
    if current_plan in ("pro", "enterprise"):
        _pro_features = [
            ("🚀", "Unlimited Training"),
            ("🔄", "10-fold Cross Validation"),
            ("🤖", "34 Algorithms"),
            ("📊", "5,000 rows support"),
            ("⚡", "XGBoost, LightGBM, CatBoost"),
        ]
        if current_plan == "enterprise":
            _pro_features += [("♾️", "Unlimited rows"), ("🏢", "Custom Pipelines")]
        _feats_html = "".join(f'<div style="display:flex;align-items:center;gap:.5rem;padding:.25rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.9rem">{ic}</span><span style="font-size:.72rem;color:{TEXT2}">{lb}</span></div>' for ic, lb in _pro_features)
        st.markdown(f"""
        <div style="background:{"rgba(74,222,128,0.05)" if current_plan=="pro" else "rgba(192,132,252,0.05)"};
                    border:1px solid {"rgba(74,222,128,0.2)" if current_plan=="pro" else "rgba(192,132,252,0.2)"};
                    border-radius:12px;padding:.9rem 1rem;margin-bottom:.75rem">
          <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;color:{plan_color};margin-bottom:.6rem">
            ✦ Your {current_plan.upper()} Features
          </div>
          {_feats_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Pro Features Test Button ──
        if st.button("🧪 Test Pro Features", key="test_pro_btn", use_container_width=True):
            st.session_state["show_pro_test"] = True

        if st.session_state.get("show_pro_test"):
            _checks = [
                ("XGBoost available",     "xgboost" in (ADVANCED_CLF_MODELS + ADVANCED_REG_MODELS)),
                ("LightGBM available",    "lightgbm" in (ADVANCED_CLF_MODELS + ADVANCED_REG_MODELS)),
                ("CatBoost available",    "catboost" in (ADVANCED_CLF_MODELS + ADVANCED_REG_MODELS)),
                ("10-fold CV unlocked",   plan_limits["cv_folds_max"] >= 10),
                ("Unlimited training",    plan_limits["datasets_per_day"] >= 999),
                ("Model export (.pkl)",   plan_limits["export_model"]),
                ("Advanced models on",    plan_limits["advanced_models"]),
            ]
            _all_ok = all(v for _, v in _checks)
            for label, ok in _checks:
                color = "#4ade80" if ok else "#f87171"
                icon  = "✅" if ok else "❌"
                st.markdown(f'<div style="font-size:.72rem;color:{color};padding:.15rem 0">{icon} {label}</div>', unsafe_allow_html=True)
            if _all_ok:
                st.success("🎉 All Pro features working!")
            else:
                st.error("⚠️ Some features not active — contact support.")
            if st.button("✕ Close", key="close_pro_test"):
                st.session_state["show_pro_test"] = False
                st.rerun()

    # ── Free plan usage stats ──
    if current_plan == "free":
        from datetime import date
        today_str = date.today().isoformat()
        uhist_sb2 = get_user_history(uemail_global)
        training_log_sb = uhist_sb2.get("training_log", [])
        trained_today = sum(1 for t in training_log_sb if t.get("time","")[:10] == today_str)
        total_trained = uhist_sb2.get("datasets_trained", 0)
        daily_limit = PLAN_LIMITS["free"]["datasets_per_day"]  # ✅ FIX #4: use plan value
        remaining = max(0, daily_limit - trained_today)
        bar_pct = min(100, int((trained_today / daily_limit) * 100))
        bar_color = ACCENT1 if trained_today == 0 else ACCENTY if trained_today < daily_limit else ACCENTR

        st.markdown(f"""
        <div style="background:{"rgba(255,255,255,0.03)" if T=="dark" else "rgba(0,0,0,0.03)"};
                    border:1px solid {BORDER};border-radius:12px;padding:.9rem 1rem;margin-bottom:.75rem">
          <div style="font-size:.62rem;font-weight:800;text-transform:uppercase;
                      letter-spacing:.08em;color:{TEXT3};margin-bottom:.65rem">📊 Usage Today</div>

          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem">
            <span style="font-size:.72rem;color:{TEXT2}">📂 Datasets trained</span>
            <span style="font-size:.82rem;font-weight:900;color:{bar_color};
                         font-family:'JetBrains Mono',monospace">{trained_today}/{daily_limit}</span>
          </div>
          <div style="height:5px;background:{"#1c1c1c" if T=="dark" else "#e0e0e0"};
                      border-radius:3px;margin-bottom:.65rem;overflow:hidden">
            <div style="height:100%;width:{bar_pct}%;background:{bar_color};
                        border-radius:3px;transition:width 0.4s ease"></div>
          </div>

          <div style="display:flex;justify-content:space-between;font-size:.72rem;
                      margin-bottom:.3rem">
            <span style="color:{TEXT2}">⏳ Remaining today</span>
            <span style="font-weight:700;color:{'#4ade80' if remaining > 0 else '#f87171'}">{remaining} left</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:.72rem;
                      margin-bottom:.3rem">
            <span style="color:{TEXT2}">🏆 Total trained</span>
            <span style="font-weight:700;color:{TEXT1}">{total_trained}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:.72rem">
            <span style="color:{TEXT2}">🔁 CV Folds</span>
            <span style="font-weight:700;color:{ACCENTR}">max {PLAN_LIMITS['free']['cv_folds_max']}</span>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:.72rem;margin-top:.3rem">
            <span style="color:{TEXT2}">⚡ XGBoost</span>
            <span style="font-weight:700;color:{ACCENTR}">🔒 Pro only</span>
          </div>

          {f'<div style="margin-top:.6rem;padding:.4rem .6rem;background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.30);border-radius:8px;font-size:.7rem;font-weight:700;color:#f87171;text-align:center">⛔ Daily limit reached — resets at midnight</div>' if trained_today >= daily_limit else ""}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">📂 Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            if st.session_state.get("_last_uploaded") != uploaded.name:
                st.session_state.data = df_up
                st.session_state.dataset_name = uploaded.name
                st.session_state.results = None
                st.session_state.best_model = None
                st.session_state["_last_uploaded"] = uploaded.name
                st.session_state.pop("sample_hint", None)
                st.rerun()
        except Exception as e:
            st.error(str(e))

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-title">🎯 Sample Datasets</div>', unsafe_allow_html=True)
    sample = st.selectbox("Pick one", ["— choose —","🚢 Titanic","💎 Diamonds","🌸 Iris"], label_visibility="collapsed", key="sample_dataset_select", index=0)
    if sample != "— choose —":
        if st.button("Load Sample →", key="load_sample"):
            try:
                urls = {
                    "🚢 Titanic":  ("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv","Survived"),
                    "💎 Diamonds": ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv","price"),
                    "🌸 Iris":     ("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv","species"),
                }
                url, hint = urls[sample]
                df_s = pd.read_csv(url)
                st.session_state.data = df_s
                st.session_state.dataset_name = sample
                st.session_state.results = None
                st.session_state.best_model = None
                st.session_state["sample_hint"] = hint
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.session_state.get("sample_hint"):
        st.info(f"💡 Target: **{st.session_state['sample_hint']}**")

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    if st.session_state.data is not None:
        df_sb = st.session_state.data
        null_pct_sb = round(df_sb.isnull().sum().sum() / df_sb.size * 100, 2)
        hs = round(max(0, 100 - null_pct_sb * 2 - df_sb.duplicated().sum() / len(df_sb) * 300), 1)
        hc = ACCENT1 if hs > 80 else ACCENTY if hs > 50 else ACCENTR
        st.markdown(f"""
        <div class="sidebar-section">
          <div class="sidebar-title">📊 Dataset Health</div>
          <div style="font-size:2.2rem;font-weight:900;color:{hc}">{hs}<span style="font-size:1rem;color:{TEXT3}">/100</span></div>
          <div style="font-size:.75rem;color:{TEXT3};margin-top:.3rem">{df_sb.isnull().sum().sum()} nulls · {df_sb.duplicated().sum()} duplicates</div>
          <div style="height:6px;background:{'#1c1c1c' if T=='dark' else '#e0e0e0'};border-radius:3px;margin-top:.75rem;overflow:hidden">
            <div style="height:100%;width:{hs}%;background:linear-gradient(90deg,{hc},{hc}99);border-radius:3px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">⚡ Stack</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-section"><span class="insight-chip">PyCaret</span><span class="insight-chip">Plotly</span><span class="insight-chip">Pandas</span><span class="insight-chip">Streamlit</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def detect_problem_type(s):
    if s.dtype == "object" or str(s.dtype) == "category": return "classification"
    if s.dtype == "bool": return "classification"
    u, n = s.nunique(), len(s)
    if u <= 10 and pd.api.types.is_integer_dtype(s): return "classification"
    if u / n < 0.05 and u <= 20: return "classification"
    return "regression"

def fmt_time(s):
    return f"{int(s//60)}m {int(s%60)}s" if s >= 60 else f"{s:.1f}s"

def chart_layout(**kwargs):
    base = dict(template=CHART_TEMPLATE, paper_bgcolor=CHART_PAPER, plot_bgcolor=CHART_PAPER,
                font=dict(family="Inter", color=CHART_FONT, size=11),
                margin=dict(t=44, b=20, l=20, r=20), title_font=dict(size=13, color=TEXT1))
    base.update(kwargs)
    return base

def plan_gate(feature_key: str, upgrade_msg: str = None):
    if plan_limits.get(feature_key, False):
        return True
    msg = upgrade_msg or f"This feature requires a higher plan. You're on **{current_plan.upper()}**."
    st.markdown(f"""
    <div class="upgrade-wall">
      <div style="font-size:2.5rem;margin-bottom:.5rem">🔒</div>
      <div style="font-size:1rem;font-weight:800;color:{ACCENTY};margin-bottom:.4rem">Feature Locked</div>
      <div style="font-size:.875rem;color:{TEXT2};margin-bottom:1rem">{msg}</div>
    </div>""", unsafe_allow_html=True)
    st.info("👑 Go to the **💳 Upgrade** tab to unlock this feature.")
    return False

# ─────────────────────────────────────────────
#  ADMIN PANEL — PIN Protected
# ─────────────────────────────────────────────
with st.expander("🔐 Admin", expanded=False):
    if not st.session_state.get("admin_unlocked", False):
        # ── PIN Login Screen ──
        st.markdown(f"""
        <div style="text-align:center;padding:1.5rem 1rem">
          <div style="font-size:3rem;margin-bottom:.5rem">🔐</div>
          <div style="font-size:1rem;font-weight:800;color:{TEXT1}">Admin Access</div>
          <div style="font-size:.8rem;color:{TEXT3};margin-top:.25rem">Enter your secret PIN to continue</div>
        </div>""", unsafe_allow_html=True)

        _attempts = st.session_state.get("admin_pin_attempts", 0)

        if _attempts >= 5:
            st.error("🚫 Too many wrong attempts. Restart the app to try again.")
        else:
            _pin_input = st.text_input(
                "Admin PIN", type="password",
                placeholder="Enter PIN...",
                key="admin_pin_field",
                label_visibility="collapsed"
            )
            _pin_col1, _pin_col2 = st.columns([1, 1])
            with _pin_col1:
                if st.button("🔓 Unlock", key="admin_unlock_btn", use_container_width=True):
                    if not ADMIN_PIN:
                        st.error("⚠️ ADMIN_PIN not set in Streamlit secrets! Add it first.")
                    elif _pin_input == ADMIN_PIN:
                        st.session_state["admin_unlocked"] = True
                        st.session_state["admin_pin_attempts"] = 0
                        st.rerun()
                    else:
                        st.session_state["admin_pin_attempts"] = _attempts + 1
                        remaining = 5 - (_attempts + 1)
                        st.error(f"❌ Wrong PIN. {remaining} attempt(s) left.")
            with _pin_col2:
                st.markdown(f'<div style="font-size:.7rem;color:{TEXT3};padding:.6rem 0;text-align:center">Attempts: {_attempts}/5</div>', unsafe_allow_html=True)

    else:
        # ── Unlock Button to Lock Again ──
        _lock_col1, _lock_col2 = st.columns([4, 1])
        with _lock_col2:
            if st.button("🔒 Lock", key="admin_lock_btn"):
                st.session_state["admin_unlocked"] = False
                st.rerun()

        if True:  # always show if unlocked
            all_payments  = load_json(PAYMENTS_FILE)
            all_users_db  = load_json(USERS_FILE)
            all_history   = load_json(HISTORY_FILE)

            pending_pays  = [p for p in all_payments.values() if p.get("status")=="pending"]
            approved_pays = [p for p in all_payments.values() if p.get("status")=="approved"]
            rejected_pays = [p for p in all_payments.values() if p.get("status")=="rejected"]
            total_revenue = sum(p.get("amount",0) for p in approved_pays)

            st.markdown(f'<div style="background:linear-gradient(135deg,rgba(192,132,252,0.10),rgba(96,165,250,0.08));border:1px solid rgba(192,132,252,0.35);border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem"><span style="font-size:2rem">🔐</span><div><div style="font-size:1.1rem;font-weight:900;color:#c084fc">Admin Control Panel</div><div style="font-size:.8rem;color:{TEXT2}">PIN Verified ✓ · Admin Access</div></div></div>', unsafe_allow_html=True)

            from datetime import date as _date
            _today_str = _date.today().isoformat()
            _active_users    = sum(1 for ud in all_users_db.values() if not ud.get("banned") and not ud.get("deactivated"))
            _banned_users    = sum(1 for ud in all_users_db.values() if ud.get("banned"))
            _verified_users  = sum(1 for ud in all_users_db.values() if ud.get("verified"))
            _suspended_users = sum(1 for ud in all_users_db.values() if ud.get("suspended"))
            _new_today       = sum(1 for ud in all_users_db.values() if ud.get("signup_date","")[:10] == _today_str)
            _pro_users       = sum(1 for em in all_users_db if get_user_plan(em) in ["pro","enterprise"])

            sa1,sa2,sa3,sa4,sa5,sa6,sa7,sa8 = st.columns(8)
            for col,lbl,val,sub in [
                (sa1,"👥 Total",len(all_users_db),"users"),
                (sa2,"✅ Active",_active_users,"accounts"),
                (sa3,"🚫 Banned",_banned_users,"users"),
                (sa4,"⏸️ Suspended",_suspended_users,"users"),
                (sa5,"✔️ Verified",_verified_users,"users"),
                (sa6,"⚡ Paid",_pro_users,"pro/ent"),
                (sa7,"💰 Revenue",f"{total_revenue:,.0f}","PKR"),
                (sa8,"🆕 Today",_new_today,"signups"),
            ]:
                with col: st.metric(lbl, val, sub)

            adm1, adm2, adm3, adm4, adm5 = st.tabs([
                f"⏳ Pending ({len(pending_pays)})",
                f"✅ Approved ({len(approved_pays)})",
                f"👥 All Users ({len(all_users_db)})",
                "📊 Activity Monitor",
                "📧 Email Log"
            ])

            with adm1:
                if not pending_pays:
                    st.success("✨ No pending payments — all caught up!")
                for pay in sorted(pending_pays, key=lambda x: x.get("submitted_at",""), reverse=True):
                    plan_c = PRICING.get(pay.get("plan",""),{}).get("color",TEXT2)
                    pid    = pay.get("id","")
                    h  = f'<div style="background:{CARD_BG};border:2px solid rgba(251,191,36,0.35);border-radius:16px;padding:1.25rem;margin-bottom:.75rem;position:relative;overflow:hidden">'
                    h += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#fbbf24,{plan_c})"></div>'
                    h += f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.2rem"><div style="flex:1;min-width:200px">'
                    h += f'<div style="font-size:.95rem;font-weight:800;color:{TEXT1}">{pay.get("name","?")}</div>'
                    h += f'<div style="font-size:.72rem;color:{TEXT3};margin-bottom:.5rem">{pay.get("email","?")}</div>'
                    h += f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">'
                    h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Plan</div><div style="font-weight:800;color:{plan_c}">{pay.get("plan","?").upper()}</div></div>'
                    h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Amount</div><div style="font-weight:800;color:#fbbf24">PKR {pay.get("amount",0):,.0f}</div></div>'
                    h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Method</div><div style="font-weight:600;color:{TEXT1}">{pay.get("payment_method","?").replace("_"," ").title()}</div></div>'
                    h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Billing</div><div style="font-weight:600;color:{TEXT1}">{pay.get("billing","?").title()}</div></div>'
                    h += f'</div></div><div style="flex:1;min-width:180px">'
                    h += f'<div style="background:{BG3};border-radius:10px;padding:.75rem 1rem;margin-bottom:.5rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3};margin-bottom:.2rem">Transaction ID</div><div style="font-size:.9rem;font-weight:700;color:#4ade80;font-family:monospace;word-break:break-all">{pay.get("txn_id","?")}</div></div>'
                    h += f'<div style="font-size:.7rem;color:{TEXT3}">📅 {pay.get("submitted_at","")[:16]}</div>'
                    h += f'<div style="font-size:.65rem;color:{TEXT3};font-family:monospace">{pid}</div>'
                    h += f'</div></div></div>'
                    st.markdown(h, unsafe_allow_html=True)
                    bc1, bc2, bc3 = st.columns([1,1,2])
                    with bc1:
                        if st.button("✅ Approve", key=f"adm_approve_{pid}"):
                            if approve_payment(pid):
                                log_activity(pay.get("email",""), "plan_approved", f"{pay.get('plan','')} activated by admin")
                                st.success(f"✅ {pay.get('name','?')} upgraded to {pay.get('plan','').upper()}!")
                                st.rerun()
                    with bc2:
                        if st.button("❌ Reject", key=f"adm_reject_{pid}"):
                            pays_rj = load_json(PAYMENTS_FILE)
                            if pid in pays_rj:
                                pays_rj[pid]["status"] = "rejected"
                                pays_rj[pid]["processed_at"] = now_str()
                                save_json(PAYMENTS_FILE, pays_rj)
                                log_activity(pay.get("email",""), "plan_rejected", "rejected by admin")
                                st.warning("❌ Payment rejected.")
                                st.rerun()
                    with bc3:
                        note_val = st.text_input("Admin note", placeholder="e.g. Txn verified ✓", key=f"adm_note_{pid}", label_visibility="collapsed")
                        if note_val:
                            pays_n = load_json(PAYMENTS_FILE)
                            if pid in pays_n:
                                pays_n[pid]["admin_note"] = note_val
                                save_json(PAYMENTS_FILE, pays_n)
                    st.markdown("---")

            with adm2:
                if not approved_pays:
                    st.info("No approved payments yet.")
                else:
                    st.metric("Total Revenue", f"PKR {total_revenue:,.0f}", f"{len(approved_pays)} payments")
                    for pay in sorted(approved_pays, key=lambda x: x.get("processed_at",""), reverse=True):
                        plan_c = PRICING.get(pay.get("plan",""),{}).get("color",TEXT2)
                        h  = f'<div style="background:{CARD_BG};border:1px solid rgba(74,222,128,0.25);border-radius:12px;padding:1rem 1.25rem;margin-bottom:.5rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap">'
                        h += f'<span style="font-size:1.3rem">✅</span>'
                        h += f'<div style="flex:1;min-width:150px"><div style="font-weight:700;color:{TEXT1}">{pay.get("name","?")}</div><div style="font-size:.72rem;color:{TEXT3}">{pay.get("email","?")}</div></div>'
                        h += f'<span style="font-size:.7rem;font-weight:800;color:{plan_c};background:{plan_c}22;border:1px solid {plan_c}55;border-radius:6px;padding:.2rem .55rem">{pay.get("plan","?").upper()}</span>'
                        h += f'<div style="font-weight:800;color:#fbbf24;font-family:monospace">PKR {pay.get("amount",0):,.0f}</div>'
                        h += f'<div style="font-size:.7rem;color:{TEXT3}">✓ {pay.get("processed_at","")[:10]}</div>'
                        h += f'<div style="font-size:.65rem;color:{TEXT3};font-family:monospace">{pay.get("id","")}</div></div>'
                        st.markdown(h, unsafe_allow_html=True)

            with adm3:
                # ── Search + Filter Bar ──
                sf1, sf2, sf3 = st.columns([3,1,1])
                with sf1:
                    user_search = st.text_input("🔍 Search name / email", key="adm_user_search", placeholder="Search...")
                with sf2:
                    filter_plan = st.selectbox("Plan", ["All","free","pro","enterprise"], key="adm_filter_plan")
                with sf3:
                    filter_status = st.selectbox("Status", ["All","Active","Banned","Suspended","Verified","Unverified"], key="adm_filter_status")

                for em, ud in sorted(all_users_db.items(), key=lambda x: x[1].get("signup_date",""), reverse=True):
                    # Search filter
                    if user_search and user_search.lower() not in em.lower() and user_search.lower() not in ud.get("name","").lower():
                        continue
                    u_plan = get_user_plan(em)
                    # Plan filter
                    if filter_plan != "All" and u_plan != filter_plan:
                        continue
                    # Status filter
                    is_banned     = ud.get("banned", False)
                    is_suspended  = ud.get("suspended", False)
                    is_deactivated= ud.get("deactivated", False)
                    is_verified   = ud.get("verified", False)
                    id_required   = ud.get("id_verification_required", False)
                    if filter_status == "Banned"     and not is_banned:     continue
                    if filter_status == "Suspended"  and not is_suspended:  continue
                    if filter_status == "Active"     and (is_banned or is_suspended or is_deactivated): continue
                    if filter_status == "Verified"   and not is_verified:   continue
                    if filter_status == "Unverified" and is_verified:       continue

                    u_pc   = PLAN_COLORS.get(u_plan,"#6b7280")
                    u_pi   = PLAN_ICONS.get(u_plan,"🌱")
                    u_hist = all_history.get(em,{})
                    login_count  = u_hist.get("login_count",0)
                    last_login   = ud.get("last_login") or u_hist.get("last_login","—")
                    join_date    = ud.get("signup_date","—")[:10]
                    admin_note   = ud.get("admin_note","")
                    suspend_until= ud.get("suspended_until","")
                    user_payments_list = [p for p in all_payments.values() if p.get("email")==em]
                    online_st    = get_online_status(ud)

                    # Status badges
                    status_badges = ""
                    if is_banned:      status_badges += f'<span style="background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.4);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">🚫 BANNED</span>'
                    if is_suspended:   status_badges += f'<span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.4);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">⏸️ SUSPENDED</span>'
                    if is_deactivated: status_badges += f'<span style="background:rgba(107,114,128,0.15);color:#9ca3af;border:1px solid rgba(107,114,128,0.4);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">🔴 INACTIVE</span>'
                    if is_verified:    status_badges += f'<span style="background:rgba(74,222,128,0.12);color:#4ade80;border:1px solid rgba(74,222,128,0.35);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">✔️ VERIFIED</span>'
                    if id_required:    status_badges += f'<span style="background:rgba(192,132,252,0.12);color:#c084fc;border:1px solid rgba(192,132,252,0.35);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">🪪 ID NEEDED</span>'
                    if ud.get("frozen"):   status_badges += f'<span style="background:rgba(96,165,250,0.15);color:#60a5fa;border:1px solid rgba(96,165,250,0.4);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">🧊 FROZEN</span>'
                    if ud.get("readonly"): status_badges += f'<span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.4);border-radius:6px;padding:.15rem .5rem;font-size:.65rem;font-weight:800;margin-left:.3rem">👁️ READ-ONLY</span>'

                    row_border = "#f87171" if is_banned else ("#60a5fa" if ud.get("frozen") else ("#fbbf24" if is_suspended else BORDER))

                    h  = f'<div style="background:{CARD_BG};border:1px solid {row_border};border-radius:14px;padding:1rem 1.25rem;margin-bottom:.5rem;position:relative;overflow:hidden">'
                    h += f'<div style="position:absolute;top:0;left:0;bottom:0;width:3px;background:{u_pc}"></div>'
                    h += f'<div style="display:flex;align-items:center;gap:.9rem;flex-wrap:wrap;margin-left:.3rem">'
                    h += f'<div style="position:relative;width:38px;height:38px;flex-shrink:0">'
                    h += f'<div style="width:38px;height:38px;border-radius:50%;background:{u_pc}20;border:2px solid {u_pc}60;display:flex;align-items:center;justify-content:center;font-size:1.1rem">{u_pi}</div>'
                    h += f'<div style="position:absolute;bottom:0;right:0;width:11px;height:11px;border-radius:50%;background:{online_st["color"]};border:2px solid {CARD_BG}"></div>'
                    h += f'</div>'
                    h += f'<div style="flex:1;min-width:180px">'
                    h += f'<div style="font-size:.92rem;font-weight:800;color:{TEXT1}">{ud.get("name","?")} {status_badges}</div>'
                    h += f'<div style="font-size:.72rem;color:{TEXT3};margin-top:.1rem">{em}</div>'
                    h += f'</div>'
                    h += f'<div style="display:flex;gap:1.25rem;flex-wrap:wrap;align-items:center">'
                    h += f'<div style="text-align:center"><div style="font-size:.85rem;font-weight:900;color:{u_pc}">{u_plan.upper()}</div><div style="font-size:.58rem;color:{TEXT3}">plan</div></div>'
                    h += f'<div style="text-align:center"><div style="font-size:.85rem;font-weight:900;color:#60a5fa">{login_count}</div><div style="font-size:.58rem;color:{TEXT3}">logins</div></div>'
                    h += f'<div style="text-align:center"><div style="font-size:.85rem;font-weight:900;color:{ACCENTY}">{len(user_payments_list)}</div><div style="font-size:.58rem;color:{TEXT3}">payments</div></div>'
                    h += f'<div style="text-align:center"><div style="font-size:.82rem;font-weight:800;color:{online_st["color"]}">{online_st["dot"]} {online_st["label"]}</div><div style="font-size:.58rem;color:{TEXT3}">last seen</div></div>'
                    h += f'<div style="text-align:right"><div style="font-size:.68rem;color:{TEXT3}">📅 Joined: {join_date}</div>'
                    h += f'<div style="font-size:.68rem;color:{TEXT3}">🕐 Last login: {str(last_login)[:16]}</div>'
                    h += f'</div></div></div>'
                    st.markdown(h, unsafe_allow_html=True)

                    with st.expander(f"⚙️ Manage — {ud.get('name','?')} ({em})", expanded=False):

                        # ── TABS inside manage panel ──
                        mt1, mt2, mt3, mt4, mt5, mt6 = st.tabs(["📋 Profile", "🔐 Account", "🧊 Freeze/Delete", "📧 Message", "📜 Activity", "💳 Payments"])

                        # ── TAB 1: Profile + Plan ──
                        with mt1:
                            p1c1, p1c2, p1c3 = st.columns(3)
                            with p1c1:
                                np_ = st.selectbox("Set Plan", ["free","pro","enterprise"],
                                    index=["free","pro","enterprise"].index(u_plan), key=f"adm_plan_{em}")
                                nm_ = st.number_input("Duration (months)", 1, 24, 1, key=f"adm_months_{em}")
                                if st.button("💾 Apply Plan", key=f"adm_apply_{em}"):
                                    upgrade_user_plan(em, np_, int(nm_))
                                    log_activity(em, "plan_changed_by_admin", f"Set to {np_} for {nm_}mo")
                                    st.success(f"✅ Plan updated!"); st.rerun()
                            with p1c2:
                                st.markdown(f'<div style="font-size:.7rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">User Info</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2}">📧 <b>Email:</b> {em}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2}">📅 <b>Joined:</b> {join_date}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2}">🕐 <b>Last Login:</b> {str(last_login)[:16]}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2}">🔢 <b>Total Logins:</b> {login_count}</div>', unsafe_allow_html=True)
                                plain_pw = ud.get("password_plain", "—")
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2}">🔑 <b>Password:</b> <span style="color:#4ade80;font-family:monospace">{plain_pw}</span></div>', unsafe_allow_html=True)
                                expiry = ud.get("plan_expiry","—")
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2}">⏳ <b>Plan Expiry:</b> {expiry}</div>', unsafe_allow_html=True)
                            with p1c3:
                                new_note = st.text_area("📝 Admin Note", value=admin_note, key=f"adm_note_profile_{em}", height=100)
                                if st.button("💾 Save Note", key=f"adm_save_note_{em}"):
                                    udb = load_json(USERS_FILE)
                                    udb[em]["admin_note"] = new_note
                                    save_json(USERS_FILE, udb)
                                    st.success("Note saved!"); st.rerun()

                        # ── TAB 2: Account Controls ──
                        with mt2:
                            ac_c1, ac_c2, ac_c3 = st.columns(3)

                            with ac_c1:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">🚫 Ban / Unban</div>', unsafe_allow_html=True)
                                if is_banned:
                                    if st.button("✅ Unban User", key=f"adm_unban_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["banned"] = False
                                        udb[em]["banned_at"] = None
                                        udb[em]["ban_reason"] = ""
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "unbanned", "Unbanned by admin")
                                        st.success("✅ User unbanned!"); st.rerun()
                                else:
                                    ban_reason = st.text_input("Ban reason", key=f"adm_ban_reason_{em}", placeholder="e.g. Spam, abuse...")
                                    if st.button("🚫 Ban User", key=f"adm_ban_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["banned"] = True
                                        udb[em]["banned_at"] = now_str()
                                        udb[em]["ban_reason"] = ban_reason
                                        save_json(USERS_FILE, udb)
                                        # Invalidate all tokens
                                        tks = load_json(TOKENS_FILE)
                                        tks = {t:v for t,v in tks.items() if v.get("email") != em}
                                        save_json(TOKENS_FILE, tks)
                                        log_activity(em, "banned", f"Banned: {ban_reason}")
                                        st.error(f"🚫 User banned!"); st.rerun()

                            with ac_c2:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">⏸️ Suspend</div>', unsafe_allow_html=True)
                                if is_suspended:
                                    st.markdown(f'<div style="font-size:.75rem;color:{ACCENTY}">Suspended until: {suspend_until}</div>', unsafe_allow_html=True)
                                    if st.button("▶️ Lift Suspension", key=f"adm_unsuspend_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["suspended"] = False
                                        udb[em]["suspended_until"] = ""
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "unsuspended", "Suspension lifted by admin")
                                        st.success("✅ Suspension lifted!"); st.rerun()
                                else:
                                    sus_days = st.number_input("Suspend for (days)", 1, 365, 7, key=f"adm_sus_days_{em}")
                                    sus_reason = st.text_input("Reason", key=f"adm_sus_reason_{em}", placeholder="e.g. Policy violation")
                                    if st.button("⏸️ Suspend", key=f"adm_suspend_{em}"):
                                        until_dt = (datetime.now() + timedelta(days=int(sus_days))).strftime("%Y-%m-%d")
                                        udb = load_json(USERS_FILE)
                                        udb[em]["suspended"] = True
                                        udb[em]["suspended_until"] = until_dt
                                        udb[em]["suspend_reason"] = sus_reason
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "suspended", f"{sus_days}d: {sus_reason}")
                                        st.warning(f"⏸️ Suspended for {sus_days} days!"); st.rerun()

                            with ac_c3:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">🔧 Other Controls</div>', unsafe_allow_html=True)

                                # Activate / Deactivate
                                if is_deactivated:
                                    if st.button("🟢 Activate Account", key=f"adm_activate_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["deactivated"] = False
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "activated", "Reactivated by admin")
                                        st.success("✅ Activated!"); st.rerun()
                                else:
                                    if st.button("🔴 Deactivate Account", key=f"adm_deactivate_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["deactivated"] = True
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "deactivated", "Deactivated by admin")
                                        st.warning("🔴 Account deactivated!"); st.rerun()

                                # Verify / Unverify
                                if is_verified:
                                    if st.button("❌ Remove Verification", key=f"adm_unverify_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["verified"] = False
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "unverified", "Verification removed")
                                        st.warning("Verification removed!"); st.rerun()
                                else:
                                    if st.button("✔️ Mark as Verified", key=f"adm_verify_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["verified"] = True
                                        udb[em]["verified_at"] = now_str()
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "verified", "Verified by admin")
                                        st.success("✔️ User verified!"); st.rerun()

                                # ID Verification required
                                if id_required:
                                    if st.button("🔓 Remove ID Requirement", key=f"adm_no_id_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["id_verification_required"] = False
                                        save_json(USERS_FILE, udb)
                                        st.success("ID requirement removed!"); st.rerun()
                                else:
                                    if st.button("🪪 Require ID Verification", key=f"adm_req_id_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["id_verification_required"] = True
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "id_required", "ID verification required by admin")
                                        st.warning("🪪 ID verification required!"); st.rerun()

                                # Reset Password
                                st.markdown("---")
                                new_pw = st.text_input("🔑 New Password", type="password", key=f"adm_new_pw_{em}", placeholder="Leave blank to skip")
                                if st.button("🔑 Reset Password", key=f"adm_reset_pw_{em}"):
                                    if new_pw and len(new_pw) >= 6:
                                        udb = load_json(USERS_FILE)
                                        udb[em]["password_hash"] = hash_password(new_pw)
                                        udb[em]["password_plain"] = new_pw
                                        save_json(USERS_FILE, udb)
                                        # Invalidate all tokens so user must re-login
                                        tks = load_json(TOKENS_FILE)
                                        tks = {t:v for t,v in tks.items() if v.get("email") != em}
                                        save_json(TOKENS_FILE, tks)
                                        log_activity(em, "password_reset", "Password reset by admin")
                                        st.success("✅ Password reset! User logged out."); st.rerun()
                                    else:
                                        st.error("Min 6 characters required.")

                                # Moderator role
                                st.markdown("---")
                                current_role = ud.get("role", "user")
                                if current_role == "moderator":
                                    st.markdown(f'<div style="font-size:.75rem;color:#60a5fa;margin-bottom:.4rem">🛡️ Currently a Moderator</div>', unsafe_allow_html=True)
                                    if st.button("👤 Remove Moderator", key=f"adm_remove_mod_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["role"] = "user"
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "moderator_removed", "Moderator role removed by admin")
                                        st.success("Role removed!"); st.rerun()
                                else:
                                    if st.button("🛡️ Make Moderator", key=f"adm_make_mod_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["role"] = "moderator"
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "moderator_assigned", "Moderator role assigned by admin")
                                        st.success("🛡️ Moderator assigned!"); st.rerun()

                        # ── TAB 3: Freeze / Read-Only / Delete ──
                        with mt3:
                            is_frozen   = ud.get("frozen", False)
                            is_readonly_u = ud.get("readonly", False)

                            fr_c1, fr_c2, fr_c3 = st.columns(3)

                            with fr_c1:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:#60a5fa;margin-bottom:.5rem">🧊 Freeze Account</div>', unsafe_allow_html=True)
                                st.caption("Instant block — user sees frozen screen, cannot do anything.")
                                if is_frozen:
                                    st.markdown(f'<div style="background:rgba(96,165,250,0.10);border:1px solid rgba(96,165,250,0.35);border-radius:8px;padding:.6rem .9rem;font-size:.75rem;color:#60a5fa;margin-bottom:.5rem">🧊 Frozen since: {ud.get("frozen_at","—")[:16]}<br>Reason: {ud.get("frozen_reason","—")}</div>', unsafe_allow_html=True)
                                    if st.button("🔓 Unfreeze Account", key=f"adm_unfreeze_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["frozen"] = False
                                        udb[em]["frozen_at"] = ""
                                        udb[em]["frozen_reason"] = ""
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "unfrozen", "Account unfrozen by admin")
                                        st.success("✅ Account unfrozen!"); st.rerun()
                                else:
                                    freeze_reason = st.text_input("Freeze reason", key=f"adm_freeze_reason_{em}", placeholder="e.g. Suspicious activity")
                                    if st.button("🧊 Freeze Instantly", key=f"adm_freeze_{em}", type="primary"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["frozen"] = True
                                        udb[em]["frozen_at"] = now_str()
                                        udb[em]["frozen_reason"] = freeze_reason
                                        save_json(USERS_FILE, udb)
                                        # Invalidate all tokens
                                        tks = load_json(TOKENS_FILE)
                                        tks = {t:v for t,v in tks.items() if v.get("email") != em}
                                        save_json(TOKENS_FILE, tks)
                                        log_activity(em, "frozen", f"Frozen: {freeze_reason}")
                                        st.info("🧊 Account frozen instantly!"); st.rerun()

                            with fr_c2:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{ACCENTY};margin-bottom:.5rem">👁️ Read-Only Mode</div>', unsafe_allow_html=True)
                                st.caption("User can login and view but cannot train, upload, or take any action.")
                                if is_readonly_u:
                                    st.markdown(f'<div style="background:rgba(251,191,36,0.10);border:1px solid rgba(251,191,36,0.35);border-radius:8px;padding:.6rem .9rem;font-size:.75rem;color:{ACCENTY};margin-bottom:.5rem">👁️ Currently in Read-Only mode</div>', unsafe_allow_html=True)
                                    if st.button("✅ Restore Full Access", key=f"adm_readonly_off_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["readonly"] = False
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "readonly_removed", "Full access restored by admin")
                                        st.success("✅ Full access restored!"); st.rerun()
                                else:
                                    ro_reason = st.text_input("Reason (optional)", key=f"adm_ro_reason_{em}", placeholder="e.g. Payment pending")
                                    if st.button("👁️ Set Read-Only", key=f"adm_readonly_on_{em}"):
                                        udb = load_json(USERS_FILE)
                                        udb[em]["readonly"] = True
                                        udb[em]["readonly_reason"] = ro_reason
                                        udb[em]["readonly_at"] = now_str()
                                        save_json(USERS_FILE, udb)
                                        log_activity(em, "readonly_set", f"Read-only: {ro_reason}")
                                        st.warning("👁️ User set to read-only!"); st.rerun()

                            with fr_c3:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{ACCENTR};margin-bottom:.5rem">🗑️ Delete Account</div>', unsafe_allow_html=True)
                                st.caption("⚠️ Permanent! All data, history, and payments will be deleted forever.")
                                confirm_del = st.text_input("Type email to confirm", key=f"adm_del_confirm_{em}", placeholder=f"{em}")
                                if st.button("🗑️ DELETE PERMANENTLY", key=f"adm_delete_{em}"):
                                    if confirm_del.strip() == em:
                                        # Delete from all collections
                                        udb = load_json(USERS_FILE)
                                        udb.pop(em, None)
                                        save_json(USERS_FILE, udb)

                                        hdb = load_json(HISTORY_FILE)
                                        hdb.pop(em, None)
                                        save_json(HISTORY_FILE, hdb)

                                        # Delete tokens
                                        tks = load_json(TOKENS_FILE)
                                        tks = {t:v for t,v in tks.items() if v.get("email") != em}
                                        save_json(TOKENS_FILE, tks)

                                        # Delete payments
                                        pays = load_json(PAYMENTS_FILE)
                                        pays = {k:v for k,v in pays.items() if v.get("email") != em}
                                        save_json(PAYMENTS_FILE, pays)

                                        st.error(f"🗑️ Account {em} permanently deleted!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Email doesn't match — not deleted.")

                        # ── TAB 4: Send Email ──
                        with mt4:
                            st.markdown(f'<div style="font-size:.8rem;color:{TEXT2};margin-bottom:.75rem">Send a direct message to <b>{ud.get("name","?")}</b> at <b>{em}</b></div>', unsafe_allow_html=True)
                            msg_subject = st.text_input("Subject", key=f"adm_msg_subj_{em}", placeholder="e.g. Account Notice")
                            msg_body    = st.text_area("Message", key=f"adm_msg_body_{em}", height=150, placeholder="Type your message here...")
                            if st.button("📧 Send Email", key=f"adm_send_msg_{em}"):
                                if msg_subject and msg_body:
                                    # Temporarily override NOTIFY_TO to send to this user
                                    try:
                                        msg = MIMEMultipart("alternative")
                                        msg["Subject"] = f"[DataForge] {msg_subject}"
                                        msg["From"]    = SMTP_USER
                                        msg["To"]      = em
                                        html_body = f"""<html><body style="font-family:Inter,sans-serif;background:#0a0a0a;color:#e5e7eb;padding:24px">
                                        <div style="max-width:520px;margin:auto;background:#111;border:1px solid #222;border-radius:16px;padding:28px">
                                        <div style="font-size:1.5rem;font-weight:900;color:#4ade80;margin-bottom:1rem">⚡ DataForge ML Studio</div>
                                        <div style="font-size:1rem;font-weight:700;color:#f9fafb;margin-bottom:.75rem">Hi {ud.get("name","there")},</div>
                                        <div style="font-size:.9rem;color:#d1fae5;line-height:1.7;white-space:pre-wrap">{msg_body}</div>
                                        <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid #222;font-size:.75rem;color:#6b7280">DataForge ML Studio · This is an official communication from admin.</div>
                                        </div></body></html>"""
                                        msg.attach(MIMEText(msg_body, "plain", "utf-8"))
                                        msg.attach(MIMEText(html_body, "html", "utf-8"))
                                        sent = False
                                        for port, use_ssl in [(465, True), (587, False)]:
                                            try:
                                                if use_ssl:
                                                    with smtplib.SMTP_SSL("smtp.gmail.com", port, timeout=10) as srv:
                                                        srv.login(SMTP_USER, SMTP_PASS)
                                                        srv.sendmail(SMTP_USER, em, msg.as_string())
                                                else:
                                                    with smtplib.SMTP("smtp.gmail.com", port, timeout=10) as srv:
                                                        srv.ehlo(); srv.starttls(); srv.ehlo()
                                                        srv.login(SMTP_USER, SMTP_PASS)
                                                        srv.sendmail(SMTP_USER, em, msg.as_string())
                                                sent = True; break
                                            except: continue
                                        if sent:
                                            log_activity(em, "admin_email_sent", msg_subject)
                                            st.success(f"📧 Email sent to {em}!")
                                        else:
                                            st.error("❌ Email failed — check SMTP credentials.")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                                else:
                                    st.error("Subject aur message dono required hain.")

                        # ── TAB 5: Login + Activity History ──
                        with mt5:
                            activity_log  = u_hist.get("activity_log", [])
                            training_log  = u_hist.get("training_log", [])

                            h4c1, h4c2 = st.columns(2)
                            with h4c1:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">📜 Login History (last 10)</div>', unsafe_allow_html=True)
                                login_events = [a for a in activity_log if a.get("action") in ["signin","signup"]][-10:][::-1]
                                if not login_events:
                                    st.caption("No login history.")
                                for ev in login_events:
                                    st.markdown(f'<div style="font-size:.75rem;color:{TEXT2};padding:.3rem 0;border-bottom:1px solid {BORDER}">🔑 {ev.get("time","")[:16]} — {ev.get("detail","login")}</div>', unsafe_allow_html=True)

                            with h4c2:
                                st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">🤖 Training History (last 10)</div>', unsafe_allow_html=True)
                                if not training_log:
                                    st.caption("No training history.")
                                for tr in training_log[-10:][::-1]:
                                    st.markdown(f'<div style="font-size:.75rem;color:{TEXT2};padding:.3rem 0;border-bottom:1px solid {BORDER}">{"🎯" if tr.get("problem_type")=="classification" else "📈"} {tr.get("time","")[:10]} — {tr.get("best_model","?")} · {tr.get("score",0):.4f}</div>', unsafe_allow_html=True)

                            st.markdown(f'<div style="font-size:.72rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin:.75rem 0 .5rem">📋 All Activity (last 20)</div>', unsafe_allow_html=True)
                            for ev in activity_log[-20:][::-1]:
                                action_color = "#4ade80" if "approved" in ev.get("action","") else "#f87171" if "banned" in ev.get("action","") else TEXT2
                                st.markdown(f'<div style="font-size:.73rem;color:{action_color};padding:.25rem 0;border-bottom:1px solid {BORDER}">{ev.get("time","")[:16]} · <b>{ev.get("action","")}</b> — {ev.get("detail","")}</div>', unsafe_allow_html=True)

                            if st.button("🗑️ Clear Training History", key=f"adm_clear_hist_{em}"):
                                hdb = load_json(HISTORY_FILE)
                                if em in hdb:
                                    hdb[em]["training_log"] = []
                                    save_json(HISTORY_FILE, hdb)
                                    log_activity(em, "history_cleared", "Training history cleared by admin")
                                    st.success("Training history cleared!"); st.rerun()

                        # ── TAB 6: Payment Attempts ──
                        with mt6:
                            all_user_pays = sorted(user_payments_list, key=lambda x: x.get("submitted_at",""), reverse=True)
                            if not all_user_pays:
                                st.info("No payment attempts from this user.")
                            else:
                                st.markdown(f'<div style="font-size:.8rem;color:{TEXT2};margin-bottom:.75rem"><b>{len(all_user_pays)}</b> payment attempt(s) found</div>', unsafe_allow_html=True)
                                for pay in all_user_pays:
                                    status = pay.get("status","pending")
                                    sc = "#fbbf24" if status=="pending" else "#4ade80" if status=="approved" else "#f87171"
                                    plan_c2 = PRICING.get(pay.get("plan",""),{}).get("color",TEXT2)
                                    h5  = f'<div style="background:{BG3};border:1px solid {sc}44;border-radius:10px;padding:.75rem 1rem;margin-bottom:.5rem">'
                                    h5 += f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem">'
                                    h5 += f'<div><span style="font-size:.7rem;font-weight:800;color:{plan_c2};background:{plan_c2}22;border:1px solid {plan_c2}44;border-radius:5px;padding:.1rem .45rem">{pay.get("plan","?").upper()}</span> <span style="font-size:.72rem;color:{TEXT2}">PKR {pay.get("amount",0):,.0f} · {pay.get("payment_method","?").replace("_"," ").title()}</span></div>'
                                    h5 += f'<span style="font-size:.7rem;font-weight:800;color:{sc}">{status.upper()}</span>'
                                    h5 += f'</div>'
                                    h5 += f'<div style="font-size:.68rem;color:{TEXT3};margin-top:.3rem">Txn: {pay.get("txn_id","?")} · {pay.get("submitted_at","")[:16]} · {pay.get("id","")}</div>'
                                    h5 += f'</div>'
                                    st.markdown(h5, unsafe_allow_html=True)

            with adm4:
                st.markdown(f'<div style="font-size:1rem;font-weight:800;color:{TEXT1};margin-bottom:1rem">📊 Live Activity Monitor</div>', unsafe_allow_html=True)

                # ── Summary Stats ──
                all_activity = []
                for em_a, hist_a in all_history.items():
                    for ev in hist_a.get("activity_log", []):
                        all_activity.append({**ev, "email": em_a, "name": all_users_db.get(em_a, {}).get("name","?")})
                all_activity = sorted(all_activity, key=lambda x: x.get("time",""), reverse=True)

                # Stats
                _today_acts = [a for a in all_activity if a.get("time","")[:10] == _today_str]
                _logins_today = sum(1 for a in _today_acts if a.get("action") == "signin")
                _signups_today = sum(1 for a in _today_acts if a.get("action") == "signup")
                _trainings_today = sum(1 for a in _today_acts if a.get("action") == "training_complete")
                _frozen_count = sum(1 for ud in all_users_db.values() if ud.get("frozen"))
                _readonly_count = sum(1 for ud in all_users_db.values() if ud.get("readonly"))
                _online_count = sum(1 for ud in all_users_db.values() if get_online_status(ud)["online"])

                am1,am2,am3,am4,am5,am6,am7 = st.columns(7)
                for col,lbl,val,color in [
                    (am1,"🟢 Online Now",_online_count,"#4ade80"),
                    (am2,"🔑 Logins Today",_logins_today,"#60a5fa"),
                    (am3,"✨ Signups Today",_signups_today,"#4ade80"),
                    (am4,"🤖 Trainings Today",_trainings_today,"#c084fc"),
                    (am5,"🧊 Frozen",_frozen_count,"#60a5fa"),
                    (am6,"👁️ Read-Only",_readonly_count,"#fbbf24"),
                    (am7,"📋 Total Events",len(all_activity),"#9ca3af"),
                ]:
                    with col:
                        st.markdown(f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:.75rem 1rem;text-align:center"><div style="font-size:.65rem;color:{TEXT3};text-transform:uppercase;font-weight:700">{lbl}</div><div style="font-size:1.6rem;font-weight:900;color:{color}">{val}</div></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

                # ── Frozen/ReadOnly users quick list ──
                frozen_users = [(em, ud) for em, ud in all_users_db.items() if ud.get("frozen") or ud.get("readonly")]
                if frozen_users:
                    st.markdown(f'<div style="font-size:.75rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin-bottom:.5rem">⚠️ Restricted Accounts</div>', unsafe_allow_html=True)
                    for em_f, ud_f in frozen_users:
                        status_f = "🧊 FROZEN" if ud_f.get("frozen") else "👁️ READ-ONLY"
                        color_f  = "#60a5fa" if ud_f.get("frozen") else "#fbbf24"
                        st.markdown(f'<div style="background:{CARD_BG};border:1px solid {color_f}44;border-radius:10px;padding:.6rem 1rem;margin-bottom:.3rem;display:flex;justify-content:space-between;align-items:center"><div><b style="color:{TEXT1}">{ud_f.get("name","?")} </b><span style="font-size:.72rem;color:{TEXT3}">{em_f}</span></div><span style="font-size:.72rem;font-weight:800;color:{color_f}">{status_f}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

                # ── Live Activity Feed ──
                am_search = st.text_input("🔍 Filter by email or action", key="am_search", placeholder="e.g. training, signin...")
                st.markdown(f'<div style="font-size:.75rem;font-weight:800;text-transform:uppercase;color:{TEXT3};margin:.5rem 0">📋 Recent Activity (last 100)</div>', unsafe_allow_html=True)

                shown = 0
                for ev in all_activity[:200]:
                    if am_search and am_search.lower() not in ev.get("email","").lower() and am_search.lower() not in ev.get("action","").lower():
                        continue
                    action = ev.get("action","")
                    a_color = "#4ade80" if action in ["signup","plan_approved","verified"] else \
                              "#f87171" if action in ["banned","frozen","deleted","plan_rejected"] else \
                              "#fbbf24" if action in ["suspended","readonly_set"] else \
                              "#60a5fa" if action == "signin" else \
                              "#c084fc" if action == "training_complete" else TEXT3
                    st.markdown(f'<div style="display:flex;gap:.75rem;align-items:center;padding:.3rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.68rem;color:{TEXT3};min-width:110px">{ev.get("time","")[:16]}</span><span style="font-size:.72rem;font-weight:700;color:{a_color};min-width:130px">{action}</span><span style="font-size:.72rem;color:{TEXT2};min-width:160px">{ev.get("name","?")} ({ev.get("email","")[:20]})</span><span style="font-size:.7rem;color:{TEXT3}">{ev.get("detail","")[:50]}</span></div>', unsafe_allow_html=True)
                    shown += 1
                    if shown >= 100: break

            with adm5:
                email_log = load_json("dataforge_email_log")
                if not email_log:
                    st.info("No email events logged yet.")
                else:
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        st.download_button("📥 Export CSV",
                            pd.DataFrame([{"ts":ts,**{k:str(v)[:80] for k,v in e.items()}} for ts,e in sorted(email_log.items(),reverse=True)[:50]]).to_csv(index=False),
                            "email_log.csv","text/csv", key="adm_dl_email")
                    with ec2:
                        if st.button("🗑️ Clear Log", key="adm_clear_email"):
                            save_json("dataforge_email_log",{})
                            st.success("Cleared!"); st.rerun()

                    for ts, entry in sorted(email_log.items(), reverse=True)[:50]:
                        has_err = "error" in entry
                        is_ok   = entry.get("status") == "sent_ok"
                        icon    = "✅" if is_ok else ("❌" if has_err else "📧")
                        col     = "#4ade80" if is_ok else ("#f87171" if has_err else "#fbbf24")
                        bdr     = "rgba(248,113,113,0.30)" if has_err else BORDER
                        h  = f'<div style="background:{CARD_BG};border:1px solid {bdr};border-radius:10px;padding:.75rem 1rem;margin-bottom:.35rem;display:flex;gap:.75rem;align-items:flex-start">'
                        h += f'<span style="font-size:1rem;flex-shrink:0">{icon}</span>'
                        h += f'<div style="flex:1;min-width:0"><div style="font-size:.8rem;font-weight:700;color:{col};white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{entry.get("subject","(no subject)")}</div>'
                        h += f'<div style="font-size:.68rem;color:{TEXT3};margin-top:.1rem">{ts}</div>'
                        if entry.get("body_preview"):
                            preview = entry["body_preview"].replace("\n","<br>").strip()
                            h += f'<div style="font-size:.72rem;color:{TEXT2};margin-top:.3rem;background:rgba(255,255,255,0.04);border-radius:6px;padding:.35rem .6rem;font-family:monospace;white-space:pre-wrap">{preview}</div>'
                        if has_err:
                            h += f'<div style="font-size:.72rem;color:#f87171;margin-top:.25rem;background:rgba(248,113,113,0.07);border-radius:6px;padding:.25rem .5rem">Error: {entry.get("error","")}</div>'
                        if entry.get("note"):
                            h += f'<div style="font-size:.7rem;color:#fbbf24;margin-top:.2rem;background:rgba(251,191,36,0.07);border-radius:6px;padding:.25rem .5rem">{entry["note"][:200]}</div>'
                        h += '</div></div>'
                        st.markdown(h, unsafe_allow_html=True)

st.markdown("---")

if is_moderator:
    with st.expander("🛡️ Moderator Panel", expanded=False):
        all_payments_mod = load_json(PAYMENTS_FILE)
        all_users_mod    = load_json(USERS_FILE)
        pending_mod      = [p for p in all_payments_mod.values() if p.get("status") == "pending"]

        st.markdown(f'<div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.35);border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem"><span style="font-size:2rem">🛡️</span><div><div style="font-size:1.1rem;font-weight:900;color:#60a5fa">Moderator Panel</div><div style="font-size:.8rem;color:{TEXT2}">{uname_global} · Moderator Access</div></div></div>', unsafe_allow_html=True)

        mod_tab1, mod_tab2 = st.tabs([
            f"⏳ Pending Payments ({len(pending_mod)})",
            f"👥 All Users ({len(all_users_mod)})"
        ])

        with mod_tab1:
            if not pending_mod:
                st.success("✨ No pending payments!")
            for pay in sorted(pending_mod, key=lambda x: x.get("submitted_at",""), reverse=True):
                pid = pay.get("id","")
                plan_c = PRICING.get(pay.get("plan",""),{}).get("color", TEXT2)
                st.markdown(f"""
                <div style="background:{CARD_BG};border:2px solid rgba(251,191,36,0.35);border-radius:16px;padding:1.25rem;margin-bottom:.75rem">
                  <div style="font-weight:800;color:{TEXT1}">{pay.get("name","?")} — {pay.get("email","?")}</div>
                  <div style="font-size:.8rem;color:{TEXT2};margin:.3rem 0">
                    Plan: <b style="color:{plan_c}">{pay.get("plan","?").upper()}</b> ·
                    Amount: <b style="color:#fbbf24">PKR {pay.get("amount",0):,.0f}</b> ·
                    Method: {pay.get("payment_method","?").replace("_"," ").title()}
                  </div>
                  <div style="font-size:.72rem;color:{TEXT3}">Txn: {pay.get("txn_id","?")} · {pay.get("submitted_at","")[:16]} · {pid}</div>
                </div>""", unsafe_allow_html=True)
                mc1, mc2 = st.columns(2)
                with mc1:
                    if st.button("✅ Approve", key=f"mod_approve_{pid}"):
                        if approve_payment(pid):
                            log_activity(pay.get("email",""), "plan_approved", f"{pay.get('plan','')} approved by moderator")
                            st.success(f"✅ Approved!"); st.rerun()
                with mc2:
                    if st.button("❌ Reject", key=f"mod_reject_{pid}"):
                        pays_rj = load_json(PAYMENTS_FILE)
                        if pid in pays_rj:
                            pays_rj[pid]["status"] = "rejected"
                            pays_rj[pid]["processed_at"] = now_str()
                            save_json(PAYMENTS_FILE, pays_rj)
                            log_activity(pay.get("email",""), "plan_rejected", "rejected by moderator")
                            st.warning("❌ Rejected."); st.rerun()
                st.markdown("---")

        with mod_tab2:
            mod_search = st.text_input("🔍 Search", key="mod_search", placeholder="Name or email...")
            for em, ud in sorted(all_users_mod.items(), key=lambda x: x[1].get("signup_date",""), reverse=True):
                if mod_search and mod_search.lower() not in em.lower() and mod_search.lower() not in ud.get("name","").lower():
                    continue
                u_plan = get_user_plan(em)
                u_pc   = PLAN_COLORS.get(u_plan, "#6b7280")
                online_st = get_online_status(ud)
                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:.9rem 1.25rem;margin-bottom:.4rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
                  <div style="flex:1;min-width:180px">
                    <div style="font-weight:700;color:{TEXT1}">{ud.get("name","?")}</div>
                    <div style="font-size:.72rem;color:{TEXT3}">{em}</div>
                  </div>
                  <span style="font-size:.75rem;font-weight:800;color:{u_pc}">{u_plan.upper()}</span>
                  <span style="font-size:.75rem;color:{online_st['color']}">{online_st['dot']} {online_st['label']}</span>
                  <div style="font-size:.68rem;color:{TEXT3}">Joined: {ud.get("signup_date","—")[:10]}</div>
                </div>""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
if st.session_state.data is not None:
    df = st.session_state.data
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = df.duplicated().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊  Data Explorer", "🧬  EDA & Insights", "⚙️  Train Model",
        "🏆  Results", "📜  My History", "💳  Upgrade"
    ])

    # ═══════════════════════════
    # TAB 1 — DATA EXPLORER
    # ═══════════════════════════
    with tab1:
        st.markdown(f"""
        <div class="stat-grid slide-up">
          <div class="stat-card {'good' if null_pct<=5 else 'warn' if null_pct<=15 else 'danger'}">
            <div class="bar"></div><div class="label">Total Rows</div>
            <div class="value">{len(df):,}</div><div class="sub">records</div>
          </div>
          <div class="stat-card">
            <div class="bar"></div><div class="label">Columns</div>
            <div class="value">{len(df.columns)}</div><div class="sub">features</div>
          </div>
          <div class="stat-card">
            <div class="bar"></div><div class="label">Numerical</div>
            <div class="value">{len(num_cols)}</div><div class="sub">numeric cols</div>
          </div>
          <div class="stat-card">
            <div class="bar"></div><div class="label">Categorical</div>
            <div class="value">{len(cat_cols)}</div><div class="sub">text cols</div>
          </div>
          <div class="stat-card {'good' if null_pct==0 else 'warn' if null_pct<=10 else 'danger'}">
            <div class="bar"></div><div class="label">Missing</div>
            <div class="value">{null_pct}%</div><div class="sub">{df.isnull().sum().sum()} cells</div>
          </div>
        </div>""", unsafe_allow_html=True)

        if dup_cnt > 0:
            st.warning(f"⚠️ **{dup_cnt}** duplicate rows found. PyCaret will handle automatically.")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚡</div><h3>Quick Actions</h3></div>""", unsafe_allow_html=True)
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            if st.button("🗑️ Drop Duplicates", key="drop_dups"):
                before = len(df)
                st.session_state.data = df.drop_duplicates().reset_index(drop=True)
                st.success(f"Removed {before - len(st.session_state.data)} duplicates!"); st.rerun()
        with qa2:
            if st.button("🧹 Drop All-Null Cols", key="drop_null_cols"):
                before = len(df.columns)
                st.session_state.data = df.dropna(axis=1, how='all')
                st.success(f"Removed {before - len(st.session_state.data.columns)} empty columns!"); st.rerun()
        with qa3:
            if st.button("📊 Show Data Types", key="show_dtypes"):
                st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column",0:"Type"}))
        with qa4:
            st.download_button("📥 Download CSV", df.to_csv(index=False),
                               f"dataset_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "text/csv")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🔍</div><h3>Data Preview</h3></div>""", unsafe_allow_html=True)
        search = st.text_input("Filter columns (comma-separated)", placeholder="e.g. Age, Sex, Survived", label_visibility="collapsed")
        show_rows = st.slider("Rows to show", 5, 100, 20, key="preview_rows")
        if search.strip():
            cols_f = [c.strip() for c in search.split(",") if c.strip() in df.columns]
            st.dataframe((df[cols_f] if cols_f else df).head(show_rows))
        else:
            st.dataframe(df.head(show_rows))

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>Column Details</h3></div>""", unsafe_allow_html=True)
            ci = pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str).values,
                                "Non-Null":df.count().values,
                                "Null %":((df.isnull().sum()/len(df))*100).round(1).astype(str)+"%",
                                "Unique":df.nunique().values})
            st.dataframe(ci.astype(str), height=300)
        with c2:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📊</div><h3>Statistical Summary</h3></div>""", unsafe_allow_html=True)
            st.dataframe(df.describe().round(3), height=300)

    # ═══════════════════════════
    # TAB 2 — EDA
    # ═══════════════════════════
    with tab2:
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🧬</div><h3>Exploratory Data Analysis</h3></div>""", unsafe_allow_html=True)

        if not num_cols:
            st.info("No numerical columns found.")
        else:
            st.markdown("#### 📈 Distribution Explorer")
            dv1, dv2 = st.columns([3, 2])
            with dv1:
                col_pick = st.selectbox("Select column", num_cols + cat_cols, key="eda_col")
            with dv2:
                chart_type = st.selectbox("Chart type", ["Histogram","Box","Violin"] if col_pick in num_cols else ["Bar Chart"], key="chart_type")

            cv1, cv2 = st.columns([3, 2])
            with cv1:
                if col_pick in num_cols:
                    if chart_type == "Histogram":
                        fig = px.histogram(df, x=col_pick, nbins=40, color_discrete_sequence=[ACCENT1], template=CHART_TEMPLATE, title=f"Distribution · {col_pick}")
                    elif chart_type == "Box":
                        fig = px.box(df, y=col_pick, color_discrete_sequence=[ACCENT1], template=CHART_TEMPLATE, title=f"Box Plot · {col_pick}")
                    else:
                        fig = px.violin(df, y=col_pick, color_discrete_sequence=[ACCENT1], box=True, template=CHART_TEMPLATE, title=f"Violin · {col_pick}")
                else:
                    vc = df[col_pick].value_counts().head(15)
                    fig = px.bar(x=vc.index, y=vc.values, color=vc.values, color_continuous_scale=[ACCENT2, ACCENT1], template=CHART_TEMPLATE, title=f"Top values · {col_pick}")
                    fig.update_layout(showlegend=False, coloraxis_showscale=False)
                fig.update_layout(**chart_layout(height=340))
                st.plotly_chart(fig, width="stretch")

            with cv2:
                s = df[col_pick]
                rows = {"Count":f"{s.count():,}","Missing":f"{s.isnull().sum()} ({s.isnull().mean()*100:.1f}%)","Unique":f"{s.nunique():,}"}
                if col_pick in num_cols:
                    rows.update({"Mean":f"{s.mean():.4f}","Std":f"{s.std():.4f}","Min":f"{s.min():.4f}","Median":f"{s.median():.4f}","Max":f"{s.max():.4f}","Skew":f"{s.skew():.3f}"})
                st.markdown(f'<div class="sidebar-section" style="margin-top:2.2rem">', unsafe_allow_html=True)
                for k, v in rows.items():
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.78rem;color:{TEXT3};font-weight:600">{k}</span><span style="font-size:.82rem;color:{TEXT1};font-family:'JetBrains Mono',monospace">{v}</span></div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            if len(num_cols) >= 2:
                st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 🔥 Correlation Heatmap")
                corr = df[num_cols[:15]].corr().round(2)
                fig_h = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale=[[0,ACCENTR],[.5,BG2],[1,ACCENT1]],
                    zmid=0, text=corr.values.round(2), texttemplate="%{text}",
                    textfont=dict(size=9, family="JetBrains Mono")))
                fig_h.update_layout(**chart_layout(height=460))
                st.plotly_chart(fig_h, width="stretch")

    # ═══════════════════════════
    # TAB 3 — TRAIN MODEL
    # ═══════════════════════════
    with tab3:
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚙️</div><h3>Training Configuration</h3></div>""", unsafe_allow_html=True)

        tc1, tc2 = st.columns([3, 1])
        with tc1:
            target_col = st.selectbox("🎯 Select Target Column", df.columns.tolist())
        with tc2:
            st.markdown("<br>", unsafe_allow_html=True)
            row_color = ACCENT1 if len(df) <= MAX_ROWS_WARNING else ACCENTY if len(df) <= MAX_ROWS_TRAINING else ACCENTR
            st.markdown(
                f'<div style="padding:.6rem 1rem;background:{BG3};border:1px solid {row_color}44;border-radius:10px;text-align:center">'
                f'<div style="font-size:.65rem;color:{TEXT3};text-transform:uppercase;font-weight:700">Dataset Size</div>'
                f'<div style="font-size:1.1rem;font-weight:900;color:{row_color}">{len(df):,}</div>'
                f'<div style="font-size:.62rem;color:{TEXT3}">rows</div></div>',
                unsafe_allow_html=True
            )

        if target_col:
            ts = df[target_col]
            ptype = detect_problem_type(ts)
            uniq = ts.nunique()
            st.session_state.problem_type = ptype

            card_cls = "clf" if ptype == "classification" else "reg"
            icon = "🎯" if ptype == "classification" else "📈"
            type_lbl = "Classification" if ptype == "classification" else "Regression"
            st.markdown(f"""
            <div class="target-card {card_cls} slide-up">
              <div class="tc-icon">{icon}</div>
              <div>
                <div class="tc-label">Problem Type Detected</div>
                <div class="tc-type">{type_lbl}</div>
                <div class="tc-meta">Target: <code>{target_col}</code> &nbsp;·&nbsp; {uniq} unique values &nbsp;·&nbsp; Auto cross-validation enabled</div>
              </div>
            </div>""", unsafe_allow_html=True)

            if len(df) > MAX_ROWS_TRAINING:
                st.error(
                    f"🚨 **Dataset {len(df):,} rows** — Streamlit Cloud is too large for this.  \n"
                    f"Training will automatically sample **{MAX_ROWS_TRAINING:,} rows** (stratified)."
                )
            elif len(df) > MAX_ROWS_WARNING:
                st.warning(
                    f"⚠️ **{len(df):,} rows** — Thoda bada hai. Training chal jayegi lekin "
                    f"agar crash ho toh {MAX_ROWS_WARNING:,} rows tak chota karo."
                )

            max_folds    = plan_limits["cv_folds_max"]
            has_advanced = plan_limits["advanced_models"]
            max_algo     = plan_limits["max_algorithms"]

            # Determine actual available model count for display
            if ptype == "classification":
                available_models = ADVANCED_CLF_MODELS if has_advanced else SAFE_CLF_MODELS
            else:
                available_models = ADVANCED_REG_MODELS if has_advanced else SAFE_REG_MODELS
            if not has_advanced:
                available_models = [m for m in available_models if m not in BLACKLISTED_FREE]

            # ── PRO MODE ACTIVE banner ──
            if has_advanced:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(74,222,128,0.12),rgba(96,165,250,0.08));
                            border:1.5px solid rgba(74,222,128,0.4);border-radius:14px;
                            padding:.9rem 1.25rem;margin:.75rem 0;display:flex;align-items:center;gap:1rem">
                  <div style="font-size:2rem">⚡</div>
                  <div>
                    <div style="font-size:.9rem;font-weight:900;color:#4ade80;letter-spacing:.02em">
                      {current_plan.upper()} MODE ACTIVE
                    </div>
                    <div style="font-size:.72rem;color:#9ca3af;margin-top:.15rem">
                      All Pro features unlocked — XGBoost ✅ &nbsp; LightGBM ✅ &nbsp; CatBoost ✅ &nbsp; 10-fold CV ✅ &nbsp; Unlimited training ✅
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{"rgba(255,255,255,0.03)" if T=="dark" else "rgba(0,0,0,0.03)"};border:1px solid {BORDER};border-radius:12px;padding:1rem 1.25rem;margin:.75rem 0">
              <span class="insight-chip" style="border-color:{plan_color};color:{plan_color}">{plan_icon} {current_plan.upper()} Plan</span>
              <span class="insight-chip">🔁 Max {max_folds}-fold CV</span>
              <span class="insight-chip">🤖 {len(available_models)} algorithms available</span>
              <span class="insight-chip">📦 {"XGBoost ✅ LightGBM ✅ CatBoost ✅" if has_advanced else "XGBoost 🔒 Pro only"}</span>
              <span class="insight-chip">💾 Max {MAX_ROWS_TRAINING:,} rows (auto-sample)</span>
            </div>""", unsafe_allow_html=True)

            with st.expander("⚙️ Advanced Configuration", expanded=False):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    train_size = st.slider("Training Split", 0.5, 0.9, 0.8, 0.05)
                with ac2:
                    # ✅ FIX #1: CV Folds — plan-aware, crash-safe (min 2 required by PyCaret)
                    if max_folds <= 2:
                        # Free plan: fixed at 2 folds — no slider needed
                        fold = 2
                        st.markdown(
                            f'<div style="padding:.55rem .9rem;background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.30);border-radius:10px;font-size:.78rem;color:#f87171;font-weight:700">'
                            f'🔒 CV Folds: 2 (Free plan)<br>'
                            f'<span style="font-weight:400;color:{TEXT3}">Upgrade to Pro for up to 10-fold CV</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        recommended_fold = min(3, max_folds) if len(df) > MAX_ROWS_WARNING else min(5, max_folds)
                        safe_default_fold = max(min(recommended_fold, max_folds), 2)
                        fold = st.slider(
                            f"CV Folds (max {max_folds})",
                            min_value=2,
                            max_value=max_folds,
                            value=safe_default_fold,
                            help=f"Plan allows max {max_folds} folds. Bade datasets pe {recommended_fold}-fold recommend."
                        )
                with ac3:
                    # ✅ FIX #3: max_models slider properly bounded to plan limit
                    max_models_slider = st.slider(
                        f"Max Models (plan: {len(available_models)})",
                        min_value=min(2, len(available_models)),
                        max_value=len(available_models),
                        value=len(available_models),
                        help=f"How many of the {len(available_models)} available models to compare"
                    )
                ac4, ac5 = st.columns(2)
                with ac4:
                    normalize = st.checkbox("Normalize Features", value=True)
                with ac5:
                    remove_out = st.checkbox("Remove Outliers", value=False)
                st.markdown(
                    f'<div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.30);'
                    f'border-radius:8px;padding:.6rem .9rem;font-size:.78rem;color:{ACCENTY}">'
                    f'💡 <b>Memory Tip:</b> Fewer folds = less RAM. For larger datasets, use 2-3 folds.</div>',
                    unsafe_allow_html=True
                )

            st.session_state.cv_fold = fold

            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                train_clicked = st.button("🚀 Launch Training", key="train_btn", disabled=is_readonly)
            with col_btn2:
                if st.session_state.results is not None:
                    if st.button("🔄 Reset Results", key="reset_btn", disabled=is_readonly):
                        st.session_state.results = None
                        st.session_state.best_model = None
                        st.session_state.training_time = None
                        force_gc()
                        st.rerun()

            if is_readonly:
                st.info("👁️ Read-only mode — training disabled.")

            if train_clicked and not is_readonly:
                can_go, block_msg = can_train(uemail_global)
                if not can_go:
                    st.error(block_msg)
                    st.info("👑 Go to **💳 Upgrade** tab to get unlimited training with Pro!")
                else:
                    progress_bar = st.progress(0)
                    status_box   = st.empty()
                    warn_box     = st.empty()
                    timeline_box = st.empty()

                    steps_labels = [
                        "📦 Data Sampling & Validation",
                        "⚙️ PyCaret Environment Setup",
                        "🤖 Model Comparison (memory-safe)",
                        "🏆 Best Model Selection",
                        "💾 Saving Artifact",
                    ]

                    def render_steps(done_count):
                        html = '<div class="step-timeline">'
                        for i, lbl in enumerate(steps_labels):
                            cls    = "done"   if i < done_count else ("active" if i == done_count else "")
                            icon_s = "✓"      if i < done_count else ("◉"      if i == done_count else str(i+1))
                            html += (
                                f'<div class="step-item {cls}">'
                                f'<div class="step-dot {cls}">{icon_s}</div>'
                                f'<div><div class="step-label">{lbl}</div></div>'
                                f'</div>'
                            )
                        return html + "</div>"

                    timeline_box.markdown(render_steps(0), unsafe_allow_html=True)
                    progress_bar.progress(5)
                    status_box.info("🚀 Training is starting...")

                    try:
                        # ✅ FIX #3: Pass max_models_slider to training function
                        best, results, elapsed, warn_msgs, trained_rows = run_memory_safe_training(
                            df           = df,
                            target_col   = target_col,
                            problem_type = ptype,
                            train_size   = train_size,
                            fold         = fold,
                            normalize    = normalize,
                            remove_out   = remove_out,
                            has_advanced = has_advanced,
                            max_models   = max_models_slider,  # ✅ FIXED
                        )

                        progress_bar.progress(100)
                        timeline_box.markdown(render_steps(len(steps_labels)), unsafe_allow_html=True)

                        st.session_state.best_model    = best
                        st.session_state.results       = results
                        st.session_state.training_time = elapsed

                        for w in warn_msgs:
                            warn_box.warning(w)

                        if uemail_global:
                            try:
                                mc_log   = results.columns[0]
                                nr_log   = results.select_dtypes(include=[np.number]).columns
                                bm_name  = str(results.iloc[0][mc_log])
                                bm_score = float(results.iloc[0][nr_log[0]]) if len(nr_log) else 0.0
                                log_training(
                                    email=uemail_global,
                                    dataset=str(st.session_state.dataset_name or "Uploaded CSV"),
                                    problem_type=str(ptype), best_model=bm_name,
                                    score=bm_score, rows=trained_rows, cols=len(df.columns)
                                )
                                log_activity(uemail_global, "training_complete",
                                            f"{bm_name} | {bm_score:.4f} | {trained_rows} rows")
                            except Exception:
                                pass

                        status_box.success(
                            f"✅ Training complete in **{fmt_time(elapsed)}** "
                            f"({trained_rows:,} rows) — 🏆 Check the Results tab!"
                        )
                        if has_advanced:
                            # Show which pro features were actually used
                            _pro_models_used = [m for m in available_models if m in ["xgboost","lightgbm","catboost"]]
                            _pro_chips = " &nbsp; ".join([f'<span style="background:rgba(74,222,128,0.15);color:#4ade80;border:1px solid rgba(74,222,128,0.4);border-radius:6px;padding:.2rem .6rem;font-size:.72rem;font-weight:800">{m.title()} ✓ PRO</span>' for m in _pro_models_used])
                            st.markdown(f"""
                            <div style="background:rgba(74,222,128,0.07);border:1px solid rgba(74,222,128,0.25);
                                        border-radius:10px;padding:.7rem 1rem;margin-top:.5rem">
                              <div style="font-size:.7rem;font-weight:800;color:#4ade80;margin-bottom:.4rem">⚡ PRO FEATURES USED THIS TRAINING</div>
                              <div style="display:flex;flex-wrap:wrap;gap:.4rem">
                                {_pro_chips}
                                <span style="background:rgba(96,165,250,0.15);color:#60a5fa;border:1px solid rgba(96,165,250,0.4);border-radius:6px;padding:.2rem .6rem;font-size:.72rem;font-weight:800">10-fold CV ✓ PRO</span>
                                <span style="background:rgba(192,132,252,0.15);color:#c084fc;border:1px solid rgba(192,132,252,0.4);border-radius:6px;padding:.2rem .6rem;font-size:.72rem;font-weight:800">Unlimited Training ✓ PRO</span>
                              </div>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.info("💡 Upgrade to Pro for XGBoost, LightGBM, and CatBoost!")
                        st.balloons()

                    except MemoryError as me:
                        progress_bar.progress(0)
                        timeline_box.empty()
                        status_box.error(
                            f"💥 **Memory Overflow!**  \n{str(me)}  \n\n"
                            f"**Quick fixes:**  \n"
                            f"- Dataset ko {MAX_ROWS_TRAINING:,} rows se kam karo  \n"
                            f"- Page refresh karo (Ctrl+R) aur dobara try karo  \n"
                            f"- CV Folds ko 2 par set karo"
                        )
                    except Exception as e:
                        progress_bar.progress(0)
                        timeline_box.empty()
                        err_str = str(e)
                        if any(kw in err_str.lower() for kw in ["memory","killed","oom","cannot allocate"]):
                            status_box.error("💥 **Memory Crash!** Dataset chota karo aur page refresh karo.")
                        else:
                            status_box.error(f"❌ Training failed: {err_str}")

    # ═══════════════════════════
    # TAB 4 — RESULTS
    # ═══════════════════════════
    with tab4:
        if st.session_state.results is None:
            st.markdown(f"""
            <div style="text-align:center;padding:5rem 2rem">
              <div style="font-size:5rem;margin-bottom:1rem;opacity:.6">🏆</div>
              <div style="font-size:1.4rem;font-weight:800;color:{TEXT1};margin-bottom:.5rem">No results yet</div>
              <div style="color:{TEXT2}">Train a model in the ⚙️ Train Model tab first</div>
            </div>""", unsafe_allow_html=True)
        else:
            res_df = st.session_state.results
            model_col   = "Model" if "Model" in res_df.columns else res_df.columns[0]
            num_res     = res_df.select_dtypes(include=[np.number]).columns.tolist()
            best_name   = res_df.iloc[0][model_col]
            metric_name = num_res[0] if num_res else "Score"
            top_score   = res_df.iloc[0][metric_name] if num_res else 0
            folds_used  = st.session_state.cv_fold or 5

            st.markdown(f"""
            <div class="trophy-banner slide-up">
              <div class="trophy-icon">🏆</div>
              <div class="trophy-text">
                <h2>{best_name}</h2>
                <p>Best model via {folds_used}-fold cross-validation · Production ready</p>
              </div>
              <div class="trophy-score">
                <div class="ts-label">{metric_name}</div>
                <div class="ts-value">{top_score:.4f}</div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Pro algorithms used badge ──
            if has_advanced:
                _pro_in_results = [m for m in res_df[model_col].tolist() if any(p in str(m).lower() for p in ["xgboost","lightgbm","catboost","gradient"])]
                _best_is_pro = any(p in str(best_name).lower() for p in ["xgboost","lightgbm","catboost","gradient"])
                _pro_count = len(_pro_in_results)
                st.markdown(f"""
                <div style="background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.2);
                            border-radius:10px;padding:.6rem 1rem;margin-bottom:.75rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
                  <span style="font-size:.75rem;font-weight:800;color:#4ade80">⚡ PRO RESULTS</span>
                  <span style="font-size:.72rem;color:#9ca3af">{_pro_count} Pro algorithms competed in this run</span>
                  {"<span style='background:rgba(74,222,128,0.15);color:#4ade80;border:1px solid rgba(74,222,128,0.4);border-radius:6px;padding:.15rem .5rem;font-size:.7rem;font-weight:800'>🏆 Best model is a PRO algorithm!</span>" if _best_is_pro else ""}
                  <span style="font-size:.7rem;color:#6b7280">XGBoost ✓ · LightGBM ✓ · CatBoost ✓ · {folds_used}-fold CV ✓</span>
                </div>""", unsafe_allow_html=True)

            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                st.download_button("📥 Export Results CSV", res_df.to_csv(index=False),
                                   f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            with ex2:
                # ✅ FIX #2: Pro users get actual .pkl file, not .txt
                if plan_limits["export_model"]:
                    if os.path.exists("best_model.pkl"):
                        with open("best_model.pkl", "rb") as pkl_f:
                            pkl_bytes = pkl_f.read()
                        st.download_button(
                            "📦 Download Model (.pkl)",
                            data=pkl_bytes,
                            file_name=f"best_model_{best_name.replace(' ','_')}.pkl",
                            mime="application/octet-stream",
                            help="Trained model file — load with PyCaret's load_model()"
                        )
                    else:
                        st.info("💾 Model file not found. Re-train to generate.")
                else:
                    if st.button("🔒 Export Model (.pkl) — Pro Only"):
                        st.warning("⚡ Upgrade to Pro to export trained models!")
            with ex3:
                if plan_limits["export_model"]:
                    st.markdown(
                        f'<div style="padding:.6rem 1rem;background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.25);border-radius:10px;font-size:.8rem;color:{ACCENT1};">'
                        f'💾 <b>best_model.pkl</b> ready<br>'
                        f'<span style="color:{TEXT3};font-size:.72rem">Load: <code>load_model("best_model")</code></span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>All Models Ranked</h3></div>""", unsafe_allow_html=True)
            styled = (res_df.style
                      .background_gradient(cmap="RdYlGn", subset=num_res)
                      .format({c: "{:.4f}" for c in num_res})
                      .set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"12px"}))
            st.dataframe(styled, height=360)

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
            ch1, ch2 = st.columns(2)
            with ch1:
                top6   = res_df.head(6)
                colors = [ACCENT1 if i == 0 else BG3 for i in range(len(top6))]
                fig_b  = go.Figure(go.Bar(x=top6[metric_name], y=top6[model_col], orientation="h",
                    marker_color=colors, text=top6[metric_name].round(4), textposition="inside",
                    textfont=dict(size=10, color="white")))
                fig_b.update_layout(**chart_layout(height=360, title=f"Top Models · {metric_name}", yaxis=dict(autorange="reversed")))
                st.plotly_chart(fig_b, width="stretch")
            with ch2:
                rc = num_res[:6]
                bv = res_df.iloc[0][rc]
                mi, ma = bv.min(), bv.max()
                nv = (bv - mi) / (ma - mi + 1e-9)
                fig_r = go.Figure(go.Scatterpolar(
                    r=list(nv.values)+[nv.values[0]], theta=list(nv.index)+[nv.index[0]],
                    fill="toself", fillcolor="rgba(74,222,128,0.18)",
                    line=dict(color=ACCENT1, width=2.5), marker=dict(size=6, color=ACCENT1)))
                fig_r.update_layout(**chart_layout(height=360, showlegend=False, title="Best Model · Metrics Radar",
                    polar=dict(bgcolor=CHART_PAPER, radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER),
                               angularaxis=dict(gridcolor=BORDER))))
                st.plotly_chart(fig_r, width="stretch")

    # ═══════════════════════════
    # TAB 5 — MY HISTORY
    # ═══════════════════════════
    with tab5:
        uhist_h = get_user_history(uemail_global)
        history_limit     = plan_limits["history_entries"]
        full_training_log = uhist_h.get("training_log", [])

        if not plan_limits["full_history"] and len(full_training_log) > history_limit:
            training_log = full_training_log[-history_limit:]
            st.markdown(f"""
            <div style="background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.30);border-radius:12px;padding:.9rem 1.25rem;margin-bottom:1rem;display:flex;align-items:center;gap:.75rem">
              <span style="font-size:1.2rem">⚠️</span>
              <div style="font-size:.85rem;color:{TEXT2}">Showing last <b style="color:{ACCENTY}">{history_limit}</b> of <b>{len(full_training_log)}</b> runs. Upgrade to Pro to see full history.</div>
            </div>""", unsafe_allow_html=True)
        else:
            training_log = full_training_log

        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🗂️</div><h3>Previous Work</h3></div>""", unsafe_allow_html=True)

        if not training_log:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem 1rem;background:{CARD_BG};border:1px solid {BORDER};border-radius:20px">
              <div style="font-size:4rem;margin-bottom:.75rem;opacity:.35">🗂️</div>
              <div style="font-size:1.1rem;font-weight:700;color:{TEXT1}">No projects yet</div>
              <div style="color:{TEXT2};font-size:.875rem">Upload a dataset and train your first model!</div>
            </div>""", unsafe_allow_html=True)
        else:
            for t in training_log[-10:][::-1]:
                pt = t.get("problem_type","—")
                pt_color = ACCENT1 if pt == "classification" else ACCENT2
                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem;margin-bottom:.75rem;display:flex;align-items:center;gap:1rem;position:relative;overflow:hidden">
                  <div style="position:absolute;top:0;left:0;bottom:0;width:3px;background:{pt_color}"></div>
                  <div style="font-size:1.5rem;margin-left:.5rem">{"🎯" if pt=="classification" else "📈"}</div>
                  <div style="flex:1">
                    <div style="font-size:.9rem;font-weight:700;color:{TEXT1}">{t.get("dataset","?")}</div>
                    <div style="font-size:.75rem;color:{TEXT3};margin-top:.15rem">{t.get("best_model","?")} · {t.get("rows",0):,} rows · {t.get("time","")[:16]}</div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;color:{TEXT3}">Score</div>
                    <div style="font-size:1.3rem;font-weight:900;color:{pt_color};font-family:'JetBrains Mono',monospace">{t.get("score",0):.4f}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
        tlog_df = pd.DataFrame(uhist_h.get("training_log", []))
        if not tlog_df.empty:
            st.download_button("📥 Export Training History CSV", tlog_df.to_csv(index=False),
                               f"training_history_{uemail_global[:10]}.csv", "text/csv")

    # ═══════════════════════════
    # TAB 6 — UPGRADE / PRICING
    # ═══════════════════════════
    with tab6:
        users_db_tab    = load_json(USERS_FILE)
        plan_expiry_tab = users_db_tab.get(uemail_global, {}).get("plan_expiry", None)

        if current_plan != "free":
            active_until_txt = f"Active until: <b>{plan_expiry_tab}</b>" if plan_expiry_tab else "Lifetime access"
            plan_banner_bg   = "rgba(74,222,128,0.08)" if current_plan == "pro" else "rgba(192,132,252,0.08)"
            plan_banner_bdr  = "rgba(74,222,128,0.35)" if current_plan == "pro" else "rgba(192,132,252,0.35)"
            st.markdown(f"""
            <div style="background:{plan_banner_bg};border:1px solid {plan_banner_bdr};border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem">
              <div style="font-size:2.5rem">{plan_icon}</div>
              <div>
                <div style="font-size:1.1rem;font-weight:800;color:{plan_color}">You're on {current_plan.upper()} Plan ✓</div>
                <div style="font-size:.85rem;color:{TEXT2};margin-top:.2rem">{active_until_txt}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem;background:{BG3};border:1px solid {BORDER};border-radius:16px;margin-bottom:1.5rem">
              <div style="font-size:1.5rem;font-weight:900;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">Unlock the Full Power of DataForge</div>
              <div style="font-size:.9rem;color:{TEXT2};margin-top:.4rem">You're on the Free plan. Upgrade to access XGBoost, LightGBM, model export, and more.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="section-head"><div class="icon-wrap">💎</div><h3>Choose Your Plan</h3></div>""", unsafe_allow_html=True)
        bill_col1, bill_col2, bill_col3 = st.columns([2, 1, 2])
        with bill_col2:
            billing = st.radio("Billing", ["Monthly", "Annual"], horizontal=True, label_visibility="collapsed", key="billing_cycle")
        is_annual = billing == "Annual"

        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(f"""
            <div class="pricing-card {'popular' if current_plan=='free' else ''}">
              <div class="plan-icon">🌱</div><div class="plan-name">Free</div>
              <div class="price-main" style="color:{TEXT1}">$0</div>
              <div class="price-period">forever free</div>
              <ul class="feature-list">
                <li class="included">3 datasets/day</li><li class="included">8 basic algorithms</li>
                <li class="included">2-fold cross validation</li><li class="included">Basic history (3 entries)</li>
                <li class="not-included">XGBoost / LightGBM</li><li class="not-included">Model export (.pkl)</li>
                <li class="not-included">Priority processing</li><li class="not-included">API access</li>
              </ul>
            </div>""", unsafe_allow_html=True)
            if current_plan == "free":
                st.markdown(f'<div style="text-align:center;padding:.75rem;background:rgba(107,114,128,0.10);border-radius:12px;font-weight:700;color:#9ca3af;font-size:.875rem;margin-top:.5rem">✓ Current Plan</div>', unsafe_allow_html=True)

        with pc2:
            pro_price = PRICING["pro"]["annual_price"] if is_annual else PRICING["pro"]["monthly_price"]
            pro_total = f"Billed ${PRICING['pro']['annual_total']}/year" if is_annual else "Billed monthly"
            st.markdown(f"""
            <div class="pricing-card popular">
              <div class="popular-badge">⭐ Most Popular</div>
              <div style="position:absolute;top:1rem;left:1rem;background:linear-gradient(135deg,#dc2626,#f87171);color:white;font-size:.65rem;font-weight:900;padding:.25rem .7rem;border-radius:999px;text-transform:uppercase;letter-spacing:.06em">🔥 99% OFF</div>
              <div class="plan-icon" style="margin-top:1.5rem">⚡</div><div class="plan-name">Pro</div>
              <div style="font-size:.8rem;color:#9ca3af;text-decoration:line-through;margin-top:.5rem">Was $19/mo</div>
              <div class="price-main" style="color:#4ade80">${pro_price}</div>
              <div class="price-period">/month · {pro_total} · <span style="color:#f87171;font-weight:800">Limited time offer!</span></div>
              <ul class="feature-list">
                <li class="included">Unlimited datasets</li><li class="included">13 algorithms</li>
                <li class="included">10-fold cross validation</li><li class="included">XGBoost, LightGBM, CatBoost ✅</li>
                <li class="included">Export trained model (.pkl) ✅</li><li class="included">50-entry history</li>
                <li class="included">Priority processing</li><li class="not-included">API access (Coming Soon)</li>
              </ul>
            </div>""", unsafe_allow_html=True)
            if current_plan == "pro":
                st.markdown(f'<div style="text-align:center;padding:.75rem;background:rgba(74,222,128,0.10);border-radius:12px;font-weight:700;color:#4ade80;font-size:.875rem;margin-top:.5rem">✓ Current Plan</div>', unsafe_allow_html=True)
            else:
                if st.button("⚡ Upgrade to Pro", key="btn_upgrade_pro"):
                    st.session_state.upgrade_plan_selected = "pro"; st.rerun()

        with pc3:
            ent_price = PRICING["enterprise"]["annual_price"] if is_annual else PRICING["enterprise"]["monthly_price"]
            ent_total = f"Billed ${PRICING['enterprise']['annual_total']}/year" if is_annual else "Billed monthly"
            sav_e     = f"Save ${PRICING['enterprise']['monthly_price']*12 - PRICING['enterprise']['annual_total']}/yr" if is_annual else ""
            sav_e_html= f'<span style="color:#c084fc;font-weight:700">· {sav_e}</span>' if sav_e else ""
            st.markdown(f"""
            <div class="pricing-card">
              <div class="plan-icon">🏢</div><div class="plan-name">Enterprise</div>
              <div class="price-main" style="color:#c084fc">${ent_price}</div>
              <div class="price-period">/month · {ent_total} {sav_e_html}</div>
              <ul class="feature-list">
                <li class="included">Everything in Pro</li><li class="included">Unlimited history</li>
                <li class="included">REST API access (Coming Soon 🔜)</li>
                <li class="included">Team members (Coming Soon 🔜)</li>
                <li class="included">Custom model pipelines</li><li class="included">Dedicated support</li>
                <li class="included">SLA guarantee</li><li class="included">On-premise deployment</li>
              </ul>
            </div>""", unsafe_allow_html=True)
            if current_plan == "enterprise":
                st.markdown(f'<div style="text-align:center;padding:.75rem;background:rgba(192,132,252,0.10);border-radius:12px;font-weight:700;color:#c084fc;font-size:.875rem;margin-top:.5rem">✓ Current Plan</div>', unsafe_allow_html=True)
            else:
                if st.button("🏢 Upgrade to Enterprise", key="btn_upgrade_ent"):
                    st.session_state.upgrade_plan_selected = "enterprise"; st.rerun()

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── PAYMENT FLOW ──
        if st.session_state.upgrade_plan_selected:
            selected_plan = st.session_state.upgrade_plan_selected
            is_ann        = is_annual
            amount_usd    = PRICING[selected_plan]["annual_price"] if is_ann else PRICING[selected_plan]["monthly_price"]
            amount_pkr    = amount_usd * 280

            st.markdown(f"""
            <div style="background:{"rgba(74,222,128,0.05)" if T=="dark" else "rgba(124,58,237,0.05)"};
                        border:2px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(124,58,237,0.25)"};
                        border-radius:20px;padding:2rem;margin-bottom:1.5rem">
              <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem">
                <div style="font-size:2.5rem">{PRICING[selected_plan]['icon']}</div>
                <div>
                  <div style="font-size:1.3rem;font-weight:900;color:{TEXT1}">Complete Your {selected_plan.title()} Upgrade</div>
                  <div style="font-size:.875rem;color:{TEXT2}">{'Annual' if is_ann else 'Monthly'} billing ·
                    <b style="color:{PRICING[selected_plan]['color']}">${amount_usd}/mo</b> ·
                    <b style="color:{ACCENTY}">≈ PKR {amount_pkr:,.0f}{'/year' if is_ann else '/month'}</b>
                  </div>
                </div>
                <div style="margin-left:auto">
            """, unsafe_allow_html=True)
            if st.button("✕ Cancel", key="cancel_upgrade"):
                st.session_state.upgrade_plan_selected = None; st.rerun()
            st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown(f'<div style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin-bottom:.75rem">Step 1 — Choose Payment Method</div>', unsafe_allow_html=True)

            pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
            pm_options = list(PAYMENT_METHODS.keys())
            pm_icons   = {"easypaisa":"📱","jazzcash":"💸","bank_transfer":"🏦","card":"💳"}
            pm_names   = {"easypaisa":"EasyPaisa","jazzcash":"JazzCash","bank_transfer":"Bank Transfer","card":"Debit/Credit Card"}

            if "selected_pm" not in st.session_state:
                st.session_state.selected_pm = "easypaisa"

            for col, pm_key in zip([pm_col1, pm_col2, pm_col3, pm_col4], pm_options):
                with col:
                    is_sel = st.session_state.selected_pm == pm_key
                    st.markdown(f"""
                    <div class="pay-method-card {'selected' if is_sel else ''}">
                      <div class="pm-icon">{pm_icons[pm_key]}</div>
                      <div class="pm-name">{pm_names[pm_key]}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button(f"{'✓ ' if is_sel else ''}Select", key=f"pm_{pm_key}"):
                        st.session_state.selected_pm = pm_key; st.rerun()

            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            pm = PAYMENT_METHODS[st.session_state.selected_pm]

            if st.session_state.selected_pm == "card":
                st.info("💳 Card payments coming soon! Please use EasyPaisa, JazzCash, or Bank Transfer for now.")
            else:
                st.markdown(f'<div style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin-bottom:.75rem">Step 2 — Send Payment</div>', unsafe_allow_html=True)

                inst_col1, inst_col2 = st.columns([3, 2])
                with inst_col1:
                    if st.session_state.selected_pm in ["easypaisa", "jazzcash"]:
                        st.markdown(f"""
                        <div class="account-box"><div class="ab-label">Account Number</div><div class="ab-value">{pm['number']}</div></div>
                        <div class="account-box"><div class="ab-label">Account Name</div><div class="ab-value">{pm['account_name']}</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="account-box"><div class="ab-label">Bank</div><div class="ab-value">{pm['bank']}</div></div>
                        <div class="account-box"><div class="ab-label">Account Title</div><div class="ab-value">{pm['account_title']}</div></div>
                        <div class="account-box"><div class="ab-label">Account Number</div><div class="ab-value">{pm['account_number']}</div></div>
                        <div class="account-box"><div class="ab-label">IBAN</div><div class="ab-value" style="font-size:.85rem">{pm['iban']}</div></div>""", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="background:{"rgba(251,191,36,0.08)" if T=="dark" else "rgba(251,191,36,0.10)"};border:1px solid rgba(251,191,36,0.35);border-radius:12px;padding:1rem;margin-top:.75rem">
                      <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{ACCENTY}">Amount to Send</div>
                      <div style="font-size:1.8rem;font-weight:900;color:{ACCENTY};font-family:'JetBrains Mono',monospace">PKR {amount_pkr:,.0f}</div>
                      <div style="font-size:.75rem;color:{TEXT3};margin-top:.2rem">(${amount_usd}/mo · {'Annual' if is_ann else 'Monthly'} billing)</div>
                    </div>""", unsafe_allow_html=True)

                with inst_col2:
                    st.markdown(f'<div class="instructions-box"><div class="inst-title">📋 How to Pay</div><ol>', unsafe_allow_html=True)
                    for step in pm["instructions"]:
                        st.markdown(f'<li style="padding:.4rem 0;font-size:.82rem;color:{TEXT2};line-height:1.5">{step}</li>', unsafe_allow_html=True)
                    st.markdown("</ol></div>", unsafe_allow_html=True)

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin-bottom:.75rem">Step 3 — Submit Transaction ID</div>', unsafe_allow_html=True)

                sub_col1, sub_col2 = st.columns([3, 2])
                with sub_col1:
                    txn_input = st.text_input("Transaction / Reference ID",
                        placeholder="e.g. TXN123456789 or 12345678",
                        help="The Transaction ID / Reference Number from your payment confirmation",
                        key="txn_id_input")
                    txn_note = st.text_area("Additional notes (optional)",
                        placeholder="Any extra info about your payment...",
                        height=80, key="txn_note_input")
                with sub_col2:
                    st.markdown(f"""
                    <div style="background:{BG3};border:1px solid {BORDER};border-radius:14px;padding:1.25rem;margin-top:1.5rem">
                      <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:.75rem">📦 Order Summary</div>
                      <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.8rem;color:{TEXT2}">Plan</span><span style="font-size:.8rem;font-weight:700;color:{PRICING[selected_plan]['color']}">{selected_plan.title()}</span></div>
                      <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.8rem;color:{TEXT2}">Billing</span><span style="font-size:.8rem;font-weight:700;color:{TEXT1}">{'Annual' if is_ann else 'Monthly'}</span></div>
                      <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.8rem;color:{TEXT2}">Amount (USD)</span><span style="font-size:.8rem;font-weight:700;color:{TEXT1}">${amount_usd}/mo</span></div>
                      <div style="display:flex;justify-content:space-between;padding:.5rem 0"><span style="font-size:.85rem;font-weight:700;color:{TEXT1}">Total (PKR)</span><span style="font-size:.9rem;font-weight:900;color:{ACCENTY}">PKR {amount_pkr:,.0f}</span></div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✅ Submit Payment & Request Upgrade", key="submit_payment"):
                    if not txn_input.strip():
                        st.error("❌ Please enter your Transaction ID / Reference Number.")
                    else:
                        pay_id = save_payment_request(
                            email=uemail_global, plan=selected_plan,
                            billing="annual" if is_ann else "monthly",
                            amount=amount_pkr, payment_method=st.session_state.selected_pm,
                            txn_id=txn_input.strip(), user_name=uname_global
                        )
                        log_activity(uemail_global, "payment_submitted", f"{selected_plan} | PKR {amount_pkr:,.0f} | {pay_id}")
                        email_sent = notify_payment_submitted(
                            user_name=uname_global, email=uemail_global,
                            plan=selected_plan, amount=amount_pkr,
                            method=st.session_state.selected_pm,
                            txn_id=txn_input.strip(), pay_id=pay_id
                        )
                        st.session_state.upgrade_plan_selected = None
                        st.success(f"""
✅ **Payment request submitted successfully!**

**Payment ID:** `{pay_id}`

We'll verify your payment within **2-24 hours** and activate your {selected_plan.title()} plan.
                        """)
                        if email_sent:
                            st.info("📧 Notification email sent to admin successfully.")
                        else:
                            email_log = load_json("dataforge_email_log")
                            last_err_msg = ""
                            for ts, entry in reversed(list(email_log.items())):
                                if "error" in entry:
                                    last_err_msg = entry.get("error", ""); break
                            st.warning(
                                f"⚠️ **Admin email notification failed.** Your payment IS saved (ID: `{pay_id}`).  \n"
                                f"**Fix:** Gmail App Password required — myaccount.google.com → Security → App passwords."
                                + (f"\n\n**Error:** `{last_err_msg}`" if last_err_msg else "")
                            )
                        st.balloons()

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── PAYMENT HISTORY ──
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>My Payment History</h3></div>""", unsafe_allow_html=True)
        user_payments = get_user_payments(uemail_global)

        # Auto-refresh if any payment is pending (check every 10 seconds)
        _has_pending = any(p.get("status") == "pending" for p in user_payments)
        if _has_pending:
            st.info("⏳ **Payment pending approval...** This page will auto-refresh every 10 seconds.")
            import time as _time
            _time.sleep(10)
            st.rerun()
        if not user_payments:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem;background:{CARD_BG};border:1px solid {BORDER};border-radius:16px">
              <div style="font-size:2.5rem;margin-bottom:.5rem;opacity:.4">💳</div>
              <div style="color:{TEXT2};font-size:.875rem">No payment history yet</div>
            </div>""", unsafe_allow_html=True)
        else:
            for pay in user_payments:
                status      = pay.get("status","pending")
                status_icon = "⏳" if status=="pending" else ("✅" if status=="approved" else "❌")
                plan_c      = PRICING.get(pay.get("plan",""),{}).get("color",TEXT2)
                badge_bg,badge_bdr,badge_col = (
                    ("rgba(251,191,36,0.12)","rgba(251,191,36,0.40)","#fbbf24") if status=="pending" else
                    ("rgba(74,222,128,0.12)","rgba(74,222,128,0.40)","#4ade80") if status=="approved" else
                    ("rgba(248,113,113,0.12)","rgba(248,113,113,0.40)","#f87171")
                )
                h  = f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem;margin-bottom:.75rem;position:relative;overflow:hidden">'
                h += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{plan_c},{badge_col})"></div>'
                h += f'<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.2rem"><div style="flex:1;min-width:180px">'
                h += f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem"><span style="font-size:.7rem;font-weight:800;color:{plan_c};background:{plan_c}22;border:1px solid {plan_c}55;border-radius:6px;padding:.15rem .5rem">{pay.get("plan","?").upper()}</span><span style="font-size:.7rem;color:{TEXT3}">{pay.get("billing","monthly").title()} billing</span></div>'
                h += f'<div style="font-size:.88rem;font-weight:700;color:{TEXT1}">PKR {pay.get("amount",0):,.0f} <span style="font-size:.75rem;font-weight:400;color:{TEXT2}">via {pay.get("payment_method","?").replace("_"," ").title()}</span></div>'
                h += f'<div style="font-size:.7rem;color:{TEXT3};margin-top:.2rem;font-family:monospace">Txn: {pay.get("txn_id","?")}</div>'
                h += f'<div style="font-size:.67rem;color:{TEXT3};margin-top:.1rem">{pay.get("submitted_at","")[:16]}</div></div>'
                h += f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:.35rem">'
                h += f'<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.35rem .85rem;border-radius:999px;font-size:.77rem;font-weight:800;background:{badge_bg};border:1px solid {badge_bdr};color:{badge_col}">{status_icon} {status.upper()}</span>'
                if pay.get("processed_at"):
                    h += f'<div style="font-size:.67rem;color:#6b7280">✓ Approved: {pay["processed_at"][:16]}</div>'
                h += f'<div style="font-size:.64rem;color:{TEXT3};font-family:monospace">{pay.get("id","")}</div>'
                h += f'</div></div>'
                if pay.get("admin_note"):
                    h += f'<div style="margin-top:.5rem;font-size:.74rem;color:#9ca3af;background:#1a1a1a;border-radius:8px;padding:.35rem .7rem">📝 {pay["admin_note"]}</div>'
                h += '</div>'
                st.markdown(h, unsafe_allow_html=True)

        # ── FAQ ──
        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">❓</div><h3>Frequently Asked Questions</h3></div>""", unsafe_allow_html=True)
        faqs = [
            ("How long does activation take?", "Usually within 2-4 hours on business days. Maximum 24 hours. We manually verify each payment to prevent fraud."),
            ("What if I send the wrong amount?", "Don't worry! Contact us with your Payment ID and correct Transaction ID. We'll sort it out."),
            ("Can I get a refund?", "Yes, within 7 days of activation if you haven't used the advanced features. Contact support@dataforge.ai"),
            ("Is my payment secure?", "Yes. We use established Pakistani payment networks (EasyPaisa, JazzCash, Meezan Bank). We never store card details."),
            ("What's the PKR exchange rate used?", f"We use a fixed rate of $1 = PKR 280. This is reviewed quarterly. Current effective rate: $19/mo = PKR {19*280:,.0f}/mo"),
            ("Can I upgrade from Pro to Enterprise?", "Yes! Submit a new payment for Enterprise. We'll credit the remaining Pro days."),
            ("When will API access be available?", "API access for Enterprise is currently in development and expected in the next major release. Enterprise subscribers will get early access."),
            ("When will Team Members feature launch?", "Team collaboration is on our roadmap for Enterprise plan. We'll notify all Enterprise users when it's ready."),
        ]
        for q, a in faqs:
            with st.expander(f"❓ {q}"):
                st.markdown(f'<p style="color:{TEXT2};font-size:.9rem;line-height:1.7;margin:0">{a}</p>', unsafe_allow_html=True)

else:
    # ── WELCOME SCREEN ──
    st.markdown(f"""
    <div class="hero-wrap slide-up">
      <h1>Drop Your Data.<br>We Do the Rest.</h1>
      <p>DataForge ML Studio — zero-code AutoML. Upload a CSV, pick a target, hit train. Get a production model in minutes.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("🧬","Smart EDA","Correlation heatmaps, distribution explorer, scatter builder, and missing value charts."),
        ("⚡","AutoML Engine","15+ algorithms compared with k-fold cross-validation. Best model wins — automatically."),
        ("🎯","Smart Detection","Auto-detects regression vs classification. Warns about ID columns. Quick data cleaning."),
        ("🏆","Rich Results","Trophy banner, radar + scatter + bar charts, metric breakdown, model export."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], feats):
        with col:
            st.markdown(f"""
            <div class="feature-card slide-up">
              <div class="fc-icon">{icon}</div>
              <h3>{title}</h3><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:1.5rem">
      <div style="font-size:1.4rem;font-weight:900;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">💎 Plans & Pricing</div>
      <div style="font-size:.875rem;color:{TEXT2};margin-top:.3rem">Upgrade after loading a dataset · Use the 💳 Upgrade tab</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-bottom:2rem">
      <div style="background:{CARD_BG};border:2px solid {BORDER};border-radius:20px;padding:1.75rem">
        <div style="font-size:2rem;margin-bottom:.5rem">🌱</div>
        <div style="font-size:1.1rem;font-weight:900;color:{TEXT1}">Free</div>
        <div style="font-size:2.5rem;font-weight:900;color:{TEXT1};margin:.5rem 0 .25rem">$0</div>
        <div style="font-size:.8rem;color:{TEXT3};margin-bottom:1.25rem">forever free</div>
        <div style="font-size:.82rem;color:{TEXT2};line-height:2">✓ 3 datasets/day<br>✓ 8 basic algorithms<br>✓ 2-fold CV<br><span style="color:{TEXT3}">✗ XGBoost / LightGBM<br>✗ Model export (.pkl)<br>✗ API access</span></div>
        <div style="margin-top:1.25rem;padding:.6rem;background:rgba(107,114,128,0.12);border-radius:10px;text-align:center;font-size:.8rem;font-weight:700;color:#9ca3af">✓ Your Current Plan</div>
      </div>
      <div style="background:{CARD_BG};border:2px solid #4ade80;border-radius:20px;padding:1.75rem;position:relative;box-shadow:0 0 28px rgba(74,222,128,0.12)">
        <div style="position:absolute;top:.9rem;right:.9rem;background:linear-gradient(135deg,#16a34a,#22c55e);color:white;font-size:.62rem;font-weight:800;padding:.2rem .6rem;border-radius:999px;text-transform:uppercase;letter-spacing:.05em">⭐ Most Popular</div>
        <div style="font-size:2rem;margin-bottom:.5rem">⚡</div>
        <div style="font-size:1.1rem;font-weight:900;color:{TEXT1}">Pro</div>
        <div style="font-size:2.5rem;font-weight:900;color:#4ade80;margin:.5rem 0 .25rem">$1<span style="font-size:1rem;font-weight:400;color:{TEXT3}">/mo</span> <span style="background:#dc2626;color:white;font-size:.6rem;font-weight:900;padding:.15rem .5rem;border-radius:999px">🔥 99% OFF</span></div>
        <div style="font-size:.8rem;color:{TEXT3};margin-bottom:1.25rem">or $15/mo billed annually</div>
        <div style="font-size:.82rem;color:{TEXT2};line-height:2">✓ Unlimited datasets<br>✓ 13 algorithms<br>✓ 10-fold CV<br>✓ XGBoost, LightGBM, CatBoost ✅<br>✓ Export model (.pkl) ✅<br>✓ 50-entry history</div>
        <div style="margin-top:1.25rem;padding:.6rem;background:linear-gradient(135deg,rgba(74,222,128,0.15),rgba(74,222,128,0.08));border:1px solid rgba(74,222,128,0.35);border-radius:10px;text-align:center;font-size:.82rem;font-weight:800;color:#4ade80">⚡ Load a dataset → 💳 Upgrade tab</div>
      </div>
      <div style="background:{CARD_BG};border:2px solid {BORDER};border-radius:20px;padding:1.75rem">
        <div style="font-size:2rem;margin-bottom:.5rem">🏢</div>
        <div style="font-size:1.1rem;font-weight:900;color:{TEXT1}">Enterprise</div>
        <div style="font-size:2.5rem;font-weight:900;color:#c084fc;margin:.5rem 0 .25rem">$79<span style="font-size:1rem;font-weight:400;color:{TEXT3}">/mo</span></div>
        <div style="font-size:.8rem;color:{TEXT3};margin-bottom:1.25rem">or $63/mo billed annually</div>
        <div style="font-size:.82rem;color:{TEXT2};line-height:2">✓ Everything in Pro<br>✓ Unlimited history<br>✓ REST API access (Coming Soon 🔜)<br>✓ Team members (Coming Soon 🔜)<br>✓ Dedicated support<br>✓ SLA guarantee</div>
        <div style="margin-top:1.25rem;padding:.6rem;background:rgba(192,132,252,0.10);border:1px solid rgba(192,132,252,0.30);border-radius:10px;text-align:center;font-size:.82rem;font-weight:800;color:#c084fc">🏢 Load a dataset → 💳 Upgrade tab</div>
      </div>
    </div>
    <div style="text-align:center;color:{TEXT3};font-size:.82rem;padding-bottom:1.5rem">
      👈 Upload a CSV/Excel or load a sample dataset from the sidebar to get started
    </div>
    """, unsafe_allow_html=True)
