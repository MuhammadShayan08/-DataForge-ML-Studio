# =====================================================================
#  DataForge ML Studio — Full App with Payment/Upgrade System
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
import warnings, time, io, smtplib, json, os, hashlib, secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataForge ML Studio", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
#  PERSISTENT STORAGE — JSON FILES
# ─────────────────────────────────────────────
USERS_FILE   = "dataforge_users.json"
HISTORY_FILE = "dataforge_history.json"
TOKENS_FILE  = "dataforge_tokens.json"
PAYMENTS_FILE  = "dataforge_payments.json"   # NEW: Payment records
ADMIN_EMAILS   = {"shayan.code1@gmail.com"}  # Add more admin emails here

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
#  PRICING CONFIG — SINGLE SOURCE OF TRUTH
# ─────────────────────────────────────────────
PRICING = {
    "pro": {
        "name": "Pro",
        "icon": "⚡",
        "monthly_price": 19,
        "annual_price": 15,   # per month when billed annually
        "annual_total": 180,
        "color": "#4ade80",
        "features": [
            "Unlimited datasets per month",
            "15+ ML algorithms",
            "10-fold cross-validation",
            "XGBoost, LightGBM, CatBoost",
            "Export trained models (.pkl)",
            "50-entry training history",
            "Priority processing queue",
            "Email support",
        ],
        "not_included": [
            "API access",
            "Team collaboration",
            "Dedicated support",
        ]
    },
    "enterprise": {
        "name": "Enterprise",
        "icon": "🏢",
        "monthly_price": 79,
        "annual_price": 63,
        "annual_total": 756,
        "color": "#c084fc",
        "features": [
            "Everything in Pro",
            "Unlimited history",
            "REST API access",
            "Unlimited team members",
            "Custom model pipelines",
            "Dedicated support channel",
            "SLA guarantee",
            "On-premise deployment option",
        ],
        "not_included": []
    }
}

PAYMENT_METHODS = {
    "easypaisa": {
        "name": "EasyPaisa",
        "icon": "📱",
        "number": "0300-1234567",
        "account_name": "DataForge Studio",
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
        "name": "JazzCash",
        "icon": "💸",
        "number": "0333-7654321",
        "account_name": "DataForge Studio",
        "instructions": [
            "Open JazzCash app on your phone",
            "Go to 'Send Money' → 'Mobile Account'",
            "Enter number: **0333-7654321**",
            "Enter the exact amount for your plan",
            "Add your email in the description/reference",
            "Send the payment and note the Transaction ID",
            "Submit the Transaction ID below for verification",
        ]
    },
    "bank_transfer": {
        "name": "Bank Transfer",
        "icon": "🏦",
        "bank": "Meezan Bank",
        "account_title": "DataForge Technologies",
        "account_number": "01234567890123",
        "iban": "PK36MEZN0001234567890123",
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
        "name": "Debit/Credit Card",
        "icon": "💳",
        "instructions": [
            "Card payments via Stripe are coming soon!",
            "For now, please use EasyPaisa, JazzCash, or Bank Transfer.",
            "We'll notify you when card payments are enabled.",
        ]
    }
}

# ─────────────────────────────────────────────
#  PERSISTENT LOGIN — TOKEN SYSTEM
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

# ─────────────────────────────────────────────
#  PLAN LIMITS
# ─────────────────────────────────────────────
PLAN_LIMITS = {
    "free": {
        "datasets_per_month": 3,
        "max_algorithms":     5,
        "cv_folds_max":       3,
        "history_entries":    3,
        "advanced_models":    False,
        "export_model":       False,
        "full_history":       False,
        "priority_queue":     False,
        "api_access":         False,
        "team_members":       1,
    },
    "pro": {
        "datasets_per_month": 999999,
        "max_algorithms":     15,
        "cv_folds_max":       10,
        "history_entries":    50,
        "advanced_models":    True,
        "export_model":       True,
        "full_history":       True,
        "priority_queue":     True,
        "api_access":         False,
        "team_members":       1,
    },
    "enterprise": {
        "datasets_per_month": 999999,
        "max_algorithms":     15,
        "cv_folds_max":       10,
        "history_entries":    999999,
        "advanced_models":    True,
        "export_model":       True,
        "full_history":       True,
        "priority_queue":     True,
        "api_access":         True,
        "team_members":       999999,
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
    plan = get_user_plan(email)
    return PLAN_LIMITS[plan]

def can_train(email: str) -> tuple:
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
def save_payment_request(email: str, plan: str, billing: str, amount: float,
                          payment_method: str, txn_id: str, user_name: str) -> str:
    """Save a pending payment request. Returns payment ID."""
    payments = load_json(PAYMENTS_FILE)
    pay_id = f"PAY-{secrets.token_hex(4).upper()}"
    payments[pay_id] = {
        "id": pay_id,
        "email": email,
        "name": user_name,
        "plan": plan,
        "billing": billing,
        "amount": amount,
        "payment_method": payment_method,
        "txn_id": txn_id,
        "status": "pending",   # pending | approved | rejected
        "submitted_at": now_str(),
        "processed_at": None,
        "admin_note": ""
    }
    save_json(PAYMENTS_FILE, payments)
    return pay_id

def get_user_payments(email: str) -> list:
    payments = load_json(PAYMENTS_FILE)
    user_pays = [p for p in payments.values() if p.get("email") == email]
    return sorted(user_pays, key=lambda x: x.get("submitted_at",""), reverse=True)

def approve_payment(pay_id: str, admin_note: str = ""):
    """Admin approves a payment — upgrades user plan."""
    payments = load_json(PAYMENTS_FILE)
    if pay_id not in payments:
        return False
    pay = payments[pay_id]
    billing = pay.get("billing", "monthly")
    months = 12 if billing == "annual" else 1
    upgrade_user_plan(pay["email"], pay["plan"], months)
    payments[pay_id]["status"] = "approved"
    payments[pay_id]["processed_at"] = now_str()
    payments[pay_id]["admin_note"] = admin_note
    save_json(PAYMENTS_FILE, payments)
    return True

# ─────────────────────────────────────────────
#  EMAIL NOTIFICATION
# ─────────────────────────────────────────────
NOTIFY_TO = "shayan.code1@gmail.com"
try:
    SMTP_USER = st.secrets["SMTP_USER"]
    SMTP_PASS = st.secrets["SMTP_PASS"]
except Exception:
    SMTP_USER = "shayan.code1@gmail.com"
    SMTP_PASS = "kiuabeuhkmwfpjie"

def send_email(subject: str, body: str):
    """Send email notification. Logs errors to dataforge_email_log.json for debugging."""
    if not SMTP_USER or not SMTP_PASS:
        log = load_json("dataforge_email_log.json")
        log[now_str()] = {"subject": subject, "body": body, "error": "No SMTP credentials"}
        save_json("dataforge_email_log.json", log)
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_TO
        # Plain text part
        msg.attach(MIMEText(body, "plain", "utf-8"))
        # HTML part — nicer formatting
        html_body = f"""
        <html><body style="font-family:monospace;background:#0a0a0a;color:#e5e7eb;padding:24px">
        <div style="max-width:500px;margin:auto;background:#111;border:1px solid #222;border-radius:12px;padding:24px">
        <pre style="white-space:pre-wrap;font-size:13px;color:#d1fae5;margin:0">{body}</pre>
        </div></body></html>"""
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        # Try SSL first (port 465), fallback to STARTTLS (port 587)
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
                        server.ehlo()
                        server.starttls()
                        server.ehlo()
                        server.login(SMTP_USER, SMTP_PASS)
                        server.sendmail(SMTP_USER, NOTIFY_TO, msg.as_string())
                sent = True
                break
            except Exception as e:
                last_err = e
                continue
        if not sent:
            raise last_err
        # Log success
        log = load_json("dataforge_email_log.json")
        log[now_str()] = {"subject": subject, "status": "sent_ok"}
        save_json("dataforge_email_log.json", log)
        return True
    except Exception as e:
        # Save full error for debugging
        log = load_json("dataforge_email_log.json")
        log[now_str()] = {
            "subject": subject,
            "body": body[:500],
            "error": str(e),
            "error_type": type(e).__name__,
            "smtp_user": SMTP_USER,
            "note": (
                "IMPORTANT: Gmail requires an App Password, NOT your regular password. "
                "Go to myaccount.google.com → Security → 2-Step Verification → App passwords. "
                "Generate one for 'Mail' and use that 16-char code as SMTP_PASS."
            )
        }
        save_json("dataforge_email_log.json", log)
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
══════════════════════════════════════════"""
    send_email(subject, body)

def notify_signin(user: dict):
    subject = f"🔑 DataForge — Sign In: {user['name']} ({user['email']})"
    history = load_json(HISTORY_FILE)
    u_hist = history.get(user['email'], {})
    login_count = u_hist.get("login_count", 0)
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

def notify_payment_submitted(user_name: str, email: str, plan: str, amount: float,
                              method: str, txn_id: str, pay_id: str):
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

 To approve: Run approve_payment("{pay_id}")
 in the Admin section of the app.
══════════════════════════════════════════"""
    return send_email(subject, body)

# ─────────────────────────────────────────────
#  USER HISTORY HELPERS
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

def log_training(email: str, dataset: str, problem_type: str, best_model: str, score: float, rows: int, cols: int):
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
    history[email]["training_log"] = history[email]["training_log"][-50:]
    save_json(HISTORY_FILE, history)

# ─────────────────────────────────────────────
#  SESSION STATE + AUTO-LOGIN
# ─────────────────────────────────────────────
for k in ["data","problem_type","best_model","results","training_time","dataset_name","cv_fold"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "signin"
if "upgrade_plan_selected" not in st.session_state:
    st.session_state.upgrade_plan_selected = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.current_user  = None
    st.session_state.login_token   = None
    try:
        qp = st.query_params
        saved_token = qp.get("token", None)
        if saved_token:
            email = validate_token(saved_token)
            if email:
                users_db = load_json(USERS_FILE)
                if email in users_db:
                    udata = users_db[email]
                    st.session_state.authenticated = True
                    st.session_state.current_user  = {"name": udata["name"], "email": email}
                    st.session_state.login_token   = saved_token
    except Exception:
        pass

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
[data-testid="stFileUploader"]>div{{background:{"#111111" if T=="dark" else "#ede9fe"} !important;border:2px dashed {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.40)"} !important;border-radius:12px !important;}}
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

/* ── PRICING CARDS ── */
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

/* ── PAYMENT METHOD CARDS ── */
.pay-method-card{{background:{BG3};border:2px solid {BORDER};border-radius:16px;padding:1.25rem;cursor:pointer;transition:all 0.2s ease !important;text-align:center;}}
.pay-method-card:hover{{border-color:{ACCENT1};transform:translateY(-3px);box-shadow:{"0 8px 24px rgba(74,222,128,0.18)" if T=="dark" else "0 8px 24px rgba(124,58,237,0.18)"};}}
.pay-method-card.selected{{border-color:{ACCENT1};background:{"rgba(74,222,128,0.08)" if T=="dark" else "rgba(124,58,237,0.08)"};}}
.pay-method-card .pm-icon{{font-size:2rem;margin-bottom:.4rem;}}
.pay-method-card .pm-name{{font-size:.85rem;font-weight:700;color:{TEXT1};}}

/* ── PAYMENT STATUS ── */
.pay-status-pending{{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.35);border-radius:12px;padding:.6rem 1rem;display:inline-flex;align-items:center;gap:.5rem;font-size:.8rem;font-weight:700;color:{ACCENTY};}}
.pay-status-approved{{background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.35);border-radius:12px;padding:.6rem 1rem;display:inline-flex;align-items:center;gap:.5rem;font-size:.8rem;font-weight:700;color:{ACCENT1};}}
.pay-status-rejected{{background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.35);border-radius:12px;padding:.6rem 1rem;display:inline-flex;align-items:center;gap:.5rem;font-size:.8rem;font-weight:700;color:{ACCENTR};}}

/* PLAN BADGE */
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

/* INSTRUCTIONS BOX */
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
#  AUTH SCREEN
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown(f"""<style>
    section[data-testid="stSidebar"]{{display:none !important;}}
    .block-container{{padding:0 !important;max-width:100% !important;}}
    [data-testid="stAppViewBlockContainer"]{{display:flex;align-items:center;justify-content:center;min-height:100vh;background:{BG} !important;}}
    </style>""", unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown(f"""
        <div style="padding:3rem 0">
          <div style="text-align:center;margin-bottom:2rem">
            <div style="font-size:3.5rem;margin-bottom:.5rem">⚡</div>
            <h1 style="font-size:2rem;font-weight:900;margin:0 0 .25rem;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">DataForge ML Studio</h1>
            <p style="color:{TEXT3};font-size:.9rem;margin:0">AutoML that actually vibes</p>
          </div>
        </div>""", unsafe_allow_html=True)

        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            signin_active = st.session_state.auth_mode == "signin"
            if st.button("🔑 Sign In", key="tab_signin", type="primary" if signin_active else "secondary"):
                st.session_state.auth_mode = "signin"; st.rerun()
        with mode_col2:
            signup_active = st.session_state.auth_mode == "signup"
            if st.button("✨ Create Account", key="tab_signup", type="primary" if signup_active else "secondary"):
                st.session_state.auth_mode = "signup"; st.rerun()

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        if st.session_state.auth_mode == "signup":
            su_name  = st.text_input("Full Name", placeholder="Enter your full name", key="su_name")
            su_email = st.text_input("Email Address", placeholder="your@email.com", key="su_email")
            su_pass  = st.text_input("Password", type="password", placeholder="Create a strong password", key="su_pass")
            su_pass2 = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="su_pass2")

            if st.button("🚀 Create Account & Enter Studio", key="do_signup"):
                if not su_name or not su_email or not su_pass:
                    st.error("❌ Please fill in all fields.")
                elif "@" not in su_email:
                    st.error("❌ Please enter a valid email address.")
                elif su_pass != su_pass2:
                    st.error("❌ Passwords do not match.")
                elif len(su_pass) < 6:
                    st.error("❌ Password must be at least 6 characters.")
                else:
                    users_db = load_json(USERS_FILE)
                    if su_email in users_db:
                        st.error("❌ An account with this email already exists.")
                    else:
                        users_db[su_email] = {
                            "name": su_name, "email": su_email,
                            "password_hash": hash_password(su_pass),
                            "signup_date": now_str(), "plan": "free"
                        }
                        save_json(USERS_FILE, users_db)
                        update_user_history(su_email, {
                            "signup_date": now_str(), "last_login": now_str(),
                            "login_count": 1, "datasets_trained": 0,
                            "training_log": [], "activity_log": [{"time": now_str(), "action": "signup", "detail": "Account created"}]
                        })
                        notify_signup({"name": su_name, "email": su_email})
                        st.session_state.authenticated = True
                        st.session_state.current_user = {"name": su_name, "email": su_email}
                        token = create_token(su_email)
                        st.session_state.login_token = token
                        try:
                            st.query_params["token"] = token
                        except:
                            pass
                        st.success(f"✅ Welcome to DataForge, {su_name}!")
                        time.sleep(0.8); st.rerun()
        else:
            si_email = st.text_input("Email Address", placeholder="your@email.com", key="si_email")
            si_pass  = st.text_input("Password", type="password", placeholder="Enter your password", key="si_pass")

            if st.button("⚡ Sign In & Launch Studio", key="do_signin"):
                if not si_email or not si_pass:
                    st.error("❌ Please enter your email and password.")
                else:
                    users_db = load_json(USERS_FILE)
                    if si_email not in users_db:
                        st.error("❌ No account found. Please Create Account first.")
                    elif users_db[si_email]["password_hash"] != hash_password(si_pass):
                        st.error("❌ Incorrect password. Please try again.")
                    else:
                        udata = users_db[si_email]
                        h = get_user_history(si_email)
                        new_count = h.get("login_count", 0) + 1
                        update_user_history(si_email, {"last_login": now_str(), "login_count": new_count})
                        log_activity(si_email, "signin", f"Login #{new_count}")
                        notify_signin({"name": udata["name"], "email": si_email})
                        st.session_state.authenticated = True
                        st.session_state.current_user = {"name": udata["name"], "email": si_email}
                        token = create_token(si_email)
                        st.session_state.login_token = token
                        try:
                            st.query_params["token"] = token
                        except:
                            pass
                        st.success(f"✅ Welcome back, {udata['name']}!")
                        time.sleep(0.8); st.rerun()

        st.markdown(f'<div style="text-align:center;margin-top:1.5rem;font-size:.78rem;color:{TEXT3}">🔒 Your information is stored securely on the server</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  GET CURRENT USER (after auth)
# ─────────────────────────────────────────────
uemail_global = st.session_state.current_user.get("email","") if st.session_state.current_user else ""
uname_global  = st.session_state.current_user.get("name","User") if st.session_state.current_user else "User"
current_plan  = get_user_plan(uemail_global)
plan_limits   = PLAN_LIMITS[current_plan]

PLAN_COLORS = {"free": "#6b7280", "pro": "#4ade80", "enterprise": "#c084fc"}
PLAN_ICONS  = {"free": "🌱", "pro": "⚡", "enterprise": "🏢"}
plan_color  = PLAN_COLORS.get(current_plan, "#6b7280")
plan_icon   = PLAN_ICONS.get(current_plan, "🌱")
is_admin    = uemail_global in ADMIN_EMAILS

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

_tcol1, _tcol2, _tcol3 = st.columns([9, 1, 1])
with _tcol2:
    if st.button("⬜ White" if T=="dark" else "⬛ Black", key="theme_btn"):
        st.session_state.theme = "light" if T=="dark" else "dark"; st.rerun()
with _tcol3:
    if st.button("🚪 Logout", key="logout_btn"):
        tok = st.session_state.get("login_token")
        if tok:
            delete_token(tok)
        try:
            st.query_params.clear()
        except:
            pass
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.login_token = None
        st.rerun()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    uhist_sb = get_user_history(uemail_global)

    # Plan status in sidebar
    users_db_sb = load_json(USERS_FILE)
    plan_expiry_sb = users_db_sb.get(uemail_global, {}).get("plan_expiry", None)
    plan_expired_sb  = users_db_sb.get(uemail_global, {}).get("plan_expired", False)
    expiry_html      = f'<div style="font-size:.65rem;color:{TEXT3};margin-top:.4rem">Expires: {plan_expiry_sb}</div>' if plan_expiry_sb and current_plan != "free" else ""
    expired_html     = f'<div style="font-size:.65rem;color:#f87171;margin-top:.4rem">⚠️ Plan expired — downgraded to Free</div>' if plan_expired_sb else ""
    name_color       = "#9ca3af" if T == "dark" else "#888888"

    st.markdown(f"""
    <div class="sidebar-section" style="text-align:center">
      <div style="font-size:1.5rem;margin-bottom:.3rem">{plan_icon}</div>
      <div style="font-size:.75rem;font-weight:700;color:{name_color};margin-bottom:.3rem">{uname_global}</div>
      <span class="plan-badge {current_plan}">{plan_icon} {current_plan.upper()} Plan</span>
      {expiry_html}
      {expired_html}
    </div>
    """, unsafe_allow_html=True)

    if current_plan == "free":
        st.markdown(f"""
        <div style="background:{"rgba(251,191,36,0.06)" if T=="dark" else "rgba(251,191,36,0.08)"};border:1px dashed {"rgba(251,191,36,0.35)" if T=="dark" else "rgba(251,191,36,0.45)"};border-radius:10px;padding:.75rem 1rem;margin-bottom:.75rem;text-align:center">
          <div style="font-size:.72rem;font-weight:700;color:{ACCENTY}">⚡ Upgrade to unlock all features</div>
          <div style="font-size:.65rem;color:{TEXT3};margin-top:.2rem">Pro from $19/mo</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title">📂 Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], label_visibility="collapsed")
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state.data = df_up
            st.session_state.dataset_name = uploaded.name
            st.success(f"✓ {len(df_up):,} rows loaded")
        except Exception as e:
            st.error(str(e))

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-title">🎯 Sample Datasets</div>', unsafe_allow_html=True)
    sample = st.selectbox("Pick one", ["— choose —","🚢 Titanic","💎 Diamonds","🌸 Iris"], label_visibility="collapsed")
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
                st.success(f"✓ {len(df_s):,} rows")
                st.info(f"💡 Target: **{hint}**")
            except Exception as e:
                st.error(str(e))

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
    if st.button("🚪 Sign Out", key="sidebar_logout"):
        tok = st.session_state.get("login_token")
        if tok:
            delete_token(tok)
        try:
            st.query_params.clear()
        except:
            pass
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.login_token = None
        st.rerun()

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

def safe_compare(ptype, n_select=1):
    fn = clf_compare if ptype == "classification" else reg_compare
    res = fn(n_select=n_select, verbose=False)
    return res[0] if isinstance(res, list) else res

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
#  ADMIN PANEL — always accessible, no dataset needed
# ─────────────────────────────────────────────
if is_admin:
    with st.expander("🔐 Admin Panel", expanded=False):
        all_payments  = load_json(PAYMENTS_FILE)
        all_users_db  = load_json(USERS_FILE)
        all_history   = load_json(HISTORY_FILE)

        pending_pays  = [p for p in all_payments.values() if p.get("status")=="pending"]
        approved_pays = [p for p in all_payments.values() if p.get("status")=="approved"]
        rejected_pays = [p for p in all_payments.values() if p.get("status")=="rejected"]
        total_revenue = sum(p.get("amount",0) for p in approved_pays)

        # Header
        st.markdown(f'<div style="background:linear-gradient(135deg,rgba(192,132,252,0.10),rgba(96,165,250,0.08));border:1px solid rgba(192,132,252,0.35);border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1rem"><span style="font-size:2rem">🔐</span><div><div style="font-size:1.1rem;font-weight:900;color:#c084fc">Admin Control Panel</div><div style="font-size:.8rem;color:{TEXT2}">{uname_global} ({uemail_global}) · Admin Access</div></div></div>', unsafe_allow_html=True)

        # Stats
        sa1,sa2,sa3,sa4,sa5 = st.columns(5)
        for col,lbl,val,sub in [
            (sa1,"Total Users",    len(all_users_db),          "registered"),
            (sa2,"⏳ Pending",     len(pending_pays),          "need approval"),
            (sa3,"✅ Approved",    len(approved_pays),         "payments"),
            (sa4,"❌ Rejected",    len(rejected_pays),         "payments"),
            (sa5,"Revenue (PKR)", f"{total_revenue:,.0f}",    "collected"),
        ]:
            with col:
                st.metric(lbl, val, sub)

        adm1, adm2, adm3, adm4 = st.tabs([
            f"⏳ Pending ({len(pending_pays)})",
            f"✅ Approved ({len(approved_pays)})",
            "👥 All Users",
            "📧 Email Log"
        ])

        # ── PENDING ──
        with adm1:
            if not pending_pays:
                st.success("✨ No pending payments — all caught up!")
            for pay in sorted(pending_pays, key=lambda x: x.get("submitted_at",""), reverse=True):
                plan_c = PRICING.get(pay.get("plan",""),{}).get("color",TEXT2)
                pid    = pay.get("id","")
                h  = f'<div style="background:{CARD_BG};border:2px solid rgba(251,191,36,0.35);border-radius:16px;padding:1.25rem;margin-bottom:.75rem;position:relative;overflow:hidden">'
                h += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#fbbf24,{plan_c})"></div>'
                h += f'<div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.2rem">'
                h += f'<div style="flex:1;min-width:200px">'
                h += f'<div style="font-size:.95rem;font-weight:800;color:{TEXT1}">{pay.get("name","?")}</div>'
                h += f'<div style="font-size:.72rem;color:{TEXT3};margin-bottom:.5rem">{pay.get("email","?")}</div>'
                h += f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem">'
                h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Plan</div><div style="font-weight:800;color:{plan_c}">{pay.get("plan","?").upper()}</div></div>'
                h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Amount</div><div style="font-weight:800;color:#fbbf24">PKR {pay.get("amount",0):,.0f}</div></div>'
                h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Method</div><div style="font-weight:600;color:{TEXT1}">{pay.get("payment_method","?").replace("_"," ").title()}</div></div>'
                h += f'<div style="background:{BG3};border-radius:8px;padding:.4rem .6rem"><div style="font-size:.6rem;text-transform:uppercase;color:{TEXT3}">Billing</div><div style="font-weight:600;color:{TEXT1}">{pay.get("billing","?").title()}</div></div>'
                h += f'</div></div>'
                h += f'<div style="flex:1;min-width:180px">'
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
                            log_activity(pay.get("email",""), "plan_rejected", f"rejected by admin")
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

        # ── APPROVED ──
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
                    h += f'<div style="font-size:.65rem;color:{TEXT3};font-family:monospace">{pay.get("id","")}</div>'
                    h += '</div>'
                    st.markdown(h, unsafe_allow_html=True)

        # ── ALL USERS ──
        with adm3:
            user_search = st.text_input("🔍 Search by name or email", key="adm_user_search")
            for em, ud in sorted(all_users_db.items(), key=lambda x: x[1].get("signup_date",""), reverse=True):
                if user_search and user_search.lower() not in em.lower() and user_search.lower() not in ud.get("name","").lower():
                    continue
                u_plan  = get_user_plan(em)
                u_pc    = PLAN_COLORS.get(u_plan,"#6b7280")
                u_pi    = PLAN_ICONS.get(u_plan,"🌱")
                u_hist  = all_history.get(em,{})
                h  = f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.4rem;display:flex;align-items:center;gap:.9rem;flex-wrap:wrap">'
                h += f'<div style="width:34px;height:34px;border-radius:50%;background:{u_pc}20;border:2px solid {u_pc}50;display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0">{u_pi}</div>'
                h += f'<div style="flex:1;min-width:160px"><div style="font-size:.9rem;font-weight:700;color:{TEXT1}">{ud.get("name","?")}</div><div style="font-size:.7rem;color:{TEXT3}">{em}</div></div>'
                h += f'<span style="font-size:.7rem;font-weight:800;color:{u_pc};background:{u_pc}18;border:1px solid {u_pc}44;border-radius:6px;padding:.2rem .55rem">{u_plan.upper()}</span>'
                h += f'<div style="text-align:center;min-width:45px"><div style="font-size:1rem;font-weight:900;color:#60a5fa">{u_hist.get("login_count",0)}</div><div style="font-size:.6rem;color:{TEXT3}">logins</div></div>'
                h += f'<div style="font-size:.7rem;color:{TEXT3}">Joined: {ud.get("signup_date","")[:10]}</div>'
                h += '</div>'
                st.markdown(h, unsafe_allow_html=True)
                with st.expander(f"⚙️ Manage {ud.get('name','?')}"):
                    mc1,mc2,mc3 = st.columns(3)
                    with mc1:
                        np_ = st.selectbox("Set Plan", ["free","pro","enterprise"], index=["free","pro","enterprise"].index(u_plan), key=f"adm_plan_{em}")
                    with mc2:
                        nm_ = st.number_input("Months", 1, 24, 1, key=f"adm_months_{em}")
                    with mc3:
                        st.write("")
                        if st.button("💾 Apply", key=f"adm_apply_{em}"):
                            upgrade_user_plan(em, np_, int(nm_))
                            log_activity(em, "plan_changed_by_admin", f"Set to {np_} for {nm_}mo")
                            st.success(f"✅ Done!"); st.rerun()

        # ── EMAIL LOG ──
        with adm4:
            email_log = load_json("dataforge_email_log.json")
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
                        save_json("dataforge_email_log.json",{})
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
                    if has_err:
                        h += f'<div style="font-size:.72rem;color:#f87171;margin-top:.25rem;background:rgba(248,113,113,0.07);border-radius:6px;padding:.25rem .5rem">Error: {entry.get("error","")}</div>'
                    if entry.get("note"):
                        h += f'<div style="font-size:.7rem;color:#fbbf24;margin-top:.2rem;background:rgba(251,191,36,0.07);border-radius:6px;padding:.25rem .5rem">{entry["note"][:200]}</div>'
                    h += '</div></div>'
                    st.markdown(h, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
if st.session_state.data is not None:
    df = st.session_state.data
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = df.duplicated().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    if is_admin:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊  Data Explorer", "🧬  EDA & Insights", "⚙️  Train Model",
            "🏆  Results", "📜  My History", "💳  Upgrade"
        ])
    else:
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
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig_h, use_container_width=True)

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
            st.info(f"📐 {len(df):,} rows available")

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

            max_folds = plan_limits["cv_folds_max"]
            has_advanced = plan_limits["advanced_models"]
            st.markdown(f"""
            <div style="background:{"rgba(255,255,255,0.03)" if T=="dark" else "rgba(0,0,0,0.03)"};border:1px solid {BORDER};border-radius:12px;padding:1rem 1.25rem;margin:.75rem 0">
              <span class="insight-chip" style="border-color:{plan_color};color:{plan_color}">{plan_icon} {current_plan.upper()} Plan</span>
              <span class="insight-chip">🔁 Max {max_folds}-fold CV</span>
              <span class="insight-chip">🤖 {"15+ algorithms" if has_advanced else "5 basic algorithms"}</span>
              <span class="insight-chip">📦 {"XGBoost, LGBM included" if has_advanced else "XGBoost locked 🔒"}</span>
            </div>""", unsafe_allow_html=True)

            with st.expander("⚙️ Advanced Configuration", expanded=False):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    train_size = st.slider("Training Split", 0.5, 0.9, 0.8, 0.05)
                with ac2:
                    fold = st.slider(f"CV Folds (max {max_folds})", 2, max_folds, min(5, max_folds))
                with ac3:
                    max_algo_slider = plan_limits["max_algorithms"]
                    max_models = st.slider(f"Max Models", 3, max_algo_slider, min(10, max_algo_slider))
                ac4, ac5 = st.columns(2)
                with ac4:
                    normalize = st.checkbox("Normalize Features", value=True)
                with ac5:
                    remove_out = st.checkbox("Remove Outliers", value=False)
            st.session_state.cv_fold = fold

            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                train_clicked = st.button("🚀 Launch Training", key="train_btn")
            with col_btn2:
                if st.session_state.results is not None:
                    if st.button("🔄 Reset Results", key="reset_btn"):
                        st.session_state.results = None
                        st.session_state.best_model = None
                        st.session_state.training_time = None
                        st.rerun()

            if train_clicked:
                steps = [
                    ("Data Preprocessing",   "Handling nulls, encoding categoricals…"),
                    ("Feature Engineering",  "Scaling, transformations, pipeline setup…"),
                    ("Model Comparison",     f"Running algorithms with {fold}-fold cross-validation…"),
                    ("Best Model Selection", "Picking winner by metric score…"),
                    ("Saving Artifact",      "Serializing model to best_model.pkl…"),
                ]
                placeholder = st.empty()

                def render_steps(done):
                    html = '<div class="step-timeline">'
                    for i, (lbl, sub) in enumerate(steps):
                        cls = "done" if i < done else ("active" if i == done else "")
                        icon_s = "✓" if i < done else ("◉" if i == done else str(i+1))
                        html += f"""<div class="step-item {cls}"><div class="step-dot {cls}">{icon_s}</div><div><div class="step-label">{lbl}</div><div class="step-sub">{sub}</div></div></div>"""
                    return html + "</div>"

                try:
                    t0 = time.time()
                    for si in range(len(steps)):
                        placeholder.markdown(render_steps(si), unsafe_allow_html=True)
                        if si == 2:
                            kw_setup = dict(data=df, target=target_col, train_size=float(train_size),
                                            fold=int(fold), normalize=normalize, verbose=False, html=False, session_id=123)
                            if remove_out and ptype == "regression" and len(df) > 100:
                                kw_setup["remove_outliers"] = True
                            kw_compare = dict(verbose=False, n_select=1)
                            if not has_advanced:
                                kw_compare["exclude"] = ['xgboost', 'lightgbm', 'catboost']
                            if ptype == "classification":
                                clf_setup(**kw_setup)
                                best = clf_compare(**kw_compare)
                                results = clf_pull()
                                clf_save(best, "best_model")
                            else:
                                reg_setup(**kw_setup)
                                best = reg_compare(**kw_compare)
                                results = reg_pull()
                                reg_save(best, "best_model")
                            st.session_state.best_model = best
                            st.session_state.results = results
                        time.sleep(0.3)

                    placeholder.markdown(render_steps(len(steps)), unsafe_allow_html=True)
                    elapsed = time.time() - t0
                    st.session_state.training_time = elapsed

                    if uemail_global:
                        try:
                            res_log = st.session_state.results
                            mc_log = res_log.columns[0]
                            nr_log = res_log.select_dtypes(include=[np.number]).columns
                            bm_name = str(res_log.iloc[0][mc_log])
                            bm_score = float(res_log.iloc[0][nr_log[0]]) if len(nr_log) else 0.0
                            log_training(email=uemail_global, dataset=str(st.session_state.dataset_name or "Uploaded CSV"),
                                         problem_type=str(ptype), best_model=bm_name, score=bm_score, rows=len(df), cols=len(df.columns))
                            log_activity(uemail_global, "training_complete", f"{bm_name} | {bm_score:.4f}")
                        except:
                            pass

                    st.success(f"✅ Training complete in **{fmt_time(elapsed)}**! Head to the 🏆 Results tab.")
                    if not has_advanced:
                        st.info("💡 Upgrade to Pro for XGBoost, LightGBM, and CatBoost!")
                    st.balloons()

                except Exception as e:
                    placeholder.empty()
                    st.error(f"❌ Training failed: {str(e)}")

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
            model_col = "Model" if "Model" in res_df.columns else res_df.columns[0]
            num_res = res_df.select_dtypes(include=[np.number]).columns.tolist()
            best_name = res_df.iloc[0][model_col]
            metric_name = num_res[0] if num_res else "Score"
            top_score = res_df.iloc[0][metric_name] if num_res else 0
            folds_used = st.session_state.cv_fold or 5

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

            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                st.download_button("📥 Export Results CSV", res_df.to_csv(index=False),
                                   f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
            with ex2:
                if plan_limits["export_model"]:
                    model_info = f"Best Model: {best_name}\n{metric_name}: {top_score:.4f}\nFolds: {folds_used}"
                    st.download_button("📋 Export Model Info", model_info, "model_info.txt", "text/plain")
                else:
                    if st.button("🔒 Export Model (.pkl) — Pro Only"):
                        st.warning("⚡ Upgrade to Pro to export trained models!")
            with ex3:
                if plan_limits["export_model"]:
                    st.info(f"💾 Saved: `best_model.pkl`")

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
                top6 = res_df.head(6)
                colors = [ACCENT1 if i == 0 else BG3 for i in range(len(top6))]
                fig_b = go.Figure(go.Bar(x=top6[metric_name], y=top6[model_col], orientation="h",
                    marker_color=colors, text=top6[metric_name].round(4), textposition="inside",
                    textfont=dict(size=10, color="white")))
                fig_b.update_layout(**chart_layout(height=360, title=f"Top Models · {metric_name}", yaxis=dict(autorange="reversed")))
                st.plotly_chart(fig_b, use_container_width=True)
            with ch2:
                rc = num_res[:6]
                bv = res_df.iloc[0][rc]
                mi, ma = bv.min(), bv.max()
                nv = (bv - mi) / (ma - mi + 1e-9)
                fig_r = go.Figure(go.Scatterpolar(
                    r=list(nv.values) + [nv.values[0]], theta=list(nv.index) + [nv.index[0]],
                    fill="toself", fillcolor=f"rgba(74,222,128,0.18)",
                    line=dict(color=ACCENT1, width=2.5), marker=dict(size=6, color=ACCENT1)))
                fig_r.update_layout(**chart_layout(height=360, showlegend=False, title="Best Model · Metrics Radar",
                    polar=dict(bgcolor=CHART_PAPER, radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER),
                               angularaxis=dict(gridcolor=BORDER))))
                st.plotly_chart(fig_r, use_container_width=True)

    # ═══════════════════════════
    # TAB 5 — MY HISTORY
    # ═══════════════════════════
    with tab5:
        uhist_h = get_user_history(uemail_global)
        history_limit = plan_limits["history_entries"]
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
                ptype = t.get("problem_type","—")
                ptype_color = ACCENT1 if ptype == "classification" else ACCENT2
                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem;margin-bottom:.75rem;display:flex;align-items:center;gap:1rem;position:relative;overflow:hidden">
                  <div style="position:absolute;top:0;left:0;bottom:0;width:3px;background:{ptype_color}"></div>
                  <div style="font-size:1.5rem;margin-left:.5rem">{"🎯" if ptype=="classification" else "📈"}</div>
                  <div style="flex:1">
                    <div style="font-size:.9rem;font-weight:700;color:{TEXT1}">{t.get("dataset","?")}</div>
                    <div style="font-size:.75rem;color:{TEXT3};margin-top:.15rem">{t.get("best_model","?")} · {t.get("rows",0):,} rows · {t.get("time","")[:16]}</div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;color:{TEXT3}">Score</div>
                    <div style="font-size:1.3rem;font-weight:900;color:{ptype_color};font-family:'JetBrains Mono',monospace">{t.get("score",0):.4f}</div>
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
        # ── Current plan status banner ──
        users_db_tab = load_json(USERS_FILE)
        plan_expiry_tab = users_db_tab.get(uemail_global, {}).get("plan_expiry", None)

        if current_plan != "free":
            active_until_txt = f"Active until: <b>{plan_expiry_tab}</b>" if plan_expiry_tab else "Lifetime access"
            plan_banner_bg   = "rgba(74,222,128,0.08)" if current_plan == "pro" else "rgba(192,132,252,0.08)"
            plan_banner_bdr  = "rgba(74,222,128,0.35)" if current_plan == "pro" else "rgba(192,132,252,0.35)"
            st.markdown(f"""
            <div style="background:{plan_banner_bg};
                        border:1px solid {plan_banner_bdr};
                        border-radius:16px;padding:1.5rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1rem">
              <div style="font-size:2.5rem">{plan_icon}</div>
              <div>
                <div style="font-size:1.1rem;font-weight:800;color:{plan_color}">You're on {current_plan.upper()} Plan ✓</div>
                <div style="font-size:.85rem;color:{TEXT2};margin-top:.2rem">
                  {active_until_txt}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem;background:{BG3};border:1px solid {BORDER};border-radius:16px;margin-bottom:1.5rem">
              <div style="font-size:1.5rem;font-weight:900;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">
                Unlock the Full Power of DataForge
              </div>
              <div style="font-size:.9rem;color:{TEXT2};margin-top:.4rem">
                You're on the Free plan. Upgrade to access XGBoost, LightGBM, model export, and more.
              </div>
            </div>""", unsafe_allow_html=True)

        # ── BILLING TOGGLE ──
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">💎</div><h3>Choose Your Plan</h3></div>""", unsafe_allow_html=True)
        bill_col1, bill_col2, bill_col3 = st.columns([2, 1, 2])
        with bill_col2:
            billing = st.radio("Billing", ["Monthly", "Annual"], horizontal=True, label_visibility="collapsed", key="billing_cycle")

        is_annual = billing == "Annual"

        # ── PRICING CARDS ──
        pc1, pc2, pc3 = st.columns(3)

        # FREE
        with pc1:
            st.markdown(f"""
            <div class="pricing-card {'popular' if current_plan=='free' else ''}">
              <div class="plan-icon">🌱</div>
              <div class="plan-name">Free</div>
              <div class="price-main" style="color:{TEXT1}">$0</div>
              <div class="price-period">forever free</div>
              <ul class="feature-list">
                <li class="included">3 datasets/month</li>
                <li class="included">5 basic algorithms</li>
                <li class="included">3-fold cross validation</li>
                <li class="included">Basic history (3 entries)</li>
                <li class="not-included">XGBoost / LightGBM</li>
                <li class="not-included">Model export (.pkl)</li>
                <li class="not-included">Priority processing</li>
                <li class="not-included">API access</li>
              </ul>
            </div>""", unsafe_allow_html=True)
            if current_plan == "free":
                st.markdown(f'<div style="text-align:center;padding:.75rem;background:rgba(107,114,128,0.10);border-radius:12px;font-weight:700;color:#9ca3af;font-size:.875rem;margin-top:.5rem">✓ Current Plan</div>', unsafe_allow_html=True)

        # PRO
        with pc2:
            pro_price    = PRICING["pro"]["annual_price"] if is_annual else PRICING["pro"]["monthly_price"]
            pro_total    = f"Billed ${PRICING['pro']['annual_total']}/year" if is_annual else "Billed monthly"
            savings      = f"Save ${PRICING['pro']['monthly_price']*12 - PRICING['pro']['annual_total']}/yr" if is_annual else ""
            savings_html = f'<span style="color:#4ade80;font-weight:700">· {savings}</span>' if savings else ""
            st.markdown(f"""
            <div class="pricing-card popular">
              <div class="popular-badge">⭐ Most Popular</div>
              <div class="plan-icon">⚡</div>
              <div class="plan-name">Pro</div>
              <div class="price-main" style="color:#4ade80">${pro_price}</div>
              <div class="price-period">/month · {pro_total} {savings_html}</div>
              <ul class="feature-list">
                <li class="included">Unlimited datasets</li>
                <li class="included">15+ algorithms</li>
                <li class="included">10-fold cross validation</li>
                <li class="included">XGBoost, LightGBM, CatBoost</li>
                <li class="included">Export trained model (.pkl)</li>
                <li class="included">50-entry history</li>
                <li class="included">Priority processing queue</li>
                <li class="not-included">API access</li>
              </ul>
            </div>""", unsafe_allow_html=True)
            if current_plan == "pro":
                st.markdown(f'<div style="text-align:center;padding:.75rem;background:rgba(74,222,128,0.10);border-radius:12px;font-weight:700;color:#4ade80;font-size:.875rem;margin-top:.5rem">✓ Current Plan</div>', unsafe_allow_html=True)
            else:
                if st.button("⚡ Upgrade to Pro", key="btn_upgrade_pro"):
                    st.session_state.upgrade_plan_selected = "pro"
                    st.rerun()

        # ENTERPRISE
        with pc3:
            ent_price      = PRICING["enterprise"]["annual_price"] if is_annual else PRICING["enterprise"]["monthly_price"]
            ent_total      = f"Billed ${PRICING['enterprise']['annual_total']}/year" if is_annual else "Billed monthly"
            savings_e      = f"Save ${PRICING['enterprise']['monthly_price']*12 - PRICING['enterprise']['annual_total']}/yr" if is_annual else ""
            savings_e_html = f'<span style="color:#c084fc;font-weight:700">· {savings_e}</span>' if savings_e else ""
            st.markdown(f"""
            <div class="pricing-card">
              <div class="plan-icon">🏢</div>
              <div class="plan-name">Enterprise</div>
              <div class="price-main" style="color:#c084fc">${ent_price}</div>
              <div class="price-period">/month · {ent_total} {savings_e_html}</div>
              <ul class="feature-list">
                <li class="included">Everything in Pro</li>
                <li class="included">Unlimited history</li>
                <li class="included">REST API access</li>
                <li class="included">Unlimited team members</li>
                <li class="included">Custom model pipelines</li>
                <li class="included">Dedicated support</li>
                <li class="included">SLA guarantee</li>
                <li class="included">On-premise deployment</li>
              </ul>
            </div>""", unsafe_allow_html=True)
            if current_plan == "enterprise":
                st.markdown(f'<div style="text-align:center;padding:.75rem;background:rgba(192,132,252,0.10);border-radius:12px;font-weight:700;color:#c084fc;font-size:.875rem;margin-top:.5rem">✓ Current Plan</div>', unsafe_allow_html=True)
            else:
                if st.button("🏢 Upgrade to Enterprise", key="btn_upgrade_ent"):
                    st.session_state.upgrade_plan_selected = "enterprise"
                    st.rerun()

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── PAYMENT FLOW ──
        if st.session_state.upgrade_plan_selected:
            selected_plan = st.session_state.upgrade_plan_selected
            is_ann = is_annual
            amount_usd = PRICING[selected_plan]["annual_price"] if is_ann else PRICING[selected_plan]["monthly_price"]
            amount_pkr = amount_usd * 280  # rough USD to PKR rate

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
                st.session_state.upgrade_plan_selected = None
                st.rerun()
            st.markdown("</div></div>", unsafe_allow_html=True)

            # ── Step 1: Choose Payment Method ──
            st.markdown(f"""
            <div style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin-bottom:.75rem">
              Step 1 — Choose Payment Method
            </div>""", unsafe_allow_html=True)

            pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
            pm_options = list(PAYMENT_METHODS.keys())
            pm_icons = {"easypaisa": "📱", "jazzcash": "💸", "bank_transfer": "🏦", "card": "💳"}
            pm_names = {"easypaisa": "EasyPaisa", "jazzcash": "JazzCash", "bank_transfer": "Bank Transfer", "card": "Debit/Credit Card"}

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
                        st.session_state.selected_pm = pm_key
                        st.rerun()

            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

            # ── Step 2: Payment Instructions ──
            pm = PAYMENT_METHODS[st.session_state.selected_pm]

            if st.session_state.selected_pm == "card":
                st.info("💳 Card payments coming soon! Please use EasyPaisa, JazzCash, or Bank Transfer for now.")
            else:
                st.markdown(f"""
                <div style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin-bottom:.75rem">
                  Step 2 — Send Payment
                </div>""", unsafe_allow_html=True)

                inst_col1, inst_col2 = st.columns([3, 2])
                with inst_col1:
                    # Account details
                    if st.session_state.selected_pm in ["easypaisa", "jazzcash"]:
                        st.markdown(f"""
                        <div class="account-box">
                          <div class="ab-label">Account Number</div>
                          <div class="ab-value">{pm['number']}</div>
                        </div>
                        <div class="account-box">
                          <div class="ab-label">Account Name</div>
                          <div class="ab-value">{pm['account_name']}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="account-box">
                          <div class="ab-label">Bank</div>
                          <div class="ab-value">{pm['bank']}</div>
                        </div>
                        <div class="account-box">
                          <div class="ab-label">Account Title</div>
                          <div class="ab-value">{pm['account_title']}</div>
                        </div>
                        <div class="account-box">
                          <div class="ab-label">Account Number</div>
                          <div class="ab-value">{pm['account_number']}</div>
                        </div>
                        <div class="account-box">
                          <div class="ab-label">IBAN</div>
                          <div class="ab-value" style="font-size:.85rem">{pm['iban']}</div>
                        </div>""", unsafe_allow_html=True)

                    # Amount to send
                    st.markdown(f"""
                    <div style="background:{"rgba(251,191,36,0.08)" if T=="dark" else "rgba(251,191,36,0.10)"};
                                border:1px solid rgba(251,191,36,0.35);border-radius:12px;padding:1rem;margin-top:.75rem">
                      <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{ACCENTY}">Amount to Send</div>
                      <div style="font-size:1.8rem;font-weight:900;color:{ACCENTY};font-family:'JetBrains Mono',monospace">
                        PKR {amount_pkr:,.0f}
                      </div>
                      <div style="font-size:.75rem;color:{TEXT3};margin-top:.2rem">
                        (${amount_usd}/mo · {'Annual' if is_ann else 'Monthly'} billing)
                      </div>
                    </div>""", unsafe_allow_html=True)

                with inst_col2:
                    st.markdown(f"""
                    <div class="instructions-box">
                      <div class="inst-title">📋 How to Pay</div>
                      <ol>""", unsafe_allow_html=True)
                    for step in pm["instructions"]:
                        st.markdown(f'<li style="padding:.4rem 0;font-size:.82rem;color:{TEXT2};line-height:1.5">{step}</li>', unsafe_allow_html=True)
                    st.markdown("</ol></div>", unsafe_allow_html=True)

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                # ── Step 3: Submit Transaction ID ──
                st.markdown(f"""
                <div style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em;color:{TEXT3};margin-bottom:.75rem">
                  Step 3 — Submit Transaction ID
                </div>""", unsafe_allow_html=True)

                sub_col1, sub_col2 = st.columns([3, 2])
                with sub_col1:
                    txn_input = st.text_input(
                        "Transaction / Reference ID",
                        placeholder="e.g. TXN123456789 or 12345678",
                        help="The Transaction ID / Reference Number from your payment confirmation",
                        key="txn_id_input"
                    )
                    txn_note = st.text_area(
                        "Additional notes (optional)",
                        placeholder="Any extra info about your payment...",
                        height=80,
                        key="txn_note_input"
                    )

                with sub_col2:
                    st.markdown(f"""
                    <div style="background:{BG3};border:1px solid {BORDER};border-radius:14px;padding:1.25rem;margin-top:1.5rem">
                      <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:.75rem">📦 Order Summary</div>
                      <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}">
                        <span style="font-size:.8rem;color:{TEXT2}">Plan</span>
                        <span style="font-size:.8rem;font-weight:700;color:{PRICING[selected_plan]['color']}">{selected_plan.title()}</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}">
                        <span style="font-size:.8rem;color:{TEXT2}">Billing</span>
                        <span style="font-size:.8rem;font-weight:700;color:{TEXT1}">{'Annual' if is_ann else 'Monthly'}</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}">
                        <span style="font-size:.8rem;color:{TEXT2}">Amount (USD)</span>
                        <span style="font-size:.8rem;font-weight:700;color:{TEXT1}">${amount_usd}/mo</span>
                      </div>
                      <div style="display:flex;justify-content:space-between;padding:.5rem 0">
                        <span style="font-size:.85rem;font-weight:700;color:{TEXT1}">Total (PKR)</span>
                        <span style="font-size:.9rem;font-weight:900;color:{ACCENTY}">PKR {amount_pkr:,.0f}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"✅ Submit Payment & Request Upgrade", key="submit_payment"):
                    if not txn_input.strip():
                        st.error("❌ Please enter your Transaction ID / Reference Number.")
                    else:
                        pay_id = save_payment_request(
                            email=uemail_global,
                            plan=selected_plan,
                            billing="annual" if is_ann else "monthly",
                            amount=amount_pkr,
                            payment_method=st.session_state.selected_pm,
                            txn_id=txn_input.strip(),
                            user_name=uname_global
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
You'll see your plan update automatically on next login.
                        """)
                        # Show email delivery status
                        if email_sent:
                            st.info("📧 Notification email sent to admin successfully.")
                        else:
                            # Read the error from log
                            email_log = load_json("dataforge_email_log.json")
                            last_entries = list(email_log.items())
                            last_err_msg = ""
                            for ts, entry in reversed(last_entries):
                                if "error" in entry:
                                    last_err_msg = entry.get("error", "")
                                    break
                            st.warning(
                                f"⚠️ **Admin email notification failed.** Your payment IS saved (ID: `{pay_id}`), "
                                f"but the admin email could not be sent.\n\n"
                                f"**Fix:** Gmail requires an **App Password** (not your regular password). "
                                f"Go to **myaccount.google.com → Security → 2-Step Verification → App passwords**, "
                                f"generate one for 'Mail', and set it as `SMTP_PASS` in your Streamlit secrets.\n\n"
                                + (f"**Error:** `{last_err_msg}`" if last_err_msg else "")
                            )
                        st.balloons()

            st.markdown("</div>", unsafe_allow_html=True)  # close payment flow div

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        # ── PAYMENT HISTORY ──
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>My Payment History</h3></div>""", unsafe_allow_html=True)

        user_payments = get_user_payments(uemail_global)
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
                pay_plan    = pay.get("plan","?").upper()
                pay_billing = pay.get("billing","monthly").title()
                pay_amount  = f"{pay.get('amount',0):,.0f}"
                pay_method  = pay.get("payment_method","?").replace("_"," ").title()
                pay_txn     = pay.get("txn_id","?")
                pay_date    = pay.get("submitted_at","")[:16]
                pay_id_disp = pay.get("id","")
                badge_bg,badge_bdr,badge_col = (
                    ("rgba(251,191,36,0.12)","rgba(251,191,36,0.40)","#fbbf24") if status=="pending" else
                    ("rgba(74,222,128,0.12)","rgba(74,222,128,0.40)","#4ade80") if status=="approved" else
                    ("rgba(248,113,113,0.12)","rgba(248,113,113,0.40)","#f87171")
                )
                # Build card as concatenated single-line strings — avoids Streamlit markdown/code-block parsing
                h  = f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem;margin-bottom:.75rem;position:relative;overflow:hidden">'
                h += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{plan_c},{badge_col})"></div>'
                h += f'<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-top:.2rem">'
                h += f'<div style="flex:1;min-width:180px">'
                h += f'<div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem">'
                h += f'<span style="font-size:.7rem;font-weight:800;color:{plan_c};background:{plan_c}22;border:1px solid {plan_c}55;border-radius:6px;padding:.15rem .5rem">{pay_plan}</span>'
                h += f'<span style="font-size:.7rem;color:{TEXT3}">{pay_billing} billing</span></div>'
                h += f'<div style="font-size:.88rem;font-weight:700;color:{TEXT1}">PKR {pay_amount} <span style="font-size:.75rem;font-weight:400;color:{TEXT2}">via {pay_method}</span></div>'
                h += f'<div style="font-size:.7rem;color:{TEXT3};margin-top:.2rem;font-family:monospace">Txn: {pay_txn}</div>'
                h += f'<div style="font-size:.67rem;color:{TEXT3};margin-top:.1rem">{pay_date}</div>'
                h += f'</div>'
                h += f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:.35rem">'
                h += f'<span style="display:inline-flex;align-items:center;gap:.35rem;padding:.35rem .85rem;border-radius:999px;font-size:.77rem;font-weight:800;background:{badge_bg};border:1px solid {badge_bdr};color:{badge_col}">{status_icon} {status.upper()}</span>'
                if pay.get("processed_at"):
                    h += f'<div style="font-size:.67rem;color:#6b7280">✓ Approved: {pay["processed_at"][:16]}</div>'
                h += f'<div style="font-size:.64rem;color:{TEXT3};font-family:monospace">{pay_id_disp}</div>'
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
        ]
        for q, a in faqs:
            with st.expander(f"❓ {q}"):
                st.markdown(f'<p style="color:{TEXT2};font-size:.9rem;line-height:1.7;margin:0">{a}</p>', unsafe_allow_html=True)

else:
    # ── WELCOME SCREEN — no interactive widgets (avoids duplicate key errors) ──
    st.markdown(f"""
    <div class="hero-wrap slide-up">
      <h1>Drop Your Data.<br>We Do the Rest.</h1>
      <p>DataForge ML Studio — zero-code AutoML. Upload a CSV, pick a target, hit train. Get a production model in minutes.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Feature cards
    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("🧬","Smart EDA",       "Correlation heatmaps, distribution explorer, scatter builder, and missing value charts."),
        ("⚡","AutoML Engine",   "15+ algorithms compared with k-fold cross-validation. Best model wins — automatically."),
        ("🎯","Smart Detection", "Auto-detects regression vs classification. Warns about ID columns. Quick data cleaning."),
        ("🏆","Rich Results",    "Trophy banner, radar + scatter + bar charts, metric breakdown, model export."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], feats):
        with col:
            st.markdown(f"""
            <div class="feature-card slide-up">
              <div class="fc-icon">{icon}</div>
              <h3>{title}</h3><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Static pricing preview — pure HTML, no Streamlit widgets (no key conflicts)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:1.5rem">
      <div style="font-size:1.4rem;font-weight:900;background:{HERO_H1_GRAD};
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text">💎 Plans & Pricing</div>
      <div style="font-size:.875rem;color:{TEXT2};margin-top:.3rem">
        Upgrade after loading a dataset · Use the 💳 Upgrade tab
      </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem;margin-bottom:2rem">

      <!-- FREE -->
      <div style="background:{CARD_BG};border:2px solid {BORDER};border-radius:20px;padding:1.75rem">
        <div style="font-size:2rem;margin-bottom:.5rem">🌱</div>
        <div style="font-size:1.1rem;font-weight:900;color:{TEXT1}">Free</div>
        <div style="font-size:2.5rem;font-weight:900;color:{TEXT1};margin:.5rem 0 .25rem">$0</div>
        <div style="font-size:.8rem;color:{TEXT3};margin-bottom:1.25rem">forever free</div>
        <div style="font-size:.82rem;color:{TEXT2};line-height:2">
          ✓ 3 datasets/month<br>✓ 5 basic algorithms<br>✓ 3-fold CV<br>
          <span style="color:{TEXT3}">✗ XGBoost / LightGBM<br>✗ Model export (.pkl)<br>✗ API access</span>
        </div>
        <div style="margin-top:1.25rem;padding:.6rem;background:rgba(107,114,128,0.12);
                    border-radius:10px;text-align:center;font-size:.8rem;font-weight:700;color:#9ca3af">
          ✓ Your Current Plan
        </div>
      </div>

      <!-- PRO -->
      <div style="background:{CARD_BG};border:2px solid #4ade80;border-radius:20px;padding:1.75rem;
                  position:relative;box-shadow:0 0 28px rgba(74,222,128,0.12)">
        <div style="position:absolute;top:.9rem;right:.9rem;background:linear-gradient(135deg,#16a34a,#22c55e);
                    color:white;font-size:.62rem;font-weight:800;padding:.2rem .6rem;
                    border-radius:999px;text-transform:uppercase;letter-spacing:.05em">⭐ Most Popular</div>
        <div style="font-size:2rem;margin-bottom:.5rem">⚡</div>
        <div style="font-size:1.1rem;font-weight:900;color:{TEXT1}">Pro</div>
        <div style="font-size:2.5rem;font-weight:900;color:#4ade80;margin:.5rem 0 .25rem">$19<span style="font-size:1rem;font-weight:400;color:{TEXT3}">/mo</span></div>
        <div style="font-size:.8rem;color:{TEXT3};margin-bottom:1.25rem">or $15/mo billed annually</div>
        <div style="font-size:.82rem;color:{TEXT2};line-height:2">
          ✓ Unlimited datasets<br>✓ 15+ algorithms<br>✓ 10-fold CV<br>
          ✓ XGBoost, LightGBM, CatBoost<br>✓ Export model (.pkl)<br>✓ 50-entry history
        </div>
        <div style="margin-top:1.25rem;padding:.6rem;
                    background:linear-gradient(135deg,rgba(74,222,128,0.15),rgba(74,222,128,0.08));
                    border:1px solid rgba(74,222,128,0.35);border-radius:10px;
                    text-align:center;font-size:.82rem;font-weight:800;color:#4ade80">
          ⚡ Load a dataset → 💳 Upgrade tab
        </div>
      </div>

      <!-- ENTERPRISE -->
      <div style="background:{CARD_BG};border:2px solid {BORDER};border-radius:20px;padding:1.75rem">
        <div style="font-size:2rem;margin-bottom:.5rem">🏢</div>
        <div style="font-size:1.1rem;font-weight:900;color:{TEXT1}">Enterprise</div>
        <div style="font-size:2.5rem;font-weight:900;color:#c084fc;margin:.5rem 0 .25rem">$79<span style="font-size:1rem;font-weight:400;color:{TEXT3}">/mo</span></div>
        <div style="font-size:.8rem;color:{TEXT3};margin-bottom:1.25rem">or $63/mo billed annually</div>
        <div style="font-size:.82rem;color:{TEXT2};line-height:2">
          ✓ Everything in Pro<br>✓ Unlimited history<br>✓ REST API access<br>
          ✓ Unlimited team members<br>✓ Dedicated support<br>✓ SLA guarantee
        </div>
        <div style="margin-top:1.25rem;padding:.6rem;
                    background:rgba(192,132,252,0.10);border:1px solid rgba(192,132,252,0.30);
                    border-radius:10px;text-align:center;font-size:.82rem;font-weight:800;color:#c084fc">
          🏢 Load a dataset → 💳 Upgrade tab
        </div>
      </div>
    </div>
    <div style="text-align:center;color:{TEXT3};font-size:.82rem;padding-bottom:1.5rem">
      👈 Upload a CSV/Excel or load a sample dataset from the sidebar to get started
    </div>
    """, unsafe_allow_html=True)
