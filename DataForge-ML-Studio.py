# %% [code]
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pycaret.classification import (
    setup as clf_setup, 
    compare_models as clf_compare,
    pull as clf_pull, 
    save_model as clf_save,
)
from pycaret.regression import (
    setup as reg_setup, 
    compare_models as reg_compare,
    pull as reg_pull, 
    save_model as reg_save,
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
TOKENS_FILE  = "dataforge_tokens.json"   # persists login across refreshes

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
#  PERSISTENT LOGIN — TOKEN SYSTEM
#  Token stored in browser via st.query_params,
#  validated against tokens file on every load.
# ─────────────────────────────────────────────
TOKEN_EXPIRY_DAYS = 30

def create_token(email: str) -> str:
    """Generate a secure random token, save it with expiry."""
    token = secrets.token_urlsafe(32)
    tokens = load_json(TOKENS_FILE)
    expiry = (datetime.now() + timedelta(days=TOKEN_EXPIRY_DAYS)).strftime("%Y-%m-%d")
    tokens[token] = {"email": email, "expiry": expiry}
    # Clean expired tokens while we're here
    today = datetime.now().strftime("%Y-%m-%d")
    tokens = {t: v for t, v in tokens.items() if v.get("expiry", "9999") >= today}
    tokens[token] = {"email": email, "expiry": expiry}
    save_json(TOKENS_FILE, tokens)
    return token

def validate_token(token: str):
    """Returns email if token is valid and not expired, else None."""
    if not token:
        return None
    tokens = load_json(TOKENS_FILE)
    entry = tokens.get(token)
    if not entry:
        return None
    today = datetime.now().strftime("%Y-%m-%d")
    if entry.get("expiry", "0000") < today:
        # Expired — remove it
        tokens.pop(token, None)
        save_json(TOKENS_FILE, tokens)
        return None
    return entry.get("email")

def delete_token(token: str):
    """Remove token on logout."""
    tokens = load_json(TOKENS_FILE)
    tokens.pop(token, None)
    save_json(TOKENS_FILE, tokens)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ─────────────────────────────────────────────
#  PLAN LIMITS — SINGLE SOURCE OF TRUTH
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
    """Returns 'free', 'pro', or 'enterprise'"""
    users_db = load_json(USERS_FILE)
    user = users_db.get(email, {})
    plan = user.get("plan", "free")
    # Check if pro plan has expired (if expiry date set)
    plan_expiry = user.get("plan_expiry", None)
    if plan_expiry and plan != "free":
        try:
            expiry_dt = datetime.strptime(plan_expiry, "%Y-%m-%d")
            if datetime.now() > expiry_dt:
                # Downgrade to free
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

# ✅ SIMPLE VERSION - Always allow training
def can_train(email: str) -> tuple[bool, str]:
    """Always allow training - no monthly limits"""
    return True, ""

def upgrade_user_plan(email: str, plan: str, months: int = 1):
    """Set user plan with expiry"""
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
#  EMAIL NOTIFICATION
# ─────────────────────────────────────────────
NOTIFY_TO   = "shayan.corner@gmail.com"
try:
    SMTP_USER = st.secrets["SMTP_USER"]
    SMTP_PASS = st.secrets["SMTP_PASS"]
except Exception:
    SMTP_USER = "shayan.corner@gmail.com"
    SMTP_PASS = "TqBfJoTlD@2012"

def send_email(subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        log = load_json("dataforge_email_log.json")
        log[now_str()] = {"subject": subject, "body": body}
        save_json("dataforge_email_log.json", log)
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SMTP_USER
        msg["To"]      = NOTIFY_TO
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, NOTIFY_TO, msg.as_string())
        return True
    except Exception as e:
        log = load_json("dataforge_email_log.json")
        log[now_str()] = {"subject": subject, "body": body, "error": str(e)}
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
 Total Logins     : {login_count + 1}
 Datasets Trained : {u_hist.get('datasets_trained', 0)}
 Last Seen        : {u_hist.get('last_login', 'First time')}
══════════════════════════════════════════"""
    send_email(subject, body)

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
#  SESSION STATE + AUTO-LOGIN FROM PERSISTED TOKEN
# ─────────────────────────────────────────────
for k in ["data","problem_type","best_model","results","training_time","dataset_name","cv_fold"]:
    if k not in st.session_state:
        st.session_state[k] = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "signin"

# ── Auto-login: check token in query params ──────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.current_user  = None
    st.session_state.login_token   = None

    # Try to restore session from ?token=... in URL
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
        pass  # query_params unavailable or malformed — stay logged out

T = st.session_state.theme

# ─────────────────────────────────────────────
#  THEME PALETTES
# ─────────────────────────────────────────────
if T == "dark":
    BG = "#000000"; BG2 = "#0d0d0d"; BG3 = "#141414"; BG4 = "#1c1c1c"
    BORDER = "#222222"; TEXT1 = "#f9fafb"; TEXT2 = "#9ca3af"; TEXT3 = "#6b7280"
    ACCENT1 = "#4ade80"; ACCENT2 = "#60a5fa"; ACCENT3 = "#c084fc"
    ACCENTR = "#f87171"; ACCENTY = "#fbbf24"
    SIDEBAR = "#0a0a0a"
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
    SIDEBAR = "#1e0a3c"
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
#MainMenu{{visibility:hidden;}}
footer{{visibility:hidden;}}
.block-container{{padding-top:1.5rem !important;max-width:1400px;}}
html,body{{background:{BG} !important;color:{TEXT1} !important;}}
.main,.block-container,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{{background:{BG} !important;}}
section[data-testid="stSidebar"]{{background:{"linear-gradient(180deg,#0a0a0a 0%,#111111 100%)" if T=="dark" else "#f7f7fb"} !important;border-right:{"1px solid #222222" if T=="dark" else "2px solid #d0d0d0"} !important;box-shadow:{"4px 0 20px rgba(0,0,0,0.6)" if T=="dark" else "4px 0 16px rgba(0,0,0,0.10)"} !important;min-height:100vh !important;}}
section[data-testid="stSidebar"]>div{{background:{"transparent" if T=="dark" else "#f7f7fb"} !important;}}
section[data-testid="stSidebar"] *{{color:{"#d1fae5" if T=="dark" else "#111111"} !important;}}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{{color:{"#f9fafb" if T=="dark" else "#111111"} !important;}}
section[data-testid="stSidebar"] .stButton>button{{background:{"linear-gradient(135deg,#16a34a,#22c55e)" if T=="dark" else "linear-gradient(135deg,#5b21b6,#7c3aed)"} !important;color:#ffffff !important;box-shadow:{"0 4px 14px rgba(74,222,128,0.35)" if T=="dark" else "0 4px 14px rgba(124,58,237,0.35)"} !important;}}

/* ── FILE UPLOADER — DARK THEME FIX ── */
[data-testid="stFileUploader"] {{
  background:{"#0f0f0f" if T=="dark" else "#f3eeff"} !important;
  border-radius:14px !important;
  padding:4px !important;
}}
[data-testid="stFileUploader"] > div {{
  background:{"#111111" if T=="dark" else "#ede9fe"} !important;
  border:2px dashed {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.40)"} !important;
  border-radius:12px !important;
}}
[data-testid="stFileUploader"] > div:hover {{
  border-color:{"rgba(74,222,128,0.70)" if T=="dark" else "rgba(124,58,237,0.70)"} !important;
  background:{"#161616" if T=="dark" else "#e8e0ff"} !important;
}}
[data-testid="stFileUploader"] label, 
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{
  color:{"#6b7280" if T=="dark" else "#7c3aed"} !important;
  background:transparent !important;
}}
[data-testid="stFileUploader"] small {{
  color:{"#4b5563" if T=="dark" else "#9ca3af"} !important;
}}
[data-testid="stFileUploaderDropzone"] {{
  background:{"#0f0f0f" if T=="dark" else "#ede9fe"} !important;
  border:2px dashed {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.40)"} !important;
  border-radius:12px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
  background:{"#141414" if T=="dark" else "#e0d9ff"} !important;
  border-color:{"rgba(74,222,128,0.65)" if T=="dark" else "rgba(124,58,237,0.65)"} !important;
}}
[data-testid="stFileUploaderDropzone"] * {{
  color:{"#6b7280" if T=="dark" else "#7c3aed"} !important;
  background:transparent !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
  background:{"#1a1a1a" if T=="dark" else "rgba(124,58,237,0.10)"} !important;
  border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(124,58,237,0.45)"} !important;
  color:{"#4ade80" if T=="dark" else "#5b21b6"} !important;
  border-radius:8px !important;
  font-weight:600 !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover {{
  background:{"rgba(74,222,128,0.10)" if T=="dark" else "rgba(124,58,237,0.18)"} !important;
  border-color:{"rgba(74,222,128,0.60)" if T=="dark" else "rgba(124,58,237,0.70)"} !important;
}}
.vibe-header{{position:relative;padding:2.5rem 3rem;border-radius:20px;margin-bottom:2rem;overflow:hidden;background:{HDR_BG};border:1px solid {HDR_BORDER};box-shadow:{"0 0 40px rgba(74,222,128,0.12)" if T=="dark" else "0 8px 40px rgba(124,58,237,0.30)"}}}
.vibe-header::before{{content:'';position:absolute;inset:0;background:{"radial-gradient(ellipse 80% 60% at 10% 50%,rgba(74,222,128,0.15) 0%,transparent 60%),radial-gradient(ellipse 60% 80% at 90% 30%,rgba(96,165,250,0.10) 0%,transparent 60%)" if T=="dark" else "radial-gradient(ellipse 80% 60% at 10% 50%,rgba(167,139,250,0.25) 0%,transparent 60%),radial-gradient(ellipse 60% 80% at 90% 30%,rgba(196,181,253,0.20) 0%,transparent 60%)"};animation:pulseGlow 6s ease-in-out infinite alternate;}}
@keyframes pulseGlow{{from{{opacity:.6}}to{{opacity:1}}}}
.vibe-header .grid-overlay{{position:absolute;inset:0;background-image:{"linear-gradient(rgba(74,222,128,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(74,222,128,0.04) 1px,transparent 1px)" if T=="dark" else "linear-gradient(rgba(167,139,250,0.10) 1px,transparent 1px),linear-gradient(90deg,rgba(167,139,250,0.10) 1px,transparent 1px)"};background-size:40px 40px;}}
.vibe-header-content{{position:relative;z-index:2;}}
.vibe-header h1{{font-size:2.8rem;font-weight:900;margin:0;letter-spacing:-.03em;line-height:1.1;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}}
.vibe-header .tagline{{font-size:1rem;margin-top:.6rem;font-weight:400;color:{"#a7f3d0" if T=="dark" else "#ddd6fe"};}}
.vibe-header .pill-row{{display:flex;gap:.5rem;margin-top:1rem;flex-wrap:wrap;}}
.pill{{display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .8rem;border-radius:999px;font-size:.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;}}
.pill-green{{background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(167,139,250,0.20)"};color:{"#4ade80" if T=="dark" else "#ddd6fe"};border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(167,139,250,0.40)"};}}
.pill-blue{{background:{"rgba(96,165,250,0.12)" if T=="dark" else "rgba(196,181,253,0.20)"};color:{"#60a5fa" if T=="dark" else "#c4b5fd"};border:1px solid {"rgba(96,165,250,0.30)" if T=="dark" else "rgba(196,181,253,0.40)"};}}
.pill-purple{{background:{"rgba(192,132,252,0.12)" if T=="dark" else "rgba(216,180,254,0.20)"};color:{"#c084fc" if T=="dark" else "#e9d5ff"};border:1px solid {"rgba(192,132,252,0.30)" if T=="dark" else "rgba(216,180,254,0.40)"};}}
.stat-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-bottom:1.5rem;}}
.stat-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:16px;padding:1.25rem 1.5rem;position:relative;overflow:hidden;cursor:default;box-shadow:{"none" if T=="dark" else "0 2px 12px rgba(124,58,237,0.08)"}}}
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
.feature-card{{background:{CARD_BG};border:1px solid {BORDER};border-radius:18px;padding:1.75rem;position:relative;overflow:hidden;box-shadow:{"none" if T=="dark" else "0 2px 12px rgba(124,58,237,0.07)"}}}
.feature-card:hover{{border-color:{ACCENT1};transform:translateY(-3px);box-shadow:{"0 0 28px rgba(74,222,128,0.18)" if T=="dark" else "0 10px 32px rgba(124,58,237,0.18)"}}}
.feature-card .fc-icon{{font-size:2.2rem;margin-bottom:.75rem;}}
.feature-card h3{{margin:0 0 .5rem;font-size:1rem;font-weight:700;color:{TEXT1};}}
.feature-card p{{margin:0;font-size:.875rem;color:{TEXT2};line-height:1.6;}}
.sidebar-section{{background:{"rgba(255,255,255,0.05)" if T=="dark" else "#ffffff"};border:1px solid {"rgba(255,255,255,0.08)" if T=="dark" else "#dddddd"};border-radius:12px;padding:1rem 1.25rem;margin-bottom:1rem;box-shadow:{"none" if T=="dark" else "0 1px 4px rgba(0,0,0,0.06)"}}}
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
.step-item.active::before{{background:linear-gradient({ACCENT1},{BORDER});}}
.step-dot{{width:34px;height:34px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:.85rem;font-weight:700;border:2px solid {BORDER};background:{BG3};color:{TEXT3};}}
.step-dot.done{{border-color:{ACCENT1};background:{"rgba(74,222,128,0.15)" if T=="dark" else "rgba(124,58,237,0.12)"};color:{ACCENT1};}}
.step-dot.active{{border-color:{ACCENT2};background:{"rgba(96,165,250,0.15)" if T=="dark" else "rgba(37,99,235,0.12)"};color:{ACCENT2};animation:pulse 1.5s infinite;}}
@keyframes pulse{{0%,100%{{box-shadow:0 0 0 0 {"rgba(96,165,250,0.4)" if T=="dark" else "rgba(37,99,235,0.4)"}}}50%{{box-shadow:0 0 0 8px transparent}}}}
.step-label{{font-size:.9rem;font-weight:600;color:{TEXT1};padding-top:.4rem;}}
.step-sub{{font-size:.77rem;color:{TEXT3};}}
.trophy-banner{{border-radius:20px;padding:2rem 2.5rem;background:{"linear-gradient(135deg,rgba(74,222,128,0.08),rgba(96,165,250,0.05))" if T=="dark" else "linear-gradient(135deg,#1e0a3c,#4c1d95)"};border:1px solid {"rgba(74,222,128,0.35)" if T=="dark" else "rgba(167,139,250,0.4)"};box-shadow:{"0 0 32px rgba(74,222,128,0.12)" if T=="dark" else "0 8px 32px rgba(124,58,237,0.30)"};display:flex;align-items:center;gap:1.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}}
.trophy-banner::before{{content:'';position:absolute;top:-40%;right:-5%;width:280px;height:280px;border-radius:50%;background:{"radial-gradient(circle,rgba(74,222,128,0.10) 0%,transparent 70%)" if T=="dark" else "radial-gradient(circle,rgba(196,181,253,0.25) 0%,transparent 70%)"}}}
.trophy-icon{{font-size:3.5rem;flex-shrink:0;position:relative;z-index:1;}}
.trophy-text{{position:relative;z-index:1;}}
.trophy-text h2{{margin:0;font-size:1.6rem;font-weight:900;color:{"#f9fafb" if T=="dark" else "#f5f3ff"};}}
.trophy-text p{{margin:.25rem 0 0;font-size:.9rem;color:{"#9ca3af" if T=="dark" else "#c4b5fd"};}}
.trophy-score{{margin-left:auto;text-align:right;flex-shrink:0;position:relative;z-index:1;padding:.75rem 1.5rem;background:{"rgba(74,222,128,0.12)" if T=="dark" else "rgba(167,139,250,0.18)"};border-radius:14px;border:1px solid {"rgba(74,222,128,0.25)" if T=="dark" else "rgba(167,139,250,0.35)"}}}
.trophy-score .ts-label{{font-size:.68rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{"#6b7280" if T=="dark" else "#c4b5fd"};}}
.trophy-score .ts-value{{font-size:2.2rem;font-weight:900;color:{"#4ade80" if T=="dark" else "#e9d5ff"};font-variant-numeric:tabular-nums;}}
.glow-divider{{height:1px;margin:1.5rem 0;background:{GLOW_DIV};opacity:{"0.4" if T=="dark" else "0.3"};}}
.insight-chip{{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .8rem;border-radius:8px;font-size:.8rem;font-weight:600;margin:.25rem;background:{BG3};border:1px solid {BORDER};color:{TEXT2 if T=="dark" else "#444444"};}}
.stTabs [data-baseweb="tab-list"]{{background:{CARD_BG} !important;border:1px solid {BORDER} !important;border-radius:14px !important;padding:6px !important;gap:4px !important;box-shadow:{"none" if T=="dark" else "0 2px 8px rgba(124,58,237,0.10)"} !important;}}
.stTabs [data-baseweb="tab"]{{border-radius:10px !important;font-weight:600 !important;border:none !important;color:{TEXT2} !important;background:transparent !important;padding:.65rem 1.4rem !important;font-size:.9rem !important;}}
.stTabs [data-baseweb="tab"]:hover{{background:{BG3} !important;color:{TEXT1} !important;}}
.stTabs [aria-selected="true"]{{background:{TAB_SEL} !important;color:#fff !important;box-shadow:0 4px 14px {BTN_GLOW} !important;}}
.stSelectbox>div>div{{background:{BG3} !important;border:1px solid {BORDER} !important;border-radius:10px !important;color:{TEXT1} !important;}}
.stSelectbox>div>div:hover{{border-color:{ACCENT1} !important;}}
.stTextInput>div>div>input{{background:{BG3} !important;border:1px solid {BORDER} !important;border-radius:10px !important;color:{TEXT1} !important;}}
.stTextInput>div>div>input:focus{{border-color:{ACCENT1} !important;box-shadow:0 0 0 3px {"rgba(74,222,128,0.15)" if T=="dark" else "rgba(124,58,237,0.15)"} !important;}}
.streamlit-expanderHeader{{background:{CARD_BG} !important;border:1px solid {BORDER} !important;border-radius:10px !important;font-weight:600 !important;color:{TEXT1} !important;}}
.streamlit-expanderHeader:hover{{border-color:{ACCENT1} !important;}}
.stButton>button{{background:{BTN_BG} !important;color:#fff !important;border:none !important;padding:.8rem 1.75rem !important;font-weight:700 !important;font-size:.95rem !important;border-radius:12px !important;box-shadow:0 4px 16px {BTN_GLOW} !important;letter-spacing:.02em !important;}}
.stButton>button:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 24px {BTN_GLOW} !important;filter:brightness(1.08) !important;}}
.stButton>button:active{{transform:translateY(0) !important;}}
.stDownloadButton>button{{background:{BG3} !important;color:{TEXT1} !important;border:1px solid {BORDER} !important;border-radius:12px !important;font-weight:600 !important;}}
.stDownloadButton>button:hover{{border-color:{ACCENT1} !important;color:{ACCENT1} !important;background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"} !important;}}

/* ── Theme toggle button — same neutral style as Logout ── */
div[data-testid="stButton"]:has(button[kind="secondary"]#theme_btn) button,
button[key="theme_btn"],
[data-testid="stButton"] button[kind="secondary"] {{}}
div[data-testid="column"]:nth-child(2) .stButton>button {{
  background:{BG2 if T=="dark" else "#ffffff"} !important;
  color:{TEXT1} !important;
  border:1px solid {BORDER} !important;
  box-shadow:{"0 2px 8px rgba(0,0,0,0.4)" if T=="dark" else "0 2px 8px rgba(0,0,0,0.12)"} !important;
  font-weight:600 !important;
  min-width:90px !important;
}}
div[data-testid="column"]:nth-child(2) .stButton>button:hover {{
  border-color:{ACCENT1} !important;
  color:{ACCENT1} !important;
  background:{"rgba(74,222,128,0.06)" if T=="dark" else "rgba(124,58,237,0.06)"} !important;
  transform:translateY(-2px) !important;
  box-shadow:{"0 4px 14px rgba(74,222,128,0.2)" if T=="dark" else "0 4px 14px rgba(124,58,237,0.2)"} !important;
}}
div[data-testid="column"]:nth-child(3) .stButton>button {{
  background:{BG2 if T=="dark" else "#ffffff"} !important;
  color:{TEXT1} !important;
  border:1px solid {BORDER} !important;
  box-shadow:{"0 2px 8px rgba(0,0,0,0.4)" if T=="dark" else "0 2px 8px rgba(0,0,0,0.12)"} !important;
  font-weight:600 !important;
  min-width:90px !important;
}}
div[data-testid="column"]:nth-child(3) .stButton>button:hover {{
  border-color:{ACCENTR} !important;
  color:{ACCENTR} !important;
  background:rgba(248,113,113,0.06) !important;
  transform:translateY(-2px) !important;
  box-shadow:0 4px 14px rgba(248,113,113,0.2) !important;
}}
.hero-wrap{{min-height:45vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:4rem 2rem;position:relative;}}
.hero-wrap h1{{font-size:3.8rem;font-weight:900;letter-spacing:-.04em;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0 0 1rem;line-height:1.05;position:relative;}}
.hero-wrap p{{font-size:1.1rem;color:{TEXT2};max-width:560px;line-height:1.7;margin:0 0 2rem;position:relative;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BG4};border-radius:3px;}}
::-webkit-scrollbar-thumb:hover{{background:{ACCENT1};}}
.dataframe{{font-family:'JetBrains Mono',monospace !important;font-size:.82rem !important;}}
[data-testid="stDataFrame"]{{border-radius:12px !important;overflow:hidden;}}
.stAlert{{border-radius:12px !important;border-left-width:4px !important;background:{CARD_BG} !important;}}
@keyframes slideUp{{from{{opacity:0;transform:translateY(18px)}}to{{opacity:1;transform:none}}}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
.slide-up{{animation:slideUp .45s ease-out both;}}
.fade-in{{animation:fadeIn .5s ease-out both;}}

/* PLAN BADGE */
.plan-badge{{display:inline-flex;align-items:center;gap:.35rem;padding:.3rem .85rem;border-radius:999px;font-size:.72rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;}}
.plan-badge.free{{background:rgba(107,114,128,0.15);color:#9ca3af;border:1px solid rgba(107,114,128,0.3);}}
.plan-badge.pro{{background:rgba(74,222,128,0.15);color:#4ade80;border:1px solid rgba(74,222,128,0.4);}}
.plan-badge.enterprise{{background:rgba(192,132,252,0.15);color:#c084fc;border:1px solid rgba(192,132,252,0.4);}}

/* UPGRADE WALL */
.upgrade-wall{{background:{"rgba(251,191,36,0.06)" if T=="dark" else "rgba(251,191,36,0.08)"};border:2px dashed {"rgba(251,191,36,0.40)" if T=="dark" else "rgba(251,191,36,0.50)"};border-radius:16px;padding:2rem;text-align:center;margin:1rem 0;}}

/* AUTH */
.auth-overlay{{position:fixed;inset:0;z-index:9999;background:{"rgba(0,0,0,0.97)" if T=="dark" else "rgba(248,244,255,0.98)"};display:flex;align-items:center;justify-content:center;backdrop-filter:blur(12px);}}

/* ── SIDEBAR NATIVE TOGGLE — MAKE VISIBLE ── */
[data-testid="collapsedControl"],
button[data-testid="collapsedControl"] {{
  display:flex !important;
  visibility:visible !important;
  opacity:1 !important;
  pointer-events:all !important;
  background:{"#16a34a" if T=="dark" else "#5b21b6"} !important;
  border-radius:0 10px 10px 0 !important;
  min-width:24px !important;
  width:24px !important;
  box-shadow:{"3px 0 12px rgba(74,222,128,0.5)" if T=="dark" else "3px 0 12px rgba(124,58,237,0.5)"} !important;
}}
[data-testid="collapsedControl"] svg,
button[data-testid="collapsedControl"] svg {{
  color:#ffffff !important;
  fill:#ffffff !important;
}}

/* Sidebar internal collapse button */
[data-testid="stSidebarCollapseButton"] button,
button[kind="header"] {{
  visibility:visible !important;
  opacity:1 !important;
  background:{"rgba(74,222,128,0.12)" if T=="dark" else "rgba(124,58,237,0.12)"} !important;
  border:1px solid {"rgba(74,222,128,0.3)" if T=="dark" else "rgba(124,58,237,0.3)"} !important;
  border-radius:8px !important;
  color:{"#4ade80" if T=="dark" else "#7c3aed"} !important;
}}

/* Floating open-sidebar button (custom) */
#sidebar-open-btn {{
  position:fixed;
  left:0;
  top:50%;
  transform:translateY(-50%);
  z-index:999999;
  background:{"linear-gradient(180deg,#16a34a,#15803d)" if T=="dark" else "linear-gradient(180deg,#5b21b6,#4c1d95)"};
  border:none;
  border-radius:0 12px 12px 0;
  width:30px;
  height:64px;
  cursor:pointer;
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow:{"4px 0 20px rgba(74,222,128,0.6)" if T=="dark" else "4px 0 20px rgba(124,58,237,0.6)"};
  transition:width 0.2s ease, box-shadow 0.2s ease;
}}
#sidebar-open-btn:hover {{
  width:40px;
  box-shadow:{"6px 0 28px rgba(74,222,128,0.8)" if T=="dark" else "6px 0 28px rgba(124,58,237,0.8)"};
}}
#sidebar-open-btn svg {{
  width:16px;
  height:16px;
  fill:white;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  AUTH SCREEN
# ─────────────────────────────────────────────
if not st.session_state.authenticated:
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"]{{display:none !important;}}
    .block-container{{padding:0 !important;max-width:100% !important;}}
    [data-testid="stAppViewBlockContainer"]{{display:flex;align-items:center;justify-content:center;min-height:100vh;background:{BG} !important;}}
    </style>
    """, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1.2, 1])
    with center_col:
        st.markdown(f"""
        <div style="padding:3rem 0;position:relative;z-index:1">
          <div style="text-align:center;margin-bottom:2rem">
            <div style="font-size:3.5rem;margin-bottom:.5rem">⚡</div>
            <h1 style="font-size:2rem;font-weight:900;margin:0 0 .25rem;background:{HERO_H1_GRAD};-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">DataForge ML Studio</h1>
            <p style="color:{TEXT3};font-size:.9rem;margin:0">AutoML that actually vibes</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            signin_active = st.session_state.auth_mode == "signin"
            if st.button("🔑 Sign In", width="stretch", key="tab_signin",
                         type="primary" if signin_active else "secondary"):
                st.session_state.auth_mode = "signin"; st.rerun()
        with mode_col2:
            signup_active = st.session_state.auth_mode == "signup"
            if st.button("✨ Create Account", width="stretch", key="tab_signup",
                         type="primary" if signup_active else "secondary"):
                st.session_state.auth_mode = "signup"; st.rerun()

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        if st.session_state.auth_mode == "signup":
            su_name  = st.text_input("Full Name", placeholder="Enter your full name", key="su_name")
            su_email = st.text_input("Email Address", placeholder="your@email.com", key="su_email")
            su_pass  = st.text_input("Password", type="password", placeholder="Create a strong password", key="su_pass")
            su_pass2 = st.text_input("Confirm Password", type="password", placeholder="Repeat your password", key="su_pass2")

            if st.button("🚀 Create Account & Enter Studio", width="stretch", key="do_signup"):
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
                        st.error("❌ An account with this email already exists. Please Sign In.")
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
                        # Create persistent token so refresh doesn't log out
                        token = create_token(su_email)
                        st.session_state.login_token = token
                        try:
                            st.query_params["token"] = token
                        except Exception:
                            pass
                        st.success(f"✅ Welcome to DataForge, {su_name}!")
                        time.sleep(0.8); st.rerun()
        else:
            si_email = st.text_input("Email Address", placeholder="your@email.com", key="si_email")
            si_pass  = st.text_input("Password", type="password", placeholder="Enter your password", key="si_pass")

            if st.button("⚡ Sign In & Launch Studio", width="stretch", key="do_signin"):
                if not si_email or not si_pass:
                    st.error("❌ Please enter your email and password.")
                elif "@" not in si_email:
                    st.error("❌ Please enter a valid email address.")
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
                        # Create persistent token so refresh doesn't log out
                        token = create_token(si_email)
                        st.session_state.login_token = token
                        try:
                            st.query_params["token"] = token
                        except Exception:
                            pass
                        st.success(f"✅ Welcome back, {udata['name']}!")
                        time.sleep(0.8); st.rerun()

        st.markdown(f'<div style="text-align:center;margin-top:1.5rem;font-size:.78rem;color:{TEXT3}">🔒 Your information is stored securely on the server</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  GET CURRENT USER PLAN (after auth)
# ─────────────────────────────────────────────
uemail_global = st.session_state.current_user.get("email","") if st.session_state.current_user else ""
uname_global  = st.session_state.current_user.get("name","User") if st.session_state.current_user else "User"
current_plan  = get_user_plan(uemail_global)
plan_limits   = PLAN_LIMITS[current_plan]

PLAN_COLORS = {"free": "#6b7280", "pro": "#4ade80", "enterprise": "#c084fc"}
PLAN_ICONS  = {"free": "🌱", "pro": "⚡", "enterprise": "🏢"}
plan_color  = PLAN_COLORS.get(current_plan, "#6b7280")
plan_icon   = PLAN_ICONS.get(current_plan, "🌱")

# ─────────────────────────────────────────────
#  ANIMATED HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="vibe-header slide-up">
  <div class="grid-overlay"></div>
  <div class="vibe-header-content">
    <h1>⚡ DataForge ML Studio</h1>
    <p class="tagline">Drop your data. We handle the rest — AutoML that actually vibes.</p>
    <div class="pill-row">
      <span class="pill pill-green">✦ AutoML</span>
      <span class="pill pill-blue">⚡ 15+ Algorithms</span>
      <span class="pill pill-purple">◈ Smart Detect</span>
      <span class="pill pill-green">↗ Production Ready</span>
      <span class="pill pill-blue">◎ Cross Validation</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  THEME TOGGLE + USER INFO + LOGOUT
# ─────────────────────────────────────────────
_tcol1, _tcol2, _tcol3 = st.columns([9, 1, 1])
with _tcol2:
    is_dark = T == "dark"
    if st.button("⬜ White" if is_dark else "⬛ Black", key="theme_btn"):
        st.session_state.theme = "light" if is_dark else "dark"; st.rerun()
with _tcol3:
    if st.button("🚪 Logout", key="logout_btn", help=f"Logged in as {uname_global}"):
        # Delete persisted token so refresh doesn't auto-login
        tok = st.session_state.get("login_token")
        if tok:
            delete_token(tok)
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.login_token = None
        st.rerun()

# ── FLOATING SIDEBAR RESTORE BUTTON ────────────────────────────────
# Appears on the left edge when sidebar is collapsed — click to reopen
st.markdown(f"""
<button id="sidebar-open-btn" title="Open Sidebar"
  style="position:fixed;left:0;top:50%;transform:translateY(-50%);z-index:999999;
    background:{"linear-gradient(180deg,#16a34a,#15803d)" if T=="dark" else "linear-gradient(180deg,#5b21b6,#4c1d95)"};
    border:none;border-radius:0 12px 12px 0;width:32px;height:64px;
    cursor:pointer;display:none;align-items:center;justify-content:center;
    box-shadow:{"5px 0 20px rgba(74,222,128,0.7)" if T=="dark" else "5px 0 20px rgba(124,58,237,0.7)"};"
  onclick="
    var found = false;
    var selectors = [
      '[data-testid=collapsedControl]',
      'button[data-testid=collapsedControl]',
      '[data-testid=stSidebarCollapseButton] button',
      'button[aria-label*=idebar]',
      'button[aria-label*=pen]',
      'button[aria-label*=ollapse]'
    ];
    for(var s of selectors) {{
      var el = window.parent.document.querySelector(s);
      if(el) {{ el.click(); found=true; break; }}
    }}
    if(!found) {{
      var allBtns = window.parent.document.querySelectorAll('button');
      for(var b of allBtns) {{
        var lbl = (b.getAttribute('aria-label')||'').toLowerCase();
        var title = (b.getAttribute('title')||'').toLowerCase();
        if(lbl.includes('sidebar') || lbl.includes('open') || title.includes('sidebar')) {{
          b.click(); found=true; break;
        }}
      }}
    }}
    document.getElementById('sidebar-open-btn').style.display='none';
  ">
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round">
    <polyline points="9 18 15 12 9 6"></polyline>
  </svg>
</button>
<script>
(function() {{
  var tries = 0;
  function watchSidebar() {{
    tries++;
    var btn = document.getElementById('sidebar-open-btn');
    if (!btn && tries < 20) {{ setTimeout(watchSidebar, 300); return; }}
    if (!btn) return;
    function check() {{
      var sb = window.parent.document.querySelector('section[data-testid="stSidebar"]');
      if (!sb) {{ btn.style.display = 'none'; return; }}
      var w = sb.getBoundingClientRect().width;
      var visible = (w > 60);
      btn.style.display = visible ? 'none' : 'flex';
    }}
    setInterval(check, 400);
    check();
  }}
  watchSidebar();
}})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    uhist_sb = get_user_history(uemail_global)
    total_trained = uhist_sb.get('datasets_trained', 0)

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
        if st.button("Load Sample →", width="stretch", key="load_sample"):
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
          <div style="font-size:2.2rem;font-weight:900;color:{hc};font-variant-numeric:tabular-nums">{hs}<span style="font-size:1rem;color:{'#6b7280' if T=='dark' else '#c4b5fd'}">/100</span></div>
          <div style="font-size:.75rem;color:{'#6b7280' if T=='dark' else '#888888'};margin-top:.3rem">{df_sb.isnull().sum().sum()} nulls · {df_sb.duplicated().sum()} duplicates</div>
          <div style="height:6px;background:{'#1c1c1c' if T=='dark' else '#e0e0e0'};border-radius:3px;margin-top:.75rem;overflow:hidden">
            <div style="height:100%;width:{hs}%;background:linear-gradient(90deg,{hc},{hc}99);border-radius:3px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.training_time:
            def fmt_time_sb(s): return f"{int(s//60)}m {int(s%60)}s" if s >= 60 else f"{s:.1f}s"
            st.markdown(f"""
            <div class="sidebar-section">
              <div class="sidebar-title">⏱ Last Training</div>
              <div style="font-size:1.5rem;font-weight:900;color:{ACCENT2}">{fmt_time_sb(st.session_state.training_time)}</div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.results is not None:
            res_sb = st.session_state.results
            mc = "Model" if "Model" in res_sb.columns else res_sb.columns[0]
            nr = res_sb.select_dtypes(include=[np.number]).columns
            if len(nr):
                top_s = res_sb.iloc[0][nr[0]]
                st.markdown(f"""
                <div class="sidebar-section">
                  <div class="sidebar-title">🏆 Best Score</div>
                  <div style="font-size:1.5rem;font-weight:900;color:{ACCENT1};font-variant-numeric:tabular-nums">{top_s:.4f}</div>
                  <div style="font-size:.75rem;color:{'#6b7280' if T=='dark' else '#888888'};margin-top:.25rem">{res_sb.iloc[0][mc]}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title" style="margin-top:.5rem">⚡ Stack</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-section"><span class="insight-chip">PyCaret</span><span class="insight-chip">Plotly</span><span class="insight-chip">Pandas</span><span class="insight-chip">Streamlit</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
    if st.button("🚪 Sign Out", width="stretch", key="sidebar_logout"):
        tok = st.session_state.get("login_token")
        if tok:
            delete_token(tok)
        try:
            st.query_params.clear()
        except Exception:
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
    """Returns True if user has access, shows upgrade wall if not"""
    if plan_limits.get(feature_key, False):
        return True
    msg = upgrade_msg or f"This feature requires a higher plan. You're on **{current_plan.upper()}**."
    st.markdown(f"""
    <div class="upgrade-wall">
      <div style="font-size:2.5rem;margin-bottom:.5rem">🔒</div>
      <div style="font-size:1rem;font-weight:800;color:{ACCENTY};margin-bottom:.4rem">Feature Locked</div>
      <div style="font-size:.875rem;color:{TEXT2};margin-bottom:1rem">{msg}</div>
    </div>
    """, unsafe_allow_html=True)
    st.info("👑 Go to the **Upgrade** tab to unlock this feature.")
    return False

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
if st.session_state.data is not None:
    df = st.session_state.data
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
    dup_cnt  = df.duplicated().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊  Data Explorer", "🧬  EDA & Insights", "⚙️  Train Model",
        "🏆  Results", "📜  My History"
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
            st.warning(f"⚠️ **{dup_cnt}** duplicate rows found ({round(dup_cnt/len(df)*100,1)}%). PyCaret will handle automatically.")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">⚡</div><h3>Quick Actions</h3></div>""", unsafe_allow_html=True)
        qa1, qa2, qa3, qa4 = st.columns(4)
        with qa1:
            if st.button("🗑️ Drop Duplicates", width="stretch", key="drop_dups"):
                before = len(df)
                st.session_state.data = df.drop_duplicates().reset_index(drop=True)
                st.success(f"Removed {before - len(st.session_state.data)} duplicates!"); st.rerun()
        with qa2:
            if st.button("🧹 Drop All-Null Cols", width="stretch", key="drop_null_cols"):
                before = len(df.columns)
                st.session_state.data = df.dropna(axis=1, how='all')
                st.success(f"Removed {before - len(st.session_state.data.columns)} empty columns!"); st.rerun()
        with qa3:
            if st.button("📊 Show Data Types", width="stretch", key="show_dtypes"):
                st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column",0:"Type"}), width="stretch")
        with qa4:
            st.download_button("📥 Download CSV", df.to_csv(index=False),
                               f"dataset_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "text/csv", width="stretch")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🔍</div><h3>Data Preview</h3></div>""", unsafe_allow_html=True)
        search = st.text_input("Filter columns (comma-separated)", placeholder="e.g. Age, Sex, Survived", label_visibility="collapsed")
        show_rows = st.slider("Rows to show", 5, 100, 20, key="preview_rows")
        if search.strip():
            cols_f = [c.strip() for c in search.split(",") if c.strip() in df.columns]
            st.dataframe((df[cols_f] if cols_f else df).head(show_rows), width="stretch")
        else:
            st.dataframe(df.head(show_rows), width="stretch")

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>Column Details</h3></div>""", unsafe_allow_html=True)
            ci = pd.DataFrame({"Column":df.columns,"Type":df.dtypes.astype(str).values,
                                "Non-Null":df.count().values,
                                "Null %":((df.isnull().sum()/len(df))*100).round(1).astype(str)+"%",
                                "Unique":df.nunique().values})
            ci_display = ci.astype(str)  # Convert all to string
            st.dataframe(ci_display, width="stretch", height=300)
        with c2:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📊</div><h3>Statistical Summary</h3></div>""", unsafe_allow_html=True)
            st.dataframe(df.describe().round(3), width="stretch", height=300)

    # ═══════════════════════════
    # TAB 2 — EDA & INSIGHTS
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
                        fig.update_traces(marker_line_color=BG, marker_line_width=.5)
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
                    rows.update({"Mean":f"{s.mean():.4f}","Std":f"{s.std():.4f}","Min":f"{s.min():.4f}","Median":f"{s.median():.4f}","Max":f"{s.max():.4f}","Skew":f"{s.skew():.3f}","Kurt":f"{s.kurtosis():.3f}"})
                st.markdown(f'<div class="sidebar-section" style="margin-top:2.2rem">', unsafe_allow_html=True)
                for k, v in rows.items():
                    st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid {BORDER}"><span style="font-size:.78rem;color:{TEXT3};font-weight:600">{k}</span><span style="font-size:.82rem;color:{TEXT1};font-family:'JetBrains Mono',monospace">{v}</span></div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

            if len(num_cols) >= 2:
                st.markdown("#### 🔥 Correlation Heatmap")
                corr = df[num_cols[:15]].corr().round(2)
                fig_h = go.Figure(go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.index,
                    colorscale=[[0,ACCENTR],[.5,BG2],[1,ACCENT1]] if T=="dark" else [[0,"#dc2626"],[.5,"#f8f4ff"],[1,"#7c3aed"]],
                    zmid=0, text=corr.values.round(2), texttemplate="%{text}",
                    textfont=dict(size=9, family="JetBrains Mono"),
                    hovertemplate="%{x} × %{y}: <b>%{z}</b><extra></extra>",
                ))
                fig_h.update_layout(**chart_layout(height=460))
                st.plotly_chart(fig_h, width="stretch")

                st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 🔵 Scatter Plot Builder")
                sb1, sb2, sb3 = st.columns(3)
                with sb1: sx = st.selectbox("X axis", num_cols, key="sx")
                with sb2: sy = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sy")
                with sb3:
                    sc_opts = ["None"] + cat_cols + num_cols
                    sc = st.selectbox("Color by", sc_opts, key="sc")
                fig_sc = px.scatter(df, x=sx, y=sy, color=None if sc=="None" else sc, template=CHART_TEMPLATE, opacity=.7,
                                    color_continuous_scale="Viridis" if sc in num_cols else None,
                                    color_discrete_sequence=[ACCENT1, ACCENT2, ACCENT3, ACCENTY, ACCENTR])
                fig_sc.update_traces(marker=dict(size=6))
                fig_sc.update_layout(**chart_layout(height=420))
                st.plotly_chart(fig_sc, width="stretch")

            if len(num_cols) >= 1 and len(cat_cols) >= 1:
                st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 📦 Box Plot by Category")
                bx1, bx2 = st.columns(2)
                with bx1: bnum = st.selectbox("Numeric column", num_cols, key="bnum")
                with bx2: bcat = st.selectbox("Category column", cat_cols, key="bcat")
                fig_box = px.box(df, x=bcat, y=bnum, color=bcat, template=CHART_TEMPLATE,
                                 color_discrete_sequence=[ACCENT1, ACCENT2, ACCENT3, ACCENTY, ACCENTR],
                                 title=f"{bnum} by {bcat}")
                fig_box.update_layout(**chart_layout(height=420, showlegend=False))
                st.plotly_chart(fig_box, width="stretch")

                st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 🕳️ Missing Values per Column")
                miss = df.isnull().sum()
                miss = miss[miss > 0].sort_values(ascending=False)
                if len(miss):
                    fig_miss = px.bar(x=miss.index, y=miss.values, color=miss.values,
                                      color_continuous_scale=[ACCENT1, ACCENTY, ACCENTR],
                                      template=CHART_TEMPLATE, labels={"x":"Column","y":"Missing Count"}, title="Missing Value Count per Column")
                    fig_miss.update_layout(**chart_layout(height=340, showlegend=False, coloraxis_showscale=False))
                    st.plotly_chart(fig_miss, width="stretch")
                else:
                    st.success("✅ No missing values found in this dataset!")
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

            if uniq == len(df) and pd.api.types.is_integer_dtype(ts):
                st.error(f"⚠️ `{target_col}` looks like an ID column ({uniq} unique values). Please select the actual target!")

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

            # ── Show what's available on this plan ──────────
            max_folds = plan_limits["cv_folds_max"]
            has_advanced = plan_limits["advanced_models"]
            st.markdown(f"""
            <div style="background:{"rgba(255,255,255,0.03)" if T=="dark" else "rgba(0,0,0,0.03)"};border:1px solid {BORDER};border-radius:12px;padding:1rem 1.25rem;margin:.75rem 0;display:flex;flex-wrap:wrap;gap:.75rem">
              <span class="insight-chip" style="border-color:{plan_color};color:{plan_color}">{plan_icon} {current_plan.upper()} Plan</span>
              <span class="insight-chip">🔁 Max {max_folds}-fold CV</span>
              <span class="insight-chip">🤖 {"15+ algorithms" if has_advanced else "5 basic algorithms"}</span>
              <span class="insight-chip">📦 {"XGBoost, LGBM included" if has_advanced else "XGBoost locked 🔒"}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

            with st.expander("⚙️ Advanced Configuration", expanded=False):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    train_size = st.slider("Training Split", 0.5, 0.9, 0.8, 0.05)
                with ac2:
                    fold = st.slider(f"CV Folds (max {max_folds})", 2, max_folds, min(5, max_folds))
                with ac3:
                    max_algo_slider = plan_limits["max_algorithms"]
                    max_models = st.slider(f"Max Models (max {max_algo_slider})", 3, max_algo_slider, min(10, max_algo_slider))
                ac4, ac5 = st.columns(2)
                with ac4:
                    normalize = st.checkbox("Normalize Features", value=True)
                with ac5:
                    remove_out = st.checkbox("Remove Outliers", value=False)
            st.session_state.cv_fold = fold

            st.markdown("<br>", unsafe_allow_html=True)

            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                train_clicked = st.button("🚀 Launch Training", width="stretch")
            with col_btn2:
                if st.session_state.results is not None:
                    if st.button("🔄 Reset Results", width="stretch"):
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
                # ✅ FIXED: PyCaret setup
                kw_setup = dict(
                    data=df, 
                    target=target_col,
                    train_size=float(train_size),
                    fold=int(fold),
                    normalize=normalize,
                    verbose=False,
                    html=False,  # Disable HTML output
                    session_id=123  # Fixed seed
                )
                
                # Only remove outliers for regression with enough data
                if remove_out and ptype == "regression" and len(df) > 100:
                    kw_setup["remove_outliers"] = True

                # ✅ FIXED: Compare models settings
                kw_compare = dict(
                    verbose=False, 
                    n_select=1,
                    sort='Accuracy' if ptype == "classification" else 'R2'
                )
                
                # Exclude advanced models on free plan
                if not has_advanced:
                    if ptype == "classification":
                        kw_compare["exclude"] = ['xgboost', 'lightgbm', 'catboost']
                    else:
                        kw_compare["exclude"] = ['xgboost', 'lightgbm', 'catboost']

                # ✅ Run PyCaret
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

        # Log training
        if uemail_global:
            try:
                res_log = st.session_state.results
                mc_log = res_log.columns[0]
                nr_log = res_log.select_dtypes(include=[np.number]).columns
                bm_name = str(res_log.iloc[0][mc_log])
                bm_score = float(res_log.iloc[0][nr_log[0]]) if len(nr_log) else 0.0
                log_training(
                    email=uemail_global, 
                    dataset=str(st.session_state.dataset_name or "Uploaded CSV"),
                    problem_type=str(ptype), 
                    best_model=bm_name, 
                    score=bm_score, 
                    rows=len(df), 
                    cols=len(df.columns)
                )
                log_activity(uemail_global, "training_complete", f"{bm_name} | score={bm_score:.4f}")
            except Exception as e:
                st.warning(f"Could not log training: {e}")

        st.success(f"✅ Training complete in **{fmt_time(elapsed)}**! Head to the 🏆 Results tab.")
        if not has_advanced:
            st.info("💡 Upgrade to Pro for XGBoost, LightGBM, and CatBoost!")
        st.balloons()
        
    except Exception as e:
        placeholder.empty()
        st.error(f"❌ Training failed: {str(e)}")
        st.info("💡 Try with a different target column or check your data for issues.")
        
        # Debug info
        with st.expander("🔍 Debug Information"):
            st.write("**Error Details:**")
            st.code(str(e))
            st.write("**Dataset Info:**")
            st.write(f"- Rows: {len(df)}")
            st.write(f"- Columns: {len(df.columns)}")
            st.write(f"- Target: {target_col}")
            st.write(f"- Problem Type: {ptype}")
            st.write(f"- Null values: {df.isnull().sum().sum()}")
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
                                   f"results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", width="stretch")
            with ex2:
                # Model export gated on plan
                if plan_limits["export_model"]:
                    model_info = f"Best Model: {best_name}\n{metric_name}: {top_score:.4f}\nFolds: {folds_used}"
                    st.download_button("📋 Export Model Info", model_info, "model_info.txt", "text/plain", width="stretch")
                else:
                    if st.button("🔒 Export Model (.pkl) — Pro Only", width="stretch"):
                        st.warning("⚡ Upgrade to Pro to export trained models!")
            with ex3:
                if plan_limits["export_model"]:
                    st.info(f"💾 Saved: `best_model.pkl`")
                else:
                    st.markdown(f'<div style="text-align:center;padding:.7rem;border-radius:10px;background:{BG3};border:1px dashed {BORDER};font-size:.8rem;color:{TEXT3}">🔒 Model export locked — Pro plan</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>All Models Ranked</h3></div>""", unsafe_allow_html=True)
            styled = (res_df.style
                      .background_gradient(cmap="RdYlGn", subset=num_res)
                      .format({c: "{:.4f}" for c in num_res})
                      .set_properties(**{"font-family":"JetBrains Mono,monospace","font-size":"12px"}))
            st.dataframe(styled, width="stretch", height=360)

            st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📊</div><h3>Performance Visuals</h3></div>""", unsafe_allow_html=True)

            ch1, ch2, ch3 = st.columns(3)
            with ch1:
                top6 = res_df.head(6)
                colors = [ACCENT1 if i == 0 else BG3 for i in range(len(top6))]
                fig_b = go.Figure(go.Bar(x=top6[metric_name], y=top6[model_col], orientation="h",
                    marker_color=colors, marker_line_color=BG, marker_line_width=1,
                    text=top6[metric_name].round(4), textposition="inside",
                    textfont=dict(size=10, color="white", family="JetBrains Mono")))
                fig_b.update_layout(**chart_layout(height=360, title=f"Top Models · {metric_name}", yaxis=dict(autorange="reversed")))
                st.plotly_chart(fig_b, width="stretch")

            with ch2:
                rc = num_res[:6]
                bv = res_df.iloc[0][rc]
                mi, ma = bv.min(), bv.max()
                nv = (bv - mi) / (ma - mi + 1e-9)
                fig_r = go.Figure(go.Scatterpolar(
                    r=list(nv.values) + [nv.values[0]], theta=list(nv.index) + [nv.index[0]],
                    fill="toself",
                    fillcolor=f"{'rgba(74,222,128,0.18)' if T=='dark' else 'rgba(124,58,237,0.18)'}",
                    line=dict(color=ACCENT1, width=2.5), marker=dict(size=6, color=ACCENT1)))
                fig_r.update_layout(**chart_layout(height=360, showlegend=False, title="Best Model · Metrics Radar",
                    polar=dict(bgcolor=CHART_PAPER, radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER),
                               angularaxis=dict(gridcolor=BORDER))))
                st.plotly_chart(fig_r, width="stretch")

            with ch3:
                if len(num_res) >= 2:
                    all8 = res_df.head(8)
                    fig_s = go.Figure()
                    for i, row in all8.iterrows():
                        is_b = row[model_col] == best_name
                        fig_s.add_trace(go.Scatter(
                            x=[row[num_res[0]]], y=[row[num_res[1]]],
                            mode="markers+text",
                            marker=dict(size=20 if is_b else 11, color=ACCENT1 if is_b else ACCENT2,
                                        symbol="star" if is_b else "circle", line=dict(color=BG, width=2)),
                            text=["⭐" if is_b else ""], textposition="top center", name=row[model_col],
                            hovertemplate=f"<b>{row[model_col]}</b><br>{num_res[0]}: {row[num_res[0]]:.4f}<br>{num_res[1]}: {row[num_res[1]]:.4f}<extra></extra>"))
                    fig_s.update_layout(**chart_layout(height=360, showlegend=False, title="Score Scatter",
                                                       xaxis_title=num_res[0], yaxis_title=num_res[1]))
                    st.plotly_chart(fig_s, width="stretch")
                else:
                    st.info("Need 2+ numeric metrics for scatter.")

    # ═══════════════════════════
    # TAB 5 — MY HISTORY (gated for free plan)
    # ═══════════════════════════
    with tab5:
        uhist_h = get_user_history(uemail_global)
        history_limit = plan_limits["history_entries"]


        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        full_training_log = uhist_h.get("training_log", [])
        # Free plan: only show last 3
        if not plan_limits["full_history"] and len(full_training_log) > history_limit:
            training_log = full_training_log[-history_limit:]
            st.markdown(f"""
            <div style="background:{"rgba(251,191,36,0.06)" if T=="dark" else "rgba(251,191,36,0.08)"};border:1px solid {"rgba(251,191,36,0.30)" if T=="dark" else "rgba(251,191,36,0.35)"};border-radius:12px;padding:.9rem 1.25rem;margin-bottom:1rem;display:flex;align-items:center;gap:.75rem">
              <span style="font-size:1.2rem">⚠️</span>
              <div style="font-size:.85rem;color:{TEXT2}">Showing last <b style="color:{ACCENTY}">{history_limit}</b> of <b>{len(full_training_log)}</b> training runs. <b>Upgrade to Pro</b> to see your full history.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            training_log = full_training_log

        st.markdown(f"""<div class="section-head"><div class="icon-wrap">🗂️</div><h3>Previous Work — Datasets & Projects</h3></div>""", unsafe_allow_html=True)

        if not training_log:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem 1rem;background:{CARD_BG};border:1px solid {BORDER};border-radius:20px;margin-bottom:1.5rem">
              <div style="font-size:4rem;margin-bottom:.75rem;opacity:.35">🗂️</div>
              <div style="font-size:1.1rem;font-weight:700;color:{TEXT1};margin-bottom:.4rem">No projects yet</div>
              <div style="color:{TEXT2};font-size:.875rem">Upload a dataset and train your first model to see your work history here!</div>
            </div>""", unsafe_allow_html=True)
        else:
            seen_datasets = {}
            for t in training_log:
                ds = t.get("dataset","Unknown")
                if ds not in seen_datasets: seen_datasets[ds] = []
                seen_datasets[ds].append(t)

            for ds_name, ds_runs in seen_datasets.items():
                best_run = max(ds_runs, key=lambda x: x.get("score",0))
                all_models = list(dict.fromkeys([r.get("best_model","—") for r in ds_runs]))
                ptype = best_run.get("problem_type","—")
                ptype_icon = "🎯" if ptype == "classification" else "📈"
                ptype_color = ACCENT1 if ptype == "classification" else ACCENT2
                runs_count = len(ds_runs)
                best_score = best_run.get("score", 0)
                rows_n = best_run.get("rows", 0); cols_n = best_run.get("cols", 0)
                first_run = ds_runs[0].get("time","")[:10]

                st.markdown(f"""
                <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:20px;padding:1.5rem;margin-bottom:1rem;position:relative;overflow:hidden">
                  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{ptype_color},{ACCENT3})"></div>
                  <div style="display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem">
                    <div style="font-size:2.2rem;flex-shrink:0">{ptype_icon}</div>
                    <div style="flex:1;min-width:0">
                      <div style="font-size:1rem;font-weight:800;color:{TEXT1};white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{ds_name}</div>
                      <div style="display:flex;gap:.5rem;margin-top:.4rem;flex-wrap:wrap">
                        <span style="background:{ptype_color}18;color:{ptype_color};border:1px solid {ptype_color}44;border-radius:6px;padding:.15rem .55rem;font-size:.68rem;font-weight:700">{ptype.upper()}</span>
                        <span style="background:{BG3};color:{TEXT3};border:1px solid {BORDER};border-radius:6px;padding:.15rem .55rem;font-size:.68rem;font-weight:600">{rows_n:,} rows × {cols_n} cols</span>
                        <span style="background:{BG3};color:{TEXT3};border:1px solid {BORDER};border-radius:6px;padding:.15rem .55rem;font-size:.68rem;font-weight:600">{runs_count} run{"s" if runs_count>1 else ""}</span>
                        <span style="background:{BG3};color:{TEXT3};border:1px solid {BORDER};border-radius:6px;padding:.15rem .55rem;font-size:.68rem;font-weight:600">First: {first_run}</span>
                      </div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;background:{ptype_color}10;border:1px solid {ptype_color}33;border-radius:12px;padding:.6rem 1rem">
                      <div style="font-size:.6rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{ptype_color};margin-bottom:.15rem">Best Score</div>
                      <div style="font-size:1.5rem;font-weight:900;color:{ptype_color};font-family:'JetBrains Mono',monospace">{best_score:.4f}</div>
                    </div>
                  </div>
                  <div style="margin-bottom:.5rem">
                    <div style="font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:{TEXT3};margin-bottom:.4rem">🤖 Models Tried</div>
                    <div style="display:flex;flex-wrap:wrap;gap:.3rem">
                """, unsafe_allow_html=True)
                for i, m in enumerate(all_models):
                    chip_bg = "rgba(74,222,128,0.10)" if i == 0 else BG3
                    chip_col = "#4ade80" if i == 0 else TEXT2
                    chip_bdr = "rgba(74,222,128,0.30)" if i == 0 else BORDER
                    st.markdown(f'<span style="background:{chip_bg};color:{chip_col};border:1px solid {chip_bdr};border-radius:6px;padding:.2rem .6rem;font-size:.72rem;font-weight:600">{"🥇 " if i==0 else ""}{m}</span>', unsafe_allow_html=True)
                st.markdown("</div></div></div>", unsafe_allow_html=True)

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

        hist_col1, hist_col2 = st.columns([3, 2])
        with hist_col1:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📈</div><h3>Score Progress Over Time</h3></div>""", unsafe_allow_html=True)
            if not training_log:
                st.info("Train a model to see your score progression here.")
            else:
                scores = [t["score"] for t in training_log]
                models = [t["best_model"] for t in training_log]
                ds_names_ch = [t.get("dataset","?")[:18] for t in training_log]
                ptypes_ch = [t.get("problem_type","?") for t in training_log]
                colors_ch = [ACCENT1 if p=="classification" else ACCENT2 for p in ptypes_ch]
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Scatter(x=list(range(1,len(scores)+1)), y=scores, mode="none", fill="tozeroy",
                    fillcolor=f"{'rgba(74,222,128,0.06)' if T=='dark' else 'rgba(124,58,237,0.06)'}", showlegend=False, hoverinfo="skip"))
                fig_hist.add_trace(go.Scatter(x=list(range(1,len(scores)+1)), y=scores, mode="lines+markers",
                    line=dict(color=ACCENT1, width=2.5), marker=dict(size=11, color=colors_ch, line=dict(color=BG, width=2)),
                    customdata=list(zip(models, ds_names_ch, ptypes_ch)),
                    hovertemplate="<b>Run #%{x}</b><br>Score: <b>%{y:.4f}</b><br>Model: %{customdata[0]}<br>Dataset: %{customdata[1]}<extra></extra>",
                    showlegend=False))
                best_s = max(scores)
                fig_hist.add_hline(y=best_s, line_dash="dot", line_color=ACCENTY, line_width=1.5,
                    annotation_text=f"Best: {best_s:.4f}", annotation_font_color=ACCENTY, annotation_font_size=10)
                fig_hist.update_layout(**chart_layout(height=300, title="Score per Training Run", xaxis_title="Run #", yaxis_title="Best Score", showlegend=False))
                st.plotly_chart(fig_hist, width="stretch")

        with hist_col2:
            st.markdown(f"""<div class="section-head"><div class="icon-wrap">📋</div><h3>Activity Log</h3></div>""", unsafe_allow_html=True)
            activity_log = list(reversed(uhist_h.get("activity_log", [])))[:25]
            act_icons = {"signup":"✨","signin":"🔑","training_complete":"🏆","logout":"🚪","payment_submitted":"💳"}
            act_colors = {"signup":ACCENT1,"signin":ACCENT2,"training_complete":ACCENTY,"logout":TEXT3,"payment_submitted":ACCENT3}
            if not activity_log:
                st.info("No activity yet.")
            else:
                st.markdown(f'<div style="max-height:560px;overflow-y:auto;padding-right:.25rem">', unsafe_allow_html=True)
                for act in activity_log:
                    a_type = act.get("action","—")
                    a_icon = act_icons.get(a_type, "◎")
                    a_color = act_colors.get(a_type, TEXT3)
                    st.markdown(f"""
                    <div style="display:flex;gap:.6rem;align-items:flex-start;padding:.6rem .75rem;margin-bottom:.35rem;background:{CARD_BG};border:1px solid {BORDER};border-radius:10px">
                      <div style="font-size:1rem;flex-shrink:0;margin-top:.05rem">{a_icon}</div>
                      <div style="flex:1;min-width:0">
                        <div style="font-size:.78rem;font-weight:700;color:{a_color};text-transform:capitalize">{a_type.replace('_',' ')}</div>
                        <div style="font-size:.7rem;color:{TEXT3};margin-top:.1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{act.get("detail","")}</div>
                      </div>
                      <div style="font-size:.65rem;color:{TEXT3};flex-shrink:0;font-family:'JetBrains Mono',monospace;white-space:nowrap">{act.get("time","")[:16]}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)
        exp1, exp2 = st.columns(2)
        with exp1:
            tlog_df = pd.DataFrame(uhist_h.get("training_log", []))
            if not tlog_df.empty:
                st.download_button("📥 Export Training History CSV", tlog_df.to_csv(index=False),
                                   f"training_history_{uemail_global[:10]}.csv", "text/csv", width="stretch")
        with exp2:
            alog_df = pd.DataFrame(uhist_h.get("activity_log", []))
            if not alog_df.empty:
                st.download_button("📥 Export Activity Log CSV", alog_df.to_csv(index=False),
                                   f"activity_log_{uemail_global[:10]}.csv", "text/csv", width="stretch")

# ═══════════════════════════
# WELCOME SCREEN
# ═══════════════════════════
else:
    st.markdown(f"""
    <div class="hero-wrap slide-up">
      <h1>Drop Your Data.<br>We Do the Rest.</h1>
      <p>DataForge ML Studio — zero-code AutoML. Upload a CSV, pick a target, hit train. Get a production model in minutes.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="glow-divider"></div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("🧬","Smart EDA",       "Correlation heatmaps, distribution explorer, scatter builder, and missing value charts."),
        ("⚡","AutoML Engine",   "15+ algorithms compared with k-fold cross-validation. Best model wins — automatically."),
        ("🎯","Smart Detection", "Auto-detects regression vs classification. Warns about ID columns. Quick data cleaning."),
        ("🏆","Rich Results",    "Trophy banner, radar + scatter + bar charts, metric breakdown bars, model export."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], feats):
        with col:
            st.markdown(f"""
            <div class="feature-card slide-up">
              <div class="fc-icon">{icon}</div>
              <h3>{title}</h3><p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;color:{TEXT3};font-size:.85rem;padding:2.5rem">
      👈 Upload a CSV/Excel or load a sample dataset from the sidebar to get started
    </div>""", unsafe_allow_html=True)
