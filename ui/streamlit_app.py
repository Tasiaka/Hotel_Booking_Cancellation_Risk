from __future__ import annotations

from io import BytesIO
import hashlib
import hmac
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import secrets
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Local configuration
# -----------------------------------------------------------------------------
API_URL = os.getenv("HOTEL_RISK_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEMO_PATH = Path(os.getenv("HOTEL_RISK_DEMO_DATA", "data/processed/main_modeling_dataset.csv"))
LOG_FILE = Path(os.getenv("HOTEL_RISK_UI_LOG_FILE", "storage/logs/streamlit.log"))
USER_STORE_PATH = Path(os.getenv("HOTEL_RISK_USER_STORE", "storage/users.json"))
CANONICAL_SAMPLE_PATH = Path("data/examples/canonical_upload_example.csv")
FLEXIBLE_SAMPLE_PATH = Path("data/examples/flexible_upload_example.csv")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

ui_logger = logging.getLogger("hotel_risk.ui")
ui_logger.setLevel(getattr(logging, os.getenv("HOTEL_RISK_LOG_LEVEL", "INFO").upper(), logging.INFO))
if not ui_logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler = RotatingFileHandler(LOG_FILE, maxBytes=3_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(formatter)
    ui_logger.addHandler(handler)

st.set_page_config(page_title="Hotel Risk", page_icon="🏨", layout="wide", initial_sidebar_state="collapsed")

DEFAULT_CURRENCY_CODE = "USD"
DEFAULT_COST_PER_ACTION = 5.0

DEFAULT_USERS = {
    "admin": {"password": "admin", "role": "admin", "label": "Администратор", "source": "demo"},
    "user": {"password": "user", "role": "user", "label": "Менеджер", "source": "demo"},
}

for key, default in {
    "auth": None,
    "predictions": None,
    "summary": None,
    "validation": None,
    "batch": None,
    "top_share": 0.20,
    "success_rate": 0.25,
    "cost_per_action": DEFAULT_COST_PER_ACTION,
    "currency_code": DEFAULT_CURRENCY_CODE,
    "ui_logs": [],
    "manager_view": "Список к проверке",
    "admin_view": "Статус",
    "last_analysis_signature": None,
    "recommended_review_ids": [],
}.items():
    st.session_state.setdefault(key, default)

# -----------------------------------------------------------------------------
# Design system
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
      --bg: #F3F7FC;
      --surface: #FFFFFF;
      --soft: #F8FAFC;
      --text: #111827;
      --text-2: #344054;
      --muted: #667085;
      --line: #E4E7EC;
      --line-2: #D0D5DD;
      --blue: #4DA0F7;
      --blue-soft: #EAF4FF;
      --blue-dark: #175CD3;
      --navy: #101828;
      --navy-2: #14213A;
      --yellow: #FFE01B;
      --yellow-dark: #F5C400;
      --green: #067647;
      --green-soft: #ECFDF3;
      --red: #B42318;
      --red-soft: #FEF3F2;
      --amber: #B54708;
      --amber-soft: #FFFAEB;
      --shadow: 0 18px 55px rgba(16, 24, 40, .10);
      --shadow-sm: 0 9px 24px rgba(16, 24, 40, .07);
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
      background: radial-gradient(circle at 12% 0%, #FFFFFF 0, #FFFFFF 28%, #F3F7FC 70%, #EEF4FB 100%) !important;
      color: var(--text) !important;
    }
    .block-container {
      max-width: 1220px;
      padding: clamp(.7rem, 1.7vw, 1.2rem) clamp(1rem, 3vw, 2.15rem) 2.4rem clamp(1rem, 3vw, 2.15rem);
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div, [data-testid="stMarkdownContainer"] * {
      color: var(--text) !important;
    }
    [data-testid="stSidebar"], #MainMenu, footer, header { display: none !important; visibility: hidden !important; }

    .hero {
      position: relative;
      overflow: hidden;
      border-radius: clamp(22px, 3vw, 34px);
      background: linear-gradient(135deg, #D9ECFF 0%, #8BC6FF 55%, #C5E1FF 100%);
      box-shadow: var(--shadow);
      padding: clamp(2rem, 4vw, 4.2rem) clamp(1.4rem, 5vw, 5rem);
      margin: 0 auto clamp(1.1rem, 2.3vw, 1.7rem) auto;
      min-height: clamp(520px, 68vh, 690px);
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .hero:before {
      content: "";
      position: absolute;
      width: clamp(150px, 21vw, 295px);
      height: clamp(150px, 21vw, 295px);
      right: 8%; top: 8%;
      border-radius: 54px;
      background: rgba(255,255,255,.22);
      transform: rotate(10deg);
    }
    .hero:after {
      content: "";
      position: absolute;
      width: clamp(320px, 50vw, 760px);
      height: clamp(320px, 50vw, 760px);
      right: -10%; bottom: -44%;
      border-radius: 50%;
      background: rgba(255,255,255,.28);
    }
    .hero-content { position: relative; z-index: 2; max-width: 850px; }
    .hero-title {
      font-size: clamp(2.75rem, 5.7vw, 5.25rem);
      line-height: .99;
      letter-spacing: -.06em;
      font-weight: 920;
      color: #111827 !important;
      margin: 0 0 clamp(1rem, 2.2vw, 1.45rem) 0;
    }
    .hero-subtitle {
      max-width: 690px;
      font-size: clamp(1.03rem, 1.35vw, 1.20rem);
      line-height: 1.55;
      font-weight: 650;
      color: #21344D !important;
      margin-bottom: clamp(1.35rem, 2.6vw, 2.05rem);
    }
    .hero-cards {
      position: relative;
      z-index: 2;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: clamp(.85rem, 2vw, 1.2rem);
      max-width: 980px;
    }
    .feature-card {
      background: linear-gradient(145deg, #111827, #14213A);
      border: 1px solid rgba(255,255,255,.14);
      border-radius: 24px;
      min-height: 138px;
      padding: 1.15rem;
      box-shadow: 0 20px 40px rgba(17,24,39,.20);
    }
    .feature-head { display: flex; align-items: center; gap: .75rem; margin-bottom: .65rem; }
    .feature-icon {
      width: 42px; height: 42px; border-radius: 50%;
      display: inline-flex; align-items: center; justify-content: center;
      background: #1D4ED8;
      color: #FFFFFF !important;
      font-size: 1.15rem;
      font-weight: 900;
    }
    .feature-title { color: #FFFFFF !important; font-weight: 850; font-size: 1.08rem; }
    .feature-text { color: #E6EEF9 !important; line-height: 1.55; font-size: .94rem; }

    .login-card-shell {
      max-width: 510px;
      margin: 0 auto 1.4rem auto;
      background: rgba(255,255,255,.97);
      border: 1px solid rgba(208,213,221,.85);
      border-radius: 26px;
      padding: clamp(1.35rem, 2vw, 2rem);
      box-shadow: var(--shadow);
    }
    .login-title {
      font-size: clamp(1.65rem, 2.5vw, 2rem);
      font-weight: 900;
      text-align: center;
      margin: 0 0 .45rem 0;
      letter-spacing: -.025em;
    }
    .login-caption { color: var(--muted) !important; text-align: center; margin-bottom: 1.1rem; line-height: 1.45; }
    .login-caption b { color: var(--blue-dark) !important; }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: .82rem 1rem;
      margin-bottom: 1rem;
      background: rgba(255,255,255,.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow-sm);
      backdrop-filter: blur(8px);
    }
    .brand { font-weight: 900; letter-spacing: -.025em; font-size: 1.08rem; }
    .role-pill {
      display:inline-flex; align-items:center; gap:.45rem;
      padding:.38rem .70rem; border-radius:999px;
      background:#EEF4FF; color:#175CD3 !important;
      font-weight:800; font-size:.82rem;
      border:1px solid #C7D7FE;
      white-space: nowrap;
    }

    .page-hero {
      background: linear-gradient(135deg, #D9ECFF 0%, #A9D5FF 100%);
      border-radius: 30px;
      padding: clamp(1.25rem, 2.8vw, 2.45rem);
      margin-bottom: 1rem;
      box-shadow: var(--shadow-sm);
      border: 1px solid rgba(255,255,255,.65);
      position: relative;
      overflow: hidden;
    }
    .page-hero:after {
      content:"";
      position:absolute; right:-42px; top:-68px;
      width:260px; height:260px; border-radius:54px;
      background:rgba(255,255,255,.22); transform:rotate(12deg);
    }
    .page-title {
      font-size: clamp(2rem, 3.9vw, 3.65rem);
      line-height: 1.04;
      letter-spacing: -.055em;
      font-weight: 930;
      max-width: 840px;
      margin: 0 0 .7rem 0;
      position: relative; z-index: 2;
    }
    .page-subtitle { position: relative; z-index: 2; max-width: 760px; color: #21344D !important; font-size: 1.03rem; line-height: 1.55; font-weight: 600; }

    .action-panel, .panel, .metric-card, .detail-card, .nav-panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow-sm);
    }
    .action-panel { padding: 1rem; margin-bottom: 1rem; }
    .panel { padding: 1rem; margin: .35rem 0 1rem 0; }
    .panel-title { font-size: 1.05rem; font-weight: 900; margin-bottom: .35rem; }
    .panel-text { color: var(--text-2) !important; line-height: 1.5; }

    .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0; }
    .metric-grid.compact { margin: .75rem 0; }
    .metric-card {
      padding: .95rem 1rem;
      min-height: 116px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      overflow: hidden;
    }
    .metric-label { font-size: .72rem; font-weight: 850; color: #667085 !important; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value {
      margin-top: .28rem;
      font-size: clamp(1.35rem, 2.0vw, 1.85rem);
      line-height: 1.08;
      font-weight: 930;
      letter-spacing: -.035em;
      color: #111827 !important;
      overflow-wrap: anywhere;
    }
    .metric-hint { margin-top: .3rem; color: #667085 !important; font-size: .80rem; line-height: 1.34; }
    .metric-card.good { border-color: #ABEFC6; background: #F6FEF9; }
    .metric-card.warn { border-color: #FEDF89; background: #FFFCF5; }
    .metric-card.bad { border-color: #FECDCA; background: #FFFBFA; }

    .detail-card { padding: 1rem; min-height: 120px; }
    .detail-title { font-weight:900; font-size:1rem; margin-bottom:.35rem; }
    .detail-text { color:var(--text-2)!important; line-height:1.55; font-size:.94rem; }
    .mini-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0; }

    .soft-alert { display:flex; gap:.75rem; align-items:flex-start; border-radius:20px; padding:.95rem 1rem; border:1px solid #ABEFC6; background:#F0FDF4; margin:.65rem 0 1rem 0; }
    .soft-alert.warn { background: var(--amber-soft); border-color: #FEDF89; }
    .soft-alert.bad { background: var(--red-soft); border-color: #FECDCA; }
    .soft-alert-icon { width:36px; height:36px; min-width:36px; border-radius:13px; display:flex; align-items:center; justify-content:center; background:#DCFAE6; color:#067647 !important; font-weight:900; font-size:1.05rem; }
    .soft-alert.warn .soft-alert-icon { background:#FEF0C7; color:#B54708!important; }
    .soft-alert.bad .soft-alert-icon { background:#FEE4E2; color:#B42318!important; }
    .soft-alert-title { font-weight:900; margin-bottom:.2rem; }
    .soft-alert-text { color:var(--text-2)!important; line-height:1.5; }

    .nav-panel { padding: .35rem; margin: .7rem 0 1rem 0; }
    div[role="radiogroup"] { display: flex; gap: .42rem; flex-wrap: wrap; }
    div[role="radiogroup"] > label {
      background: #FFFFFF !important;
      border: 1px solid var(--line) !important;
      border-radius: 999px !important;
      padding: .42rem .78rem !important;
      margin: 0 !important;
      box-shadow: none !important;
      min-height: 38px;
    }
    div[role="radiogroup"] > label:hover { background: var(--blue-soft) !important; border-color: #B2DDFF !important; }
    div[role="radiogroup"] > label * { font-size: .92rem !important; font-weight: 760 !important; color: #344054 !important; }
    div[role="radiogroup"] input:checked + div * { color: #175CD3 !important; }

    div.stButton > button, div.stDownloadButton > button, a[data-testid="stBaseButton-secondary"] {
      border-radius: 18px !important;
      min-height: 50px;
      font-weight: 850 !important;
      border: 1px solid var(--line-2) !important;
      box-shadow: var(--shadow-sm) !important;
    }
    div.stButton > button[kind="primary"], div.stButton > button[data-testid="baseButton-primary"], div.stFormSubmitButton > button {
      background: linear-gradient(180deg, #FFE842 0%, #FFD700 100%) !important;
      color: #111827 !important;
      border: 0 !important;
      border-radius: 18px !important;
      min-height: 50px;
      font-weight: 900 !important;
      box-shadow: 0 14px 30px rgba(245, 196, 0, .28) !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button:hover { transform: translateY(-1px); background: linear-gradient(180deg, #FFE01B 0%, #F5C400 100%) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm); background: var(--surface); }

    @media (max-width: 1000px) {
      .hero-cards, .mini-grid, .metric-grid { grid-template-columns: 1fr; }
      .hero { min-height: auto; }
      .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def html_escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def ui_log(message: str, **context: Any) -> None:
    line = message if not context else message + " | " + " ".join(f"{k}={v}" for k, v in context.items())
    ui_logger.info(line)
    st.session_state.ui_logs.append(line)
    st.session_state.ui_logs = st.session_state.ui_logs[-80:]


# -----------------------------------------------------------------------------
# Local authentication
# -----------------------------------------------------------------------------
def _normalize_username(username: str | None) -> str:
    return (username or "").strip().lower()


def _load_registered_users() -> dict[str, Any]:
    if not USER_STORE_PATH.exists():
        return {"users": {}}
    try:
        data = json.loads(USER_STORE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        ui_log("user_store_read_failed", error=exc)
        return {"users": {}}
    if not isinstance(data, dict):
        return {"users": {}}
    users = data.get("users", {})
    if not isinstance(users, dict):
        users = {}
    return {"users": users}


def _save_registered_users(data: dict[str, Any]) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_STORE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_password(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        120_000,
    )
    return digest.hex()


def _new_password_record(password: str) -> dict[str, str]:
    salt_hex = secrets.token_hex(16)
    return {"salt": salt_hex, "password_hash": _hash_password(password, salt_hex), "algorithm": "pbkdf2_sha256"}


def _verify_password(password: str, record: dict[str, Any]) -> bool:
    salt_hex = str(record.get("salt", ""))
    expected = str(record.get("password_hash", ""))
    if not salt_hex or not expected:
        return False
    actual = _hash_password(password, salt_hex)
    return hmac.compare_digest(actual, expected)


def authenticate_user(username: str | None, password: str | None) -> dict[str, str] | None:
    """Authenticate demo or locally registered users."""
    normalized = _normalize_username(username)
    password = password or ""
    if not normalized or not password:
        return None

    demo = DEFAULT_USERS.get(normalized)
    if demo and hmac.compare_digest(password, str(demo["password"])):
        return {
            "username": normalized,
            "role": str(demo["role"]),
            "label": str(demo["label"]),
            "source": str(demo.get("source", "demo")),
        }

    store = _load_registered_users()
    record = store.get("users", {}).get(normalized)
    if not isinstance(record, dict):
        return None
    if not _verify_password(password, record):
        return None
    return {
        "username": normalized,
        "role": "user",
        "label": str(record.get("label") or normalized),
        "source": "registered",
    }


def register_user(username: str | None, password: str | None, password_repeat: str | None, label: str | None = None) -> tuple[bool, str]:
    normalized = _normalize_username(username)
    password = password or ""
    password_repeat = password_repeat or ""
    display_label = (label or normalized).strip()

    if not re.fullmatch(r"[a-zA-Z0-9_.-]{3,32}", normalized):
        return False, "Логин должен содержать 3–32 символа: латиница, цифры, точка, дефис или подчеркивание."
    if normalized in DEFAULT_USERS:
        return False, "Этот логин зарезервирован для demo-доступа."
    if len(password) < 4:
        return False, "Пароль должен быть не короче 4 символов."
    if password != password_repeat:
        return False, "Пароли не совпадают."

    store = _load_registered_users()
    users = store.setdefault("users", {})
    if normalized in users:
        return False, "Пользователь с таким логином уже существует."

    record = _new_password_record(password)
    record.update({
        "role": "user",
        "label": display_label or normalized,
        "source": "registered",
    })
    users[normalized] = record
    _save_registered_users(store)
    return True, "Аккаунт создан. Выполнен вход как менеджер."


def users_for_admin_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for username, record in DEFAULT_USERS.items():
        rows.append({
            "Логин": username,
            "Роль": str(record["role"]),
            "Имя": str(record["label"]),
            "Тип": "demo",
        })
    store = _load_registered_users()
    for username, record in sorted(store.get("users", {}).items()):
        if not isinstance(record, dict):
            continue
        rows.append({
            "Логин": username,
            "Роль": "user",
            "Имя": str(record.get("label") or username),
            "Тип": "registered",
        })
    return pd.DataFrame(rows)


def fmt_int(x: float | int | None) -> str:
    try:
        return f"{float(x):,.0f}".replace(",", " ")
    except Exception:
        return "—"


CURRENCY_OPTIONS = {
    "EUR": "€",
    "USD": "$",
    "RUB": "₽",
    "GBP": "£",
}


def current_currency_symbol() -> str:
    code = str(st.session_state.get("currency_code", DEFAULT_CURRENCY_CODE))
    return CURRENCY_OPTIONS.get(code, CURRENCY_OPTIONS[DEFAULT_CURRENCY_CODE])


def current_currency_label() -> str:
    code = str(st.session_state.get("currency_code", DEFAULT_CURRENCY_CODE))
    if code not in CURRENCY_OPTIONS:
        code = DEFAULT_CURRENCY_CODE
        st.session_state.currency_code = code
    return f"{code} {current_currency_symbol()}"


def fmt_money(x: float | int | None, symbol: str | None = None) -> str:
    text = fmt_int(x)
    if text == "—":
        return "—"
    currency_symbol = current_currency_symbol() if symbol is None else symbol
    return text if not currency_symbol else f"{text} {currency_symbol}"


def money_column_format() -> str:
    symbol = current_currency_symbol()
    return "%.0f" if not symbol else f"%.0f {symbol}"


def fmt_pct(x: float | int | None) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "—"


def metric_card(label: str, value: str, hint: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return f"""
    <div class="metric-card{tone_class}">
      <div>
        <div class="metric-label">{html_escape(label)}</div>
        <div class="metric-value">{html_escape(value)}</div>
      </div>
      <div class="metric-hint">{html_escape(hint)}</div>
    </div>
    """


def render_metric_grid(cards: list[tuple[str, str, str] | tuple[str, str, str, str]], compact: bool = False) -> None:
    """Render KPI cards with native Streamlit elements.

    No extra status badges are rendered inside KPI cards. The UI should look like
    a working business tool, not a debug report.
    """
    if not cards:
        return
    if compact:
        per_row = min(len(cards), 4)
    elif len(cards) == 6:
        per_row = 3
    else:
        per_row = min(len(cards), 4)
    for start in range(0, len(cards), per_row):
        cols = st.columns(per_row)
        for col, item in zip(cols, cards[start:start + per_row]):
            if len(item) == 3:
                label, value, hint = item
            else:
                label, value, hint, _tone = item
            with col:
                with st.container(border=True):
                    st.caption(str(label))
                    st.markdown(f"### {value}")
                    st.caption(str(hint))


def render_info_stack(cards: list[tuple[str, str, str]]) -> None:
    """Vertical admin information stack.

    Used on the admin page instead of horizontal KPI tiles, because the admin
    page needs compact technical facts rather than dashboard cards.
    """
    for label, value, hint in cards:
        with st.container(border=True):
            st.caption(str(label))
            st.markdown(f"### {value}")
            if hint:
                st.caption(str(hint))


def soft_alert(title: str, text: str, kind: str = "ok", icon: str = "✓") -> None:
    kind_class = "" if kind == "ok" else kind
    st.markdown(
        f"""
        <div class="soft-alert {kind_class}">
          <div class="soft-alert-icon">{html_escape(icon)}</div>
          <div>
            <div class="soft-alert-title">{html_escape(title)}</div>
            <div class="soft-alert-text">{text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav_radio(label: str, options: list[str], key: str) -> str:
    st.markdown('<div class="nav-panel">', unsafe_allow_html=True)
    value = st.radio(label, options, horizontal=True, label_visibility="collapsed", key=key)
    st.markdown('</div>', unsafe_allow_html=True)
    return value


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    if DEMO_PATH.exists():
        df = pd.read_csv(DEMO_PATH)
        if "split" in df.columns:
            df = df[df["split"].eq("test")]
        return df.reset_index(drop=True)
    return pd.DataFrame()


def read_sample(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return b""


def build_input_signature(source: str, uploaded: Any, df: pd.DataFrame) -> str:
    """Return a stable fingerprint of the currently selected input data.

    Streamlit keeps session_state between reruns. Without an explicit fingerprint,
    an already calculated dashboard can remain visible after the user uploads a
    different CSV and before they press «Запустить анализ». That is dangerous for
    a decision-support product: the screen would show metrics for the previous
    file while the controls show the new file.
    """
    if source == "demo":
        if DEMO_PATH.exists():
            stat = DEMO_PATH.stat()
            return f"demo:{DEMO_PATH.resolve()}:{stat.st_size}:{int(stat.st_mtime)}:{len(df)}"
        return f"demo:missing:{len(df)}"

    if uploaded is None:
        return "upload:none"

    try:
        content = uploaded.getvalue()
        digest = hashlib.sha256(content).hexdigest()[:20]
        size = len(content)
    except Exception:
        # Fallback for unexpected UploadedFile implementations. It is enough to
        # distinguish the visible dataset and force the manager to rerun analysis.
        digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:20]
        size = int(getattr(uploaded, "size", 0) or 0)

    return f"upload:{getattr(uploaded, 'name', 'file')}:{size}:{digest}:{len(df)}"


def clear_analysis_state() -> None:
    """Remove analysis artifacts that belong to a previous input dataset."""
    for key in ["predictions", "summary", "validation", "batch"]:
        st.session_state[key] = None
    st.session_state["recommended_review_ids"] = []


def optimize_review_list(actionable_pred: pd.DataFrame, success_rate: float, cost_per_action: float) -> tuple[pd.DataFrame, dict[str, float]]:
    """Choose the review list size automatically by expected business effect.

    The manager should not manually guess whether 5%, 20% or 50% of bookings
    should be reviewed. We sort by financial priority and take the prefix that
    maximizes: prevented_expected_loss - manual_review_cost. If every prefix is
    unprofitable, the recommended list is empty.
    """
    if actionable_pred.empty or "expected_loss" not in actionable_pred.columns:
        return actionable_pred.head(0).copy(), {
            "review_count": 0.0,
            "review_expected_loss": 0.0,
            "protected": 0.0,
            "cost": 0.0,
            "net": 0.0,
        }

    sort_col = "business_priority_score" if "business_priority_score" in actionable_pred.columns else "expected_loss"
    ranked = actionable_pred.sort_values(sort_col, ascending=False).reset_index(drop=True).copy()
    expected = pd.to_numeric(ranked["expected_loss"], errors="coerce").fillna(0).clip(lower=0)
    if expected.sum() <= 0:
        return ranked.head(0).copy(), {
            "review_count": 0.0,
            "review_expected_loss": 0.0,
            "protected": 0.0,
            "cost": 0.0,
            "net": 0.0,
        }

    cumulative_loss = expected.cumsum()
    counts = pd.Series(range(1, len(ranked) + 1), dtype=float)
    protected = cumulative_loss * float(success_rate)
    costs = counts * float(cost_per_action)
    net = protected - costs
    best_idx = int(net.idxmax())
    best_net = float(net.iloc[best_idx])

    if best_net <= 0:
        return ranked.head(0).copy(), {
            "review_count": 0.0,
            "review_expected_loss": 0.0,
            "protected": 0.0,
            "cost": 0.0,
            "net": 0.0,
        }

    top = ranked.iloc[: best_idx + 1].copy()
    return top, {
        "review_count": float(best_idx + 1),
        "review_expected_loss": float(cumulative_loss.iloc[best_idx]),
        "protected": float(protected.iloc[best_idx]),
        "cost": float(costs.iloc[best_idx]),
        "net": best_net,
    }


def api_get(path: str) -> Any:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=6)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        ui_log("api_get_failed", path=path, error=exc)
        return {"status": "offline", "error": str(exc)}


def api_health() -> dict:
    result = api_get("/health")
    return result if isinstance(result, dict) else {"status": "offline"}


def dataframe_payload(df: pd.DataFrame) -> dict:
    """Build a strict JSON payload for API calls.

    Raw hotel CSV files often contain NaN/NaT values in columns like company,
    agent or country. The requests json= encoder refuses non-finite floats,
    so pandas to_json is used as the canonical sanitizer: NaN/NaT become null
    and timestamps become ISO-compatible strings.
    """
    try:
        records = json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception as exc:
        raise ValueError("CSV содержит значения, которые не удалось привести к JSON для API.") from exc
    return {"bookings": records}


def validate_df(df: pd.DataFrame) -> dict:
    payload = dataframe_payload(df)
    response = requests.post(f"{API_URL}/api/v1/validate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def score_df(df: pd.DataFrame) -> dict:
    payload = dataframe_payload(df)
    limit = max(1, len(df))
    response = requests.post(f"{API_URL}/api/v1/score?limit={limit}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def upload_batch(df: pd.DataFrame, name: str = "manager_scoring", return_predictions: bool = True) -> dict:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    files = {"file": (f"{name}.csv", buffer, "text/csv")}
    params = {"name": name, "return_predictions": str(return_predictions).lower(), "prediction_limit": max(1, len(df))}
    response = requests.post(f"{API_URL}/api/v1/batches/upload", params=params, files=files, timeout=420)
    response.raise_for_status()
    return response.json()


def normalize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty:
        return pred
    out = pred.copy()
    for col in ["risk_score", "cancellation_probability", "expected_loss", "booking_value", "business_priority_score", "total_nights"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    if "booking_id" in out.columns:
        out["booking_id"] = out["booking_id"].astype(str)
    return out


def logout_button() -> None:
    if st.button("Выйти", use_container_width=True):
        for key in ["auth", "predictions", "summary", "validation", "batch"]:
            st.session_state[key] = None
        st.rerun()


def render_topbar() -> dict:
    health = api_health()
    user = st.session_state.auth or {"label": "—"}
    status = "API online" if health.get("status") == "ok" else "API offline"
    model_status = health.get("model_status", "unknown")
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">Hotel Risk</div>
          <div style="display:flex; gap:.55rem; align-items:center; flex-wrap:wrap; justify-content:flex-end;">
            <span class="role-pill">{html_escape(user.get('label', '—'))}</span>
            <span class="role-pill">{html_escape(status)}</span>
            <span class="role-pill">model: {html_escape(str(model_status))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return health

# -----------------------------------------------------------------------------
# Login page
# -----------------------------------------------------------------------------
def render_login() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-content">
            <h1 class="hero-title">Контроль риска<br/>отмен бронирований</h1>
            <div class="hero-subtitle">
              Загрузите бронирования, получите список заявок к проверке и оцените ожидаемые потери.<br/>
            </div>
          </div>
          <div class="hero-cards">
            <div class="feature-card">
              <div class="feature-head"><div class="feature-icon">👤</div><div class="feature-title">Менеджеру</div></div>
              <div class="feature-text">Рабочий список броней, финансовый приоритет и действие по каждой заявке.</div>
            </div>
            <div class="feature-card">
              <div class="feature-head"><div class="feature-icon">🛡</div><div class="feature-title">Админу</div></div>
              <div class="feature-text">Статус модели, СУБД, batch-история, API и пользователи системы.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, middle, right = st.columns([1, 0.72, 1])
    with middle:
        with st.container(border=True):
            st.markdown('<div class="login-title">Вход в систему</div>', unsafe_allow_html=True)
            auth_mode = st.radio("Режим", ["Вход", "Регистрация"], horizontal=True, label_visibility="collapsed")

            if auth_mode == "Вход":
                st.markdown('<div class="login-caption">Демо-доступ: <b>admin/admin</b> и <b>user/user</b>. Зарегистрированные пользователи входят как менеджеры.</div>', unsafe_allow_html=True)
                with st.form("login_form", clear_on_submit=False):
                    login = st.text_input("Логин", placeholder="admin, user или ваш логин")
                    password = st.text_input("Пароль", placeholder="пароль", type="password")
                    submitted = st.form_submit_button("Войти", use_container_width=True)
                if submitted:
                    auth = authenticate_user(login, password)
                    if auth:
                        st.session_state.auth = auth
                        ui_log("login_ok", user=auth["username"], role=auth["role"], source=auth.get("source"))
                        st.rerun()
                    else:
                        st.error("Неверный логин или пароль.")

            else:
                st.markdown('<div class="login-caption">Новый аккаунт получает роль <b>менеджера</b>. Администраторский доступ остается только у demo-логина <b>admin/admin</b>.</div>', unsafe_allow_html=True)
                with st.form("registration_form", clear_on_submit=False):
                    username = st.text_input("Логин", placeholder="например, manager01")
                    label = st.text_input("Имя в интерфейсе", placeholder="например, Менеджер смены")
                    password = st.text_input("Пароль", placeholder="минимум 4 символа", type="password")
                    password_repeat = st.text_input("Повторите пароль", placeholder="еще раз", type="password")
                    submitted = st.form_submit_button("Зарегистрироваться", use_container_width=True)
                if submitted:
                    ok, message = register_user(username, password, password_repeat, label)
                    if ok:
                        st.session_state.auth = {"username": username.strip(), "role": "user", "label": (label or username).strip(), "source": "registered"}
                        ui_log("registration_ok", user=username.strip())
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

# -----------------------------------------------------------------------------
# Manager UI
# -----------------------------------------------------------------------------
def render_manager() -> None:
    health = render_topbar()
    _, right = st.columns([1, .13])
    with right:
        logout_button()

    st.markdown(
        """
        <div class="page-hero">
          <h1 class="page-title">Брони к проверке</h1>
          <div class="page-subtitle">
            Рабочий список бронирований с наибольшими ожидаемыми потерями и действиями для менеджера.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="action-panel">', unsafe_allow_html=True)
    source_col, upload_col, currency_col, sample_col, run_col = st.columns([.85, 1.2, .9, 1.05, .85])
    with source_col:
        source = st.radio(
            "Источник данных",
            ["demo", "upload"],
            horizontal=True,
            format_func=lambda x: "Демо" if x == "demo" else "CSV",
        )
    with upload_col:
        uploaded = st.file_uploader("CSV с бронированиями", type=["csv"], disabled=(source == "demo"))
    with currency_col:
        currency_labels = [f"{code} {symbol}" for code, symbol in CURRENCY_OPTIONS.items()]
        current_code = str(st.session_state.get("currency_code", DEFAULT_CURRENCY_CODE))
        if current_code not in CURRENCY_OPTIONS:
            current_code = DEFAULT_CURRENCY_CODE
            st.session_state.currency_code = current_code
        current_label = f"{current_code} {CURRENCY_OPTIONS[current_code]}"
        selected_currency = st.selectbox(
            "Валюта сумм в CSV",
            options=currency_labels,
            index=currency_labels.index(current_label) if current_label in currency_labels else currency_labels.index(f"{DEFAULT_CURRENCY_CODE} {CURRENCY_OPTIONS[DEFAULT_CURRENCY_CODE]}"),
            help="Выберите валюту, в которой указаны adr/стоимость в загруженном CSV. Сервис не конвертирует суммы, а только меняет знак отображения.",
        )
        st.session_state.currency_code = selected_currency.split()[0]
    with sample_col:
        st.caption("Примеры CSV")
        s1, s2 = st.columns(2)
        with s1:
            st.download_button("Полный", data=read_sample(CANONICAL_SAMPLE_PATH), file_name="canonical_upload_example.csv", mime="text/csv", use_container_width=True)
        with s2:
            st.download_button("Гибкий", data=read_sample(FLEXIBLE_SAMPLE_PATH), file_name="flexible_upload_example.csv", mime="text/csv", use_container_width=True)
    with run_col:
        st.write("")
        run = st.button("Запустить анализ", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        df = load_demo() if source == "demo" else (pd.read_csv(uploaded) if uploaded is not None else pd.DataFrame())
    except Exception as exc:
        ui_log("csv_read_failed", error=exc)
        soft_alert(
            "CSV не удалось прочитать",
            "Файл должен быть обычным CSV: разделитель — запятая или стандартный CSV-формат, первая строка — названия колонок.",
            kind="bad",
            icon="!",
        )
        return

    if df.empty:
        clear_analysis_state()
        st.session_state.last_analysis_signature = None
        soft_alert("Данные не выбраны", "Загрузите CSV или выберите демо-набор.", kind="warn", icon="!")
        return

    current_signature = build_input_signature(source, uploaded, df)
    analysis_is_current = current_signature == st.session_state.get("last_analysis_signature")
    has_previous_result = any(st.session_state.get(key) is not None for key in ["predictions", "summary", "validation", "batch"])

    if not analysis_is_current:
        if has_previous_result:
            clear_analysis_state()
            ui_log("manager_input_changed_results_cleared", rows=len(df), source=source)
        if not run:
            st.caption(f"К анализу готово: {fmt_int(len(df))} бронирований. Валюта отображения: {current_currency_label()}.")
            soft_alert(
                "Данные изменились",
                "Предыдущий расчет скрыт, чтобы не показывать метрики от старого файла. Нажмите <b>«Запустить анализ»</b>, чтобы получить результат для текущего набора данных.",
                kind="warn",
                icon="!",
            )
            return

    st.caption(f"К анализу готово: {fmt_int(len(df))} бронирований. Валюта отображения: {current_currency_label()}.")

    if health.get("status") != "ok":
        soft_alert("API недоступен", f"Запустите FastAPI локально. Проверяемый адрес: <b>{html_escape(API_URL)}</b>.", kind="bad", icon="×")
        return

    if run:
        clear_analysis_state()
        try:
            with st.spinner("Считаю риск отмены и сохраняю результат в СУБД одним запуском..."):
                result = upload_batch(df, name="manager_scoring", return_predictions=True)
                st.session_state.batch = {k: v for k, v in result.items() if k not in {"summary", "predictions", "validation"}}
                st.session_state.validation = result.get("validation") or validate_df(df)
                st.session_state.summary = result.get("summary", {})
                st.session_state.predictions = normalize_predictions(pd.DataFrame(result.get("predictions", [])))
                st.session_state.manager_view = "Список к проверке"
                st.session_state.last_analysis_signature = current_signature
            ui_log("manager_scoring_done", rows=len(df), batch_id=(st.session_state.batch or {}).get("id"))
        except ValueError as exc:
            st.session_state.last_analysis_signature = None
            ui_log("csv_payload_failed", error=exc)
            soft_alert(
                "CSV не прошел проверку",
                "Файл содержит значения, которые нельзя корректно передать в сервис. Проверьте структуру CSV и используйте пример файла из блока «Примеры CSV».",
                kind="bad",
                icon="!",
            )
            return
        except requests.exceptions.RequestException as exc:
            st.session_state.last_analysis_signature = None
            ui_log("scoring_api_failed", error=exc)
            soft_alert(
                "CSV не прошел проверку",
                "Сервис не смог обработать файл. Проверьте, что в CSV есть колонки бронирования: отель, даты заезда, ночи, гости, канал/сегмент, тип тарифа и цена за ночь.",
                kind="bad",
                icon="!",
            )
            return
        except Exception as exc:
            st.session_state.last_analysis_signature = None
            ui_log("scoring_unexpected_failed", error=exc)
            soft_alert(
                "CSV не прошел проверку",
                "Файл не удалось обработать. Скачайте пример CSV и приведите названия колонок к одному из поддерживаемых форматов.",
                kind="bad",
                icon="!",
            )
            return

    pred = normalize_predictions(st.session_state.predictions) if isinstance(st.session_state.predictions, pd.DataFrame) else pd.DataFrame()
    validation = st.session_state.validation

    if validation:
        status = str(validation.get("quality_status", "green"))
        kind = "ok" if status == "green" else ("warn" if status == "yellow" else "bad")
        soft_alert(
            "Анализ завершен",
            f"Обработано строк: <b>{fmt_int(validation.get('row_count', len(df)))}</b>. "
            f"Качество CSV: <b>{html_escape(status)}</b>. {html_escape(str(validation.get('quality_message', '')))}",
            kind=kind,
            icon="✓" if kind == "ok" else "!",
        )
    if health.get("warning") or str(health.get("model_status", "")).lower() == "fallback":
        soft_alert(
            "Включена fallback-модель",
            "Сервис работает в демонстрационном rule-based режиме. Для защиты и пилота нужен артефакт <b>models/hw8_model.joblib</b>.",
            kind="bad",
            icon="!",
        )

    if pred.empty:
        st.info("Нажмите «Запустить анализ», чтобы получить рабочий список.")
        return

    prob_col = "cancellation_probability" if "cancellation_probability" in pred.columns else "risk_score"
    if "actionable" in pred.columns:
        actionable_mask = pred["actionable"].astype(bool)
    else:
        actionable_mask = pred["risk_category"].isin(["High", "Critical"]) & (pred.get("expected_loss", 0) >= 100)
    total_revenue = float(pred["booking_value"].sum()) if "booking_value" in pred else 0.0
    expected_loss_all = float(pred["expected_loss"].sum()) if "expected_loss" in pred else 0.0
    avg_risk = float(pred[prob_col].mean()) if prob_col in pred else 0.0

    tariff_protected = 0.0
    if {"booking_value", "deposit_loss_factor", prob_col}.issubset(pred.columns):
        tariff_protected = float((
            pd.to_numeric(pred[prob_col], errors="coerce").fillna(0).clip(lower=0, upper=1)
            * pd.to_numeric(pred["booking_value"], errors="coerce").fillna(0).clip(lower=0)
            * (1 - pd.to_numeric(pred["deposit_loss_factor"], errors="coerce").fillna(1).clip(lower=0, upper=1))
        ).sum())

    # Working list for a manager: all financially meaningful High/Critical bookings.
    # It is not optimized away by a hypothetical ROI scenario, because the user
    # needs a complete priority queue, not only bookings that pass a manual cost assumption.
    actionable_pred = pred.loc[actionable_mask].copy().sort_values("business_priority_score", ascending=False)
    recommended = actionable_pred.copy()
    review_count = int(len(recommended))
    review_expected_loss = float(recommended["expected_loss"].sum()) if "expected_loss" in recommended else 0.0
    protected = float(review_expected_loss * float(st.session_state.success_rate))
    cost = float(review_count * float(st.session_state.cost_per_action))
    net = float(protected - cost)
    loss_concentration = (review_expected_loss / expected_loss_all) if expected_loss_all > 0 else 0.0
    st.session_state["recommended_review_ids"] = recommended["booking_id"].astype(str).tolist() if "booking_id" in recommended.columns else []

    st.caption(
        f"Денежные показатели считаются в валюте исходного CSV: {current_currency_label()}. "
        "Конвертация валюты не выполняется."
    )

    render_metric_grid([
        ("Сумма всех броней", fmt_money(total_revenue), "если все выбранные брони состоятся"),
        ("Ожидаемые потери", fmt_money(expected_loss_all), "вероятность отмены × сумма брони"),
        ("Средний риск", fmt_pct(avg_risk), "средняя вероятность отмены"),
        ("К проверке", fmt_int(review_count), "приоритетные брони"),
        ("Потери в списке", fmt_money(review_expected_loss), "ожидаемые потери выбранных броней"),
        ("Доля риска в списке", fmt_pct(loss_concentration), "концентрация ожидаемых потерь"),
    ])

    with st.expander("Параметры оценки действий", expanded=False):
        with st.form("scenario_settings"):
            a, b = st.columns(2)
            prevented_percent = a.slider(
                "Процент потерь, который можно предотвратить",
                min_value=5,
                max_value=70,
                value=int(round(float(st.session_state.success_rate) * 100)),
                step=5,
            )
            cost_per_action = b.number_input(
                f"Стоимость проверки одной брони, {current_currency_label()}",
                min_value=0.0,
                max_value=1_000_000.0,
                value=float(st.session_state.get("cost_per_action", DEFAULT_COST_PER_ACTION)),
                step=1.0,
            )
            if st.form_submit_button("Пересчитать", use_container_width=True):
                st.session_state.success_rate = prevented_percent / 100.0
                st.session_state.cost_per_action = cost_per_action
                st.rerun()
        e1, e2, e3 = st.columns(3)
        e1.metric("Можно сохранить", fmt_money(protected))
        e2.metric("Стоимость проверки", fmt_money(cost))
        e3.metric("Оценка действий", fmt_money(net))

    view_name = nav_radio("Раздел", ["Список к проверке", "Обзор", "Сегменты", "Карточка брони", "История"], "manager_view")

    if view_name == "Обзор":
        scatter = pred.copy().reset_index(drop=True)
        scatter["Вероятность отмены"] = pd.to_numeric(scatter[prob_col], errors="coerce").fillna(0)
        scatter["Сумма брони"] = pd.to_numeric(scatter.get("booking_value", 0), errors="coerce").fillna(0).clip(lower=0)
        scatter["Категория риска"] = scatter.get("risk_category", "—")
        scatter["ID брони"] = scatter.get("booking_id", scatter.index + 1).astype(str)
        hover_cols = [c for c in ["ID брони", "hotel", "market_segment", "deposit_type", "Сумма брони"] if c in scatter.columns]
        fig = px.scatter(
            scatter,
            x="Вероятность отмены",
            y="Сумма брони",
            color="Категория риска",
            color_discrete_map={"Low": "#16A34A", "Medium": "#2563EB", "High": "#D97706", "Critical": "#DC2626"},
            hover_data=hover_cols,
            title="Распределение броней по риску",
            template="plotly_white",
        )
        fig.update_traces(marker=dict(size=7, opacity=0.58, line=dict(width=0)))
        fig.update_layout(
            height=520,
            margin=dict(l=20, r=20, t=60, b=20),
            font_color="#111827",
            xaxis_title="Вероятность отмены",
            yaxis_title="Сумма брони",
            legend_title_text="Категория",
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    elif view_name == "Список к проверке":
        view = pred.copy()
        recommended_ids = set(st.session_state.get("recommended_review_ids", []))
        if recommended_ids and "booking_id" in view.columns:
            view = view[view["booking_id"].astype(str).isin(recommended_ids)].copy()
        else:
            view = view.head(0).copy()
        view = view.sort_values("business_priority_score", ascending=False) if "business_priority_score" in view else view

        arrival_base_date = None
        if "arrival_date" in pred:
            full_arrival = pd.to_datetime(pred["arrival_date"], errors="coerce")
            if full_arrival.notna().any():
                arrival_base_date = full_arrival.min()

        def filter_options(frame: pd.DataFrame, column: str) -> list[str]:
            if column not in frame:
                return []
            return sorted(frame[column].dropna().astype(str).unique().tolist())

        risk_order = ["Critical", "High", "Medium", "Low"]
        risk_options = [x for x in risk_order if "risk_category" in view and x in set(view["risk_category"].dropna().astype(str))]
        risk_options += [x for x in filter_options(view, "risk_category") if x not in risk_options]

        with st.expander("Фильтры списка", expanded=True):
            f1, f2, f3 = st.columns([1.15, 1, 1])
            horizon = f1.selectbox("Заезд от ближайшей даты в файле", ["Все", "7 дней", "14 дней", "30 дней"], index=0)
            hotel_filter = f2.multiselect("Отель", filter_options(view, "hotel"), placeholder="Все отели")
            market_filter = f3.multiselect("Сегмент", filter_options(view, "market_segment"), placeholder="Все сегменты")
            f4, f5, f6 = st.columns([1, 1, 1])
            channel_filter = f4.multiselect("Канал", filter_options(view, "distribution_channel"), placeholder="Все каналы")
            category_filter = f5.multiselect("Категория риска", risk_options, placeholder="Все категории")
            min_loss = f6.number_input("Мин. ожидаемая потеря", min_value=0.0, value=0.0, step=100.0)
            if arrival_base_date is not None:
                st.caption(f"Горизонт заезда считается от ближайшей даты заезда в загруженном CSV: {arrival_base_date.date()}.")

        if hotel_filter and "hotel" in view:
            view = view[view["hotel"].astype(str).isin(hotel_filter)]
        if market_filter and "market_segment" in view:
            view = view[view["market_segment"].astype(str).isin(market_filter)]
        if channel_filter and "distribution_channel" in view:
            view = view[view["distribution_channel"].astype(str).isin(channel_filter)]
        if category_filter and "risk_category" in view:
            view = view[view["risk_category"].astype(str).isin(category_filter)]
        if min_loss > 0 and "expected_loss" in view:
            view = view[pd.to_numeric(view["expected_loss"], errors="coerce").fillna(0) >= float(min_loss)]
        if horizon != "Все" and "arrival_date" in view and arrival_base_date is not None:
            try:
                horizon_days = int(horizon.split()[0])
                arrival = pd.to_datetime(view["arrival_date"], errors="coerce")
                view = view[(arrival.notna()) & (arrival >= arrival_base_date) & (arrival <= arrival_base_date + pd.Timedelta(days=horizon_days))]
            except Exception:
                pass

        if view.empty:
            st.info("В этом файле нет броней, которые требуют ручной проверки по выбранным фильтрам.")
            return
        table_cols = ["booking_id", "hotel", "market_segment", "distribution_channel", "total_nights", "booking_value", prob_col, "expected_loss", "risk_category", "recommended_action"]
        display = view[[c for c in table_cols if c in view.columns]].rename(columns={
            "booking_id": "ID брони", "hotel": "Отель", "market_segment": "Сегмент", "distribution_channel": "Канал", "total_nights": "Ночи",
            "booking_value": "Сумма брони", prob_col: "Риск отмены", "risk_category": "Категория", "expected_loss": "Ожидаемая потеря",
            "recommended_action": "Что сделать",
        })
        column_config = {
            "Сумма брони": st.column_config.NumberColumn("Сумма брони", format=money_column_format()),
            "Риск отмены": st.column_config.NumberColumn("Риск отмены", format="%.1%"),
            "Ожидаемая потеря": st.column_config.NumberColumn("Ожидаемая потеря", format=money_column_format()),
            "Ночи": st.column_config.NumberColumn("Ночи", format="%d"),
        }
        st.dataframe(display, width="stretch", hide_index=True, height=500, column_config=column_config)
        st.download_button("Скачать список CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="hotel_risk_top_list.csv", mime="text/csv")

    elif view_name == "Сегменты":
        left, right = st.columns(2)
        with left:
            if "market_segment" in pred.columns:
                seg = (
                    pred.groupby("market_segment", dropna=False)
                    .agg(Брони=("booking_id", "count"), Средний_риск=(prob_col, "mean"), Ожидаемая_потеря=("expected_loss", "sum"))
                    .reset_index()
                    .rename(columns={"market_segment": "Сегмент продаж"})
                    .sort_values("Ожидаемая_потеря", ascending=False)
                    .head(10)
                )
                fig = px.bar(
                    seg,
                    x="Ожидаемая_потеря",
                    y="Сегмент продаж",
                    orientation="h",
                    title="Где сосредоточена ожидаемая потеря",
                    template="plotly_white",
                )
                fig.update_traces(marker_color="#175CD3")
                fig.update_layout(
                    height=420,
                    yaxis={"categoryorder": "total ascending"},
                    margin=dict(l=20, r=20, t=55, b=20),
                    font_color="#111827",
                    xaxis_title="Ожидаемая потеря",
                    yaxis_title="Сегмент продаж",
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with right:
            if "deposit_type" in pred.columns:
                dep = (
                    pred.groupby("deposit_type", dropna=False)
                    .agg(Брони=("booking_id", "count"), Средний_риск=(prob_col, "mean"), Сумма_броней=("booking_value", "sum"), Ожидаемая_потеря=("expected_loss", "sum"))
                    .reset_index()
                    .rename(columns={"deposit_type": "Тип тарифа"})
                )
                dep["Средний риск"] = dep.pop("Средний_риск").map(lambda x: f"{float(x):.1%}")
                dep = dep.rename(columns={"Сумма_броней": "Сумма броней", "Ожидаемая_потеря": "Ожидаемая потеря"})
                for money_col in ["Сумма броней", "Ожидаемая потеря"]:
                    dep[money_col] = dep[money_col].map(lambda x: fmt_money(x))
                st.dataframe(dep[["Тип тарифа", "Брони", "Средний риск", "Сумма броней", "Ожидаемая потеря"]], width="stretch", hide_index=True)

    elif view_name == "Карточка брони":
        sorted_pred = pred.sort_values("business_priority_score", ascending=False).reset_index(drop=True)
        booking_id = st.selectbox("Выберите бронь", sorted_pred["booking_id"].astype(str).tolist())
        row = sorted_pred[sorted_pred["booking_id"].astype(str).eq(str(booking_id))].iloc[0]
        render_metric_grid([
            ("Риск отмены", fmt_pct(row[prob_col]), "расчетная вероятность"),
            ("Ожидаемая потеря", fmt_money(row.get("expected_loss", 0)), "расчетная денежная оценка"),
            ("Категория", str(row.get("risk_category", "—")), "уровень внимания"),
            ("Сумма брони", fmt_money(row.get("booking_value", 0)), "цена за ночь × ночи"),
        ], compact=True)
        st.markdown(f"<div class='panel'><div class='panel-title'>Рекомендованное действие</div><div class='panel-text'>{html_escape(row.get('recommended_action', '—'))}</div></div>", unsafe_allow_html=True)
        with st.expander("Бизнес-факторы риска", expanded=True):
            factors = row.get("top_factors", [])
            if isinstance(factors, str):
                factors = [factors]
            if not factors:
                st.write("Бизнес-факторы риска не переданы сервисом.")
            for factor in factors:
                st.write(f"• {factor}")

    elif view_name == "История":
        st.caption("Сохраненные запуски анализа.")
        if st.session_state.batch:
            soft_alert("Последний результат сохранен", f"Batch ID: <b>{st.session_state.batch.get('id')}</b>. Сохранены бронирования, прогнозы и агрегированные суммы риска.", icon="✓")
        batches = api_get("/api/v1/batches?limit=20")
        if isinstance(batches, list) and batches:
            hist = pd.DataFrame(batches).rename(columns={"id": "Batch ID", "name": "Название", "status": "Статус", "row_count": "Строк", "high_risk_count": "Требуют внимания", "total_expected_loss": "Ожидаемая потеря", "created_at": "Создан"})
            cols = [c for c in ["Batch ID", "Название", "Статус", "Строк", "Требуют внимания", "Ожидаемая потеря", "Создан"] if c in hist.columns]
            st.dataframe(hist[cols], width="stretch", hide_index=True)
        else:
            st.info("История пока пустая.")

# -----------------------------------------------------------------------------
# Admin UI
# -----------------------------------------------------------------------------
def render_admin() -> None:
    health = render_topbar()
    _, right = st.columns([1, .13])
    with right:
        logout_button()

    st.markdown(
        """
        <div class="page-hero">
          <h1 class="page-title">Админская панель</h1>
          <div class="page-subtitle">Технический контур для проверки ДЗ: API, модель, СУБД, batch-история, схема данных, логи и worker-сценарий.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_info = api_get("/api/v1/model-info") if health.get("status") == "ok" else {}
    overview = api_get("/api/v1/analytics/overview") if health.get("status") == "ok" else {}
    metrics = model_info.get("metrics", {}) if isinstance(model_info, dict) else {}
    model_name = str(model_info.get("model_name", "—"))
    if len(model_name) > 34:
        model_name_short = model_name[:31] + "…"
    else:
        model_name_short = model_name

    render_info_stack([
        ("Модель", model_name, "активный артефакт"),
        ("СУБД", str(health.get("database", "—")), "SQLite локально / PostgreSQL в Docker"),
        ("Batch-запусков", fmt_int(overview.get("batches_total", 0) if isinstance(overview, dict) else 0), "история скорингов"),
    ])

    view_name = nav_radio("Раздел", ["Статус", "СУБД", "API", "Пользователи", "Логи"], "admin_view")

    if view_name == "Статус":
        left, right_col = st.columns(2)
        with left:
            st.markdown("### Health")
            st.json(health)
        with right_col:
            st.markdown("### Метрики модели")
            st.json(metrics)
        with st.expander("Полная информация о модели", expanded=False):
            st.json(model_info)

    elif view_name == "СУБД":
        batches = api_get("/api/v1/batches?limit=50") if health.get("status") == "ok" else []
        if isinstance(batches, list) and batches:
            st.dataframe(pd.DataFrame(batches), width="stretch", hide_index=True)
        else:
            st.info("Сохраненных batch-запусков пока нет.")
        with st.expander("Агрегированная статистика", expanded=False):
            st.json(overview)

    elif view_name == "API":
        schema = api_get("/api/v1/schema") if health.get("status") == "ok" else {}
        st.markdown("### Основные endpoint'ы")
        st.code(
            f"""GET  {API_URL}/health
GET  {API_URL}/api/v1/model-info
GET  {API_URL}/api/v1/schema
POST {API_URL}/api/v1/validate
POST {API_URL}/api/v1/score
POST {API_URL}/api/v1/batches/upload
GET  {API_URL}/api/v1/batches
GET  {API_URL}/api/v1/batches/{{batch_id}}/export
POST {API_URL}/api/v1/jobs/score-file"""
        )
        with st.expander("Схема входных данных", expanded=False):
            st.json(schema)

    elif view_name == "Пользователи":
        st.markdown("### Пользователи")
        st.caption("Зарегистрированные аккаунты получают роль менеджера. Демо-администратор: admin/admin.")
        st.dataframe(users_for_admin_table(), width="stretch", hide_index=True)
        st.caption(f"Файл локального хранилища пользователей: {USER_STORE_PATH}")

    elif view_name == "Логи":
        st.markdown("### Локальные команды запуска")
        st.code(
            """uvicorn hotel_risk.api.main:app --host 127.0.0.1 --port 8000 --reload
export HOTEL_RISK_API_URL=http://127.0.0.1:8000
streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501"""
        )
        st.markdown("### UI log")
        st.code("\n".join(st.session_state.ui_logs[-40:]) or "Пока нет событий UI.")

# -----------------------------------------------------------------------------
# Main route
# -----------------------------------------------------------------------------
if not st.session_state.auth:
    render_login()
elif st.session_state.auth["role"] == "admin":
    render_admin()
else:
    render_manager()
