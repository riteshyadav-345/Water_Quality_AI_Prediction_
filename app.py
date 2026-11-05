"""
AquaCheck - AI Water Potability Predictor (SDG 6: Clean Water and Sanitation)
Run:  streamlit run app.py
"""
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AquaCheck - SDG 6", page_icon="💧", layout="wide")
appearance = st.sidebar.radio("Appearance", ["Light", "Dark"], horizontal=True, key="appearance")

st.markdown("""
<style>
:root {
  color-scheme: light;
  --text-color: #17384d;
  --text-secondary: #4f6b7b;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background-color: #f5fafb !important;
  color: var(--text-color) !important;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp small,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stRadio"] label,
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stSelectbox"] label,
.stApp [data-testid="stNumberInput"] label,
.stApp [data-testid="stFileUploader"] label {
  color: var(--text-color) !important;
}
.stApp [data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] *,
.stApp [data-baseweb="input"] *, .stApp [data-baseweb="textarea"] * {
  color: #17384d !important;
  -webkit-text-fill-color: #17384d !important;
}
.stApp [data-testid="stSidebar"] *,
.stApp [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
  color: #17384d;
}
.stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #4f6b7b !important; }
.stApp [data-testid="stAlert"] p, .stApp [data-testid="stAlert"] li { color: #17384d !important; }
.stApp .stTabs [data-baseweb="tab"] { color: #36566a !important; }
.stApp .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff !important; }
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root {
  --aqua-ink: #102f46;
  --aqua-muted: #5f7f91;
  --aqua-blue: #087eaa;
  --aqua-teal: #11a89b;
  --aqua-pale: #e8f7f8;
}
.stApp {
  background: radial-gradient(ellipse at 8% 0%, rgba(146, 224, 226, .24), transparent 34%),
              linear-gradient(180deg, #f4fbfc 0%, #f8fbfd 45%, #f3f8fa 100%);
  color: var(--aqua-ink);
  font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { color: var(--aqua-ink); font-family: 'Manrope', sans-serif; letter-spacing: -.025em; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #e7f6f7 0%, #f1fafb 65%, #f8fbfc 100%);
  border-right: 1px solid #d5e9ed;
}
[data-testid="stSidebar"] h1 { color: #075b78; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: white; border: 1px solid #d5e9ed; border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 4px 14px rgba(16, 68, 91, .05);
}
.aqua-hero {
  position: relative; overflow: hidden; margin: 8px 0 22px; padding: 27px 32px;
  border-radius: 22px; color: white;
  background: linear-gradient(115deg, #075b78 0%, #087eaa 52%, #11a89b 100%);
  box-shadow: 0 14px 34px rgba(5, 93, 123, .18);
}
.aqua-hero:after {
  content: ''; position: absolute; width: 230px; height: 230px; border-radius: 50%;
  right: 7%; top: -145px; border: 1px solid rgba(255,255,255,.2);
  box-shadow: 0 0 0 24px rgba(255,255,255,.045), 0 0 0 50px rgba(255,255,255,.035);
}
.aqua-hero h1 { margin: 0 0 7px; color: white; font-size: clamp(1.65rem, 3vw, 2.3rem); }
.aqua-hero p { margin: 0; max-width: 720px; color: rgba(255,255,255,.83); font-size: 1rem; }
.aqua-kicker { margin-bottom: 8px; color: #b9f5eb; font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.stTabs [data-baseweb="tab-list"] {
  gap: 9px; padding: 8px; border: 1px solid #dcebed; border-radius: 17px;
  background: rgba(255,255,255,.82); box-shadow: 0 5px 18px rgba(16,68,91,.055);
}
.stTabs [data-baseweb="tab"] {
  height: 43px; padding: 0 17px; border-radius: 11px; border: 1px solid transparent;
  color: #537487; background: transparent; font-weight: 600; transition: all .18s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #075b78; background: #edf8f8; }
.stTabs [aria-selected="true"] {
  color: white !important; background: linear-gradient(110deg, #087eaa, #11a89b) !important;
  box-shadow: 0 5px 13px rgba(8,126,170,.2);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
[data-testid="stMetric"] {
  border: 1px solid #dcebed; border-radius: 15px; padding: 15px 17px;
  background: rgba(255,255,255,.9); box-shadow: 0 4px 15px rgba(16,68,91,.045);
}
[data-testid="stMetricLabel"] { color: var(--aqua-muted); font-weight: 600; }
[data-testid="stMetricValue"] { color: #075b78; font-family: 'Manrope', sans-serif; }
.stButton > button, .stDownloadButton > button {
  border: 1px solid #b8dfe2; border-radius: 11px; color: #075b78; background: white;
  font-weight: 700; transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #0e9c9c; color: #075b78; background: #e9f8f7; transform: translateY(-1px);
}
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #dcebed; border-radius: 13px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid #dcebed; border-radius: 13px; background: rgba(255,255,255,.76); }
div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
  border-radius: 10px; border-color: #cfe2e7; background: rgba(255,255,255,.92);
}
@media (max-width: 700px) {
  .aqua-hero { padding: 22px; border-radius: 17px; }
  .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
  .stTabs [data-baseweb="tab"] { padding: 0 12px; white-space: nowrap; }
}
</style>
""", unsafe_allow_html=True)

if appearance == "Dark":
    st.markdown("""
    <style>
    :root { color-scheme: dark; --aqua-ink: #e5f2f7; --aqua-muted: #a9c2ce; --text-color: #e5f2f7; --text-secondary: #a9c2ce; }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
      background: radial-gradient(ellipse at 8% 0%, rgba(14,103,128,.2), transparent 35%), #0b1721 !important;
      color: #e5f2f7 !important;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label,
    .stApp small, .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stRadio"] label, .stApp [data-testid="stCheckbox"] label,
    .stApp [data-testid="stSelectbox"] label, .stApp [data-testid="stNumberInput"] label,
    .stApp [data-testid="stFileUploader"] label { color: #e5f2f7 !important; }
    .stApp [data-testid="stCaptionContainer"] { color: #a9c2ce !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#102534,#0d1d29 70%,#0b1721) !important; border-color: #254454 !important; }
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color: #e5f2f7 !important; }
    .stApp input, .stApp textarea, .stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] *,
    .stApp [data-baseweb="textarea"] * { color: #e5f2f7 !important; -webkit-text-fill-color: #e5f2f7 !important; }
    .stApp div[data-baseweb="select"] > div, .stApp [data-testid="stNumberInput"] input,
    .stApp [data-baseweb="input"] > div { background: #142633 !important; border-color: #345261 !important; }
    [data-testid="stMetric"], [data-testid="stSidebar"] [data-testid="stMetric"] {
      background: #132633 !important; border-color: #294654 !important; box-shadow: 0 5px 18px rgba(0,0,0,.16) !important;
    }
    [data-testid="stMetricLabel"] { color: #a9c2ce !important; }
    [data-testid="stMetricValue"] { color: #8fe1d9 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #122430 !important; border-color: #294654 !important; }
    .stTabs [data-baseweb="tab"] { color: #c1d5de !important; }
    .stTabs [data-baseweb="tab"]:hover { background: #203b49 !important; color: #ffffff !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #ffffff !important; }
    .stButton > button, .stDownloadButton > button {
      background: #142b39 !important; border-color: #3a6574 !important; color: #e5f2f7 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #1d3d4b !important; color: #ffffff !important; }
    [data-testid="stExpander"] { background: #122430 !important; border-color: #294654 !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border-color: #294654 !important; }
    [data-testid="stAlert"] { background: #162a37 !important; border: 1px solid #345261 !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] li { color: #e5f2f7 !important; }
    [data-testid="stProgressBar"] div[role="progressbar"] { background: #203947 !important; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 1. Parameters: safe range (lo, hi), scale s, weight w for the WQI, default d
#    Limits follow BIS IS 10500 / WHO drinking-water guidelines.
# ----------------------------------------------------------------------------
PARAMS = [
    dict(k="ph", n="pH", u="", lo=6.5, hi=8.5, s=1.5, w=.18, mn=0.0, mx=14.0, st=.1, d=7.4),
    dict(k="tds", n="TDS", u="mg/L", lo=0, hi=500, s=500, w=.18, mn=0.0, mx=5000.0, st=10.0, d=220.0),
    dict(k="hard", n="Hardness", u="mg/L", lo=0, hi=300, s=300, w=.10, mn=0.0, mx=1000.0, st=5.0, d=150.0),
    dict(k="chl", n="Chloramines", u="mg/L", lo=0, hi=4, s=4, w=.10, mn=0.0, mx=15.0, st=.1, d=2.5),
    dict(k="sul", n="Sulfate", u="mg/L", lo=0, hi=250, s=250, w=.10, mn=0.0, mx=1000.0, st=5.0, d=60.0),
    dict(k="cond", n="Conductivity", u="µS/cm", lo=0, hi=800, s=800, w=.08, mn=0.0, mx=5000.0, st=10.0, d=350.0),
    dict(k="toc", n="Organic carbon", u="mg/L", lo=0, hi=4, s=4, w=.08, mn=0.0, mx=30.0, st=.1, d=1.5),
    dict(k="tur", n="Turbidity", u="NTU", lo=0, hi=5, s=5, w=.18, mn=0.0, mx=100.0, st=.1, d=0.8),
]
KEYS = [p["k"] for p in PARAMS]

PRESETS = {
    "Tap water": [7.4, 220, 150, 2.5, 60, 350, 1.5, .8],
    "RO water": [6.8, 40, 20, 1, 10, 60, .5, .2],
    "Borewell": [7.9, 780, 420, 3, 320, 1100, 3, 2],
    "River": [8.9, 600, 280, 5.2, 180, 900, 7, 22],
    "Pond": [9.2, 1200, 350, 6, 300, 1500, 12, 45],
}

TREAT = {
    "ph_hi": "pH is too high: use RO or acid dosing.",
    "ph_lo": "pH is too low: neutralising filter or a little soda ash.",
    "tds": "Use an RO purifier or mix with low-TDS water.",
    "hard": "Use a water softener; boiling helps for temporary hardness.",
    "chl": "Use an activated-carbon filter or let water stand uncovered.",
    "sul": "Use RO or anion-exchange treatment.",
    "cond": "Many dissolved salts: use RO or de-ionisation.",
    "toc": "Use coagulation plus activated carbon, then disinfect.",
    "tur": "Let it settle, add alum, then sand or cloth filter.",
}

KAGGLE_MAP = {"ph": "ph", "Hardness": "hard", "Solids": "tds", "Chloramines": "chl",
              "Sulfate": "sul", "Conductivity": "cond", "Organic_carbon": "toc", "Turbidity": "tur"}


def dev(p, v):
    """How far a value is outside its safe range, as a fraction of the scale (0 = safe)."""
    v = np.asarray(v, dtype=float)
    return np.clip(np.maximum(0, np.maximum((p["lo"] - v) / p["s"], (v - p["hi"]) / p["s"])), 0, 3)


def features(df):
    return np.column_stack([dev(p, df[p["k"]].values) for p in PARAMS])


# ----------------------------------------------------------------------------
# 2. Data + model
# ----------------------------------------------------------------------------
def make_data(n=1000, seed=7):
    rng = np.random.default_rng(seed)
    cols = {}
    for p in PARAMS:
        lo = p["lo"] or p["hi"] * 0.1
        inside = rng.random(n) < 0.92
        good = rng.uniform(lo, p["hi"], n)
        if p["k"] == "ph":
            bad = np.where(rng.random(n) < .5, rng.uniform(4, 6.5, n), rng.uniform(8.5, 11.5, n))
        else:
            bad = p["hi"] * (1 + rng.random(n) * 1.5)
        cols[p["k"]] = np.where(inside, good, bad)
    df = pd.DataFrame(cols)
    score = sum(p["w"] * dev(p, df[p["k"]].values) for p in PARAMS)
    y = (score < .03).astype(int)
    flip = rng.random(n) < .04                       # 4% label noise
    df["potable"] = np.where(flip, 1 - y, y)
    return df


@st.cache_resource
def train_synthetic():
    df = make_data()
    X, y = features(df), df["potable"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=7)
    m = LogisticRegression(C=10, max_iter=2000).fit(Xtr, ytr)
    return dict(kind="synthetic", model=m, train=m.score(Xtr, ytr), test=m.score(Xte, yte),
                cm=confusion_matrix(yte, m.predict(Xte)), imp=np.abs(m.coef_[0]), n=len(df))


@st.cache_resource
def train_csv(raw: bytes):
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        return dict(error=f"Could not read the CSV: {exc}")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=KAGGLE_MAP)
    need = KEYS + ["Potability"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        return dict(error=f"Missing columns: {miss}. Use the Kaggle 'water_potability.csv' format.")
    df = df[need].copy()
    for col in need:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Potability"])
    df[KEYS] = df[KEYS].fillna(df[KEYS].median())
    df = df.dropna(subset=KEYS)
    df = df[np.isfinite(df[KEYS]).all(axis=1)]
    df = df[df["Potability"].isin([0, 1])]
    if len(df) < 10:
        return dict(error="The CSV needs at least 10 usable rows after removing invalid values.")
    if df["Potability"].nunique() < 2:
        return dict(error="The Potability column must contain both 0 (not potable) and 1 (potable).")
    if df["Potability"].value_counts().min() < 2:
        return dict(error="Each Potability class needs at least 2 rows for a stratified train/test split.")
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            df[KEYS].values, df["Potability"].astype(int).values,
            test_size=.2, random_state=7, stratify=df["Potability"]
        )
    except ValueError as exc:
        return dict(error=f"Could not create a train/test split: {exc}")
    m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=7).fit(Xtr, ytr)
    return dict(kind="csv", model=m, train=m.score(Xtr, ytr), test=m.score(Xte, yte),
                cm=confusion_matrix(yte, m.predict(Xte)), imp=m.feature_importances_, n=len(df))


def predict(bundle, values):
    row = pd.DataFrame([values], columns=KEYS)
    X = features(row) if bundle["kind"] == "synthetic" else row.values
    return float(bundle["model"].predict_proba(X)[0, 1])


# ----------------------------------------------------------------------------
# 3. Helpers
# ----------------------------------------------------------------------------
def status(x):
    return ("🟢 Safe", "Safe") if x == 0 else ("🟡 Borderline", "Borderline") if x < .2 else ("🔴 Unsafe", "Unsafe")


def limit_text(p):
    return "6.5 to 8.5" if p["k"] == "ph" else f"up to {p['hi']}"


def apply_preset():
    name = st.session_state["preset"]
    if name in PRESETS:
        for p, v in zip(PARAMS, PRESETS[name]):
            st.session_state[p["k"]] = float(v)


# session defaults
for p in PARAMS:
    st.session_state.setdefault(p["k"], float(p["d"]))
st.session_state.setdefault("log", [])

# ----------------------------------------------------------------------------
# 4. Sidebar: model choice
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("💧 AquaCheck")
    st.caption("AI water potability predictor\nSDG 6: Clean Water & Sanitation")
    mode = st.radio("Training data", ["Synthetic (BIS/WHO rules)", "Upload Kaggle CSV"])
    bundle = train_synthetic()
    if mode.startswith("Upload"):
        up = st.file_uploader("water_potability.csv", type="csv")
        if up is not None:
            b = train_csv(up.getvalue())
            if "error" in b:
                st.error(b["error"])
            else:
                bundle = b
                st.success("Model trained on your CSV (Random Forest).")
        else:
            st.info("No file yet. Using the synthetic model.")
    st.metric("Test accuracy", f"{bundle['test'] * 100:.1f}%")
    st.caption("Educational tool only. Confirm drinking water with a certified lab.")

st.markdown("""
<section class="aqua-hero">
  <div class="aqua-kicker">AquaCheck · Clean water, clearer choices</div>
  <h1>Water quality &amp; home water toolkit</h1>
  <p>Explore a water-quality screening estimate, practical conservation ideas, and simple household water guidance.</p>
</section>
""", unsafe_allow_html=True)
tabs = st.tabs(["💧  Water Check", "🌿  Save Water", "⚙️  Purifier Guide", "🌧️  Rain Harvest", "🧼  Hygiene", "✨  About AI"])

# ----------------------------------------------------------------------------
# 5. Predict
# ----------------------------------------------------------------------------
with tabs[0]:
    st.selectbox("Quick sample", ["Custom"] + list(PRESETS), key="preset", on_change=apply_preset)
    cols = st.columns(4)
    for i, p in enumerate(PARAMS):
        with cols[i % 4]:
            st.number_input(f"{p['n']} {('(' + p['u'] + ')') if p['u'] else ''}", min_value=p["mn"], max_value=p["mx"],
                            step=float(p["st"]), key=p["k"], help=f"Safe: {limit_text(p)}")
    vals = [float(st.session_state[k]) for k in KEYS]
    x = [float(dev(p, v)) for p, v in zip(PARAMS, vals)]
    prob = predict(bundle, vals)
    wqi = round(max(0, 100 - 100 * sum(p["w"] * min(2 * xi, 1) for p, xi in zip(PARAMS, x))))
    cat = "Excellent" if wqi >= 80 else "Good" if wqi >= 60 else "Poor" if wqi >= 40 else "Very poor"

    st.divider()
    if prob >= .7:
        st.success("✅ Likely potable (screening estimate; verify with a certified lab)")
        verdict = "Potable"
    elif prob >= .4:
        st.warning("⚠️ Uncertain screening result: treat and verify before drinking")
        verdict = "Doubtful"
    else:
        st.error("❌ Unlikely potable: do not drink without treatment and verification")
        verdict = "Not potable"

    m1, m2 = st.columns(2)
    m1.metric("AI confidence (potable)", f"{prob * 100:.0f}%")
    m1.progress(min(max(prob, 0.0), 1.0))
    m2.metric("Water Quality Index", f"{wqi}/100", cat)
    m2.progress(wqi / 100)

    if x[7] > 0 or x[6] > 0:
        st.info("🦠 Turbidity or organic matter can shelter germs (diarrhoea, typhoid, hepatitis A risk).")
    elif x[1] > 0 or x[4] > 0:
        st.info("⚠️ High salts and sulfate can upset the stomach and damage pipes.")
    elif x[0] > 0:
        st.info("pH outside the safe range can taste bad and corrode pipes.")

    table = pd.DataFrame({
        "Parameter": [p["n"] for p in PARAMS],
        "Value": [f"{v} {p['u']}".strip() for p, v in zip(PARAMS, vals)],
        "Safe limit": [limit_text(p) for p in PARAMS],
        "Status": [status(xi)[0] for xi in x],
    })
    left, right = st.columns([3, 2])
    left.dataframe(table, hide_index=True)
    ratio = pd.Series({p["n"]: min(v / p["hi"] * 100, 300) for p, v in zip(PARAMS, vals) if p["k"] != "ph"})
    right.caption("Value as % of safe limit (100 = limit)")
    right.bar_chart(ratio)

    st.subheader("Suggested treatment")
    tips = []
    for p, v, xi in zip(PARAMS, vals, x):
        if xi > 0:
            key = ("ph_lo" if v < p["lo"] else "ph_hi") if p["k"] == "ph" else p["k"]
            tips.append(f"**{p['n']}:** {TREAT[key]}")
    if tips:
        tips.append("**General:** boil for 1 minute or use a UV purifier, then confirm with a lab test.")
        st.markdown("\n".join(f"- {t}" for t in tips))
    else:
        st.write("No treatment needed 🎉 Store water in a clean, covered container.")

    report = "AquaCheck screening report (not a safety certification)\n" + "\n".join(
        f"{p['n']}: {v} {p['u']} [{status(xi)[1]}]" for p, v, xi in zip(PARAMS, vals, x)
    ) + f"\nVerdict: {verdict} ({prob * 100:.0f}% potable)\nWQI: {wqi}/100 ({cat})"

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("＋ Save to log"):
        st.session_state["log"].append({"Time": datetime.now().strftime("%H:%M:%S"), "Verdict": verdict,
                                        "Potable %": round(prob * 100), "WQI": wqi,
                                        **{p["n"]: v for p, v in zip(PARAMS, vals)}})
    b2.download_button("📋 Download report", report, file_name="aquacheck_report.txt")

    if st.session_state["log"]:
        st.subheader("Test log")
        log = pd.DataFrame(st.session_state["log"])
        st.dataframe(log, hide_index=True)
        st.download_button("Download log (CSV)", log.to_csv(index=False), file_name="aquacheck_log.csv")

# ----------------------------------------------------------------------------
# 6. Water saver (SDG 6.4)
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Daily water-use calculator")
    c = st.columns(3)
    n = int(c[0].number_input("People in home", 1, 50, 4))
    shw = c[1].number_input("Shower minutes (per person)", 0.0, 60.0, 8.0)
    fl = c[2].number_input("Toilet flushes (per person)", 0.0, 30.0, 5.0)
    c = st.columns(3)
    br = c[0].number_input("Tap running while brushing (min, per person)", 0.0, 20.0, 3.0)
    ld = c[1].number_input("Laundry loads per week (home)", 0.0, 30.0, 4.0)
    dw = c[2].number_input("Tap open for dishes (min per day, home)", 0.0, 120.0, 15.0)
    items = {"Bathing": shw * 9, "Toilet": fl * 6, "Brushing": br * 6,
             "Laundry": ld * 60 / 7 / n, "Dishes": dw * 6 / n, "Drinking and cooking": 8}
    total = sum(items.values())
    st.metric("Water use per person per day", f"{total:.0f} L",
              "above 135 L benchmark" if total > 135 else "within 135 L benchmark", delta_color="inverse")
    st.bar_chart(pd.Series(items))
    saving = max(0, shw * 9 - 15) + max(0, br * 6 - 1)
    st.markdown(f"""
- Bucket bath (about 15 L) and a mug while brushing can save **{saving:.0f} L** per person per day, **{saving * n * 365 / 1000:.0f} kilolitres** a year for your home.
- Fix leaking taps: one drip per second wastes around 10,000 L a year.
- Reuse RO reject water for mopping and plants.
- Run washing machines with a full load.
""")

# ----------------------------------------------------------------------------
# 7. Purifier advisor
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Purifier advisor")
    st.caption("Uses the values from the Predict tab.")
    src = st.selectbox("Water source", ["Municipal / tap", "Borewell / handpump", "Tanker / storage tank",
                                        "River / pond / open well"])
    g = dict(zip(KEYS, vals))
    germ = not src.startswith("Municipal") or g["tur"] > 1 or g["toc"] > 4
    ro = g["tds"] > 500 or g["hard"] > 300 or g["sul"] > 250 or g["cond"] > 800
    rows = []
    if g["tur"] > 1:
        rows.append(("Sediment / cloth pre-filter", f"Turbidity is {g['tur']} NTU, remove particles first.", "every 3 to 6 months"))
    if g["chl"] > 4 or g["toc"] > 4:
        rows.append(("Activated carbon filter", "Chloramines or organic carbon are high.", "every 6 months"))
    if ro:
        rows.append(("RO purifier", "TDS, hardness, sulfate or conductivity is above the limit. Reuse reject water.", "membrane every 2 to 3 years"))
    if germ:
        rows.append(("UV or UF stage", "Source or turbidity suggests germs. RO alone does not guarantee disinfection.", "UV lamp every 12 months"))
    if g["ph"] < 6.5 or g["ph"] > 8.5:
        rows.append(("pH-balancing cartridge", f"pH is {g['ph']}, outside 6.5 to 8.5.", "about 6 to 12 months"))
    if not rows:
        rows.append(("Basic UF or carbon candle", "Values are within limits, light filtering is enough.", "every 6 months"))
    rows.append(("Storage tank hygiene", "Keep tanks covered and clean.", "scrub every 6 months"))
    st.dataframe(pd.DataFrame(rows, columns=["Stage", "Why", "Maintenance"]), hide_index=True)
    st.caption("Tip: if TDS is under 500 mg/L you often do not need RO. It wastes water.")
    with st.expander("Emergency disinfection (no purifier)"):
        st.markdown("""
- **Boil:** rolling boil for 1 minute, cool and keep covered.
- **Chlorine:** about 2 drops of plain 5% bleach per litre (or a chlorine tablet as per label), mix, wait 30 minutes.
- **Sunlight (SODIS):** clear PET bottle of clear water, 6 hours in full sun.
- Cloudy water? Strain through clean cloth first.
""")

# ----------------------------------------------------------------------------
# 8. Rainwater harvesting
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Rainwater harvesting calculator")
    st.caption("Use your city's annual rainfall.")
    c = st.columns(4)
    area = c[0].number_input("Roof area (m²)", 0.0, 100000.0, 100.0)
    rain = c[1].number_input("Annual rainfall (mm)", 0.0, 6000.0, 800.0)
    roof = c[2].selectbox("Roof type", {"Concrete": .85, "Clay tiles": .75, "Metal sheet": .9}.keys())
    ppl = int(c[3].number_input("People", 1, 100, 4, key="rw_ppl"))
    coef = {"Concrete": .85, "Clay tiles": .75, "Metal sheet": .9}[roof]
    litres = area * rain * coef * .9               # 10% first-flush loss
    need = 45 * ppl                                # non-potable use per day
    k1, k2, k3 = st.columns(3)
    k1.metric("Usable harvest", f"{litres:,.0f} L/year")
    k2.metric("Days of non-potable supply", f"{litres / need:.0f}")
    k3.metric("Tank for 15-day dry spell", f"{need * 15:,.0f} L")
    st.markdown("""
- Fit a mesh and a first-flush diverter on the downpipe.
- Send extra water to a recharge pit or borewell to lift groundwater.
- Keep the tank covered. Treat the water before drinking.
""")

# ----------------------------------------------------------------------------
# 9. Sanitation check (SDG 6.2)
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Home sanitation and hygiene check")
    HC = [("Safe toilet (sewer or septic) at home", "Use a safe toilet. Open defecation pollutes water sources."),
          ("Hands washed with soap after toilet and before meals", "Wash hands with soap. It greatly reduces diarrhoea."),
          ("Water tank cleaned in the last 6 months", "Scrub and rinse the tank every 6 months."),
          ("Drinking water stored in a covered container", "Use a covered container with a tap. Do not dip hands or cups."),
          ("Drinking water is filtered, boiled or UV treated", "Treat drinking water before use."),
          ("No stagnant water around the house", "Empty coolers, pots and tyres weekly to stop mosquitoes."),
          ("Taps and pipes have no leaks", "Fix leaks. Dirty water can enter leaking pipes."),
          ("Wet and dry waste kept separate", "Separate waste and compost wet waste. Keep garbage away from water."),
          ("Septic tank or soak pit is about 10 m or more from the borewell", "Keep septic tanks away from drinking-water sources."),
          ("Sanitary waste is wrapped and binned, never flushed", "Wrap sanitary waste and bin it. Never flush it.")]
    ticks = [st.checkbox(h[0], key=f"hc{i}") for i, h in enumerate(HC)]
    score = round(sum(ticks) / len(HC) * 100)
    st.metric("Hygiene score", f"{score}%", "Excellent" if score >= 80 else "Getting there" if score >= 50 else "Needs attention")
    st.progress(score / 100)
    missing = [h[1] for h, t in zip(HC, ticks) if not t]
    if missing:
        st.markdown("**Fix these first**\n" + "\n".join(f"- {m}" for m in missing))
    else:
        st.success("Great hygiene! Share these habits with neighbours.")
    st.caption("Loose motions or vomiting? Give ORS and see a doctor if it is severe or lasts more than a day.")

# ----------------------------------------------------------------------------
# 10. About the AI
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("How the AI works")
    if bundle["kind"] == "synthetic":
        st.markdown(f"""
1. **Data:** {bundle['n']} synthetic water samples made from BIS (IS 10500) and WHO limits, with 4% label noise.
2. **Features:** for each of 8 parameters, how far the value is outside its safe range.
3. **Model:** Logistic Regression (scikit-learn), 80% train / 20% test.
4. **Output:** probability of potable water, WQI score and treatment advice.
""")
    else:
        st.markdown(f"Model: **Random Forest** trained on your CSV ({bundle['n']} rows), 80% train / 20% test.")
    a, b = st.columns(2)
    a.metric("Training accuracy", f"{bundle['train'] * 100:.1f}%")
    b.metric("Test accuracy", f"{bundle['test'] * 100:.1f}%")
    st.write("**Confusion matrix (test set)**")
    st.dataframe(pd.DataFrame(bundle["cm"], index=["Actual: not potable", "Actual: potable"],
                              columns=["Pred: not potable", "Pred: potable"]))
    st.write("**Feature importance**")
    st.bar_chart(pd.Series(bundle["imp"], index=[p["n"] for p in PARAMS]))
    st.markdown("""
**SDG 6 targets:** 6.1 safe drinking water, 6.2 sanitation and hygiene, 6.3 water quality, 6.4 water-use efficiency.

Note: the Kaggle dataset is noisy, so expect only about 60 to 70% accuracy on it. The synthetic model scores higher
because its labels follow clear rules.
""")
