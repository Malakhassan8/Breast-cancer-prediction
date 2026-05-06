import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="CancerLens — Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0f1117;
    --card: #1a1d27;
    --accent: #e05c7a;
    --benign: #4caf8a;
    --malignant: #e05c7a;
    --text: #e8eaf0;
    --muted: #7a7f9a;
    --border: #2a2d3e;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.hero {
    background: linear-gradient(135deg, #1a1d27 0%, #21253a 50%, #1a1d27 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🔬';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.08;
}
.hero h1 {
    font-size: 2.6rem;
    margin: 0 0 0.3rem 0;
    background: linear-gradient(135deg, #e8eaf0, #e05c7a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0; }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val {
    font-size: 1.8rem;
    font-weight: 600;
    font-family: 'DM Serif Display', serif;
}
.metric-card .lbl { color: var(--muted); font-size: 0.8rem; margin-top: 0.2rem; }

.result-benign {
    background: linear-gradient(135deg, #0d2b1f, #1a3d2b);
    border: 2px solid var(--benign);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-malignant {
    background: linear-gradient(135deg, #2b0d18, #3d1a23);
    border: 2px solid var(--malignant);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title { font-family: 'DM Serif Display', serif; font-size: 2rem; margin: 0.5rem 0; }
.result-sub { color: var(--muted); font-size: 0.9rem; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.stSlider > div > div > div { background: var(--accent) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #c0425f) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: var(--text) !important; font-size: 0.9rem !important; }

.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature info ─────────────────────────────────────────────────
FEATURES = [
    ("Clump Thickness",           "Thickness of cell clumps. Higher = more likely malignant."),
    ("Uniformity of Cell Size",   "How consistent cell sizes are. Irregular sizes indicate cancer."),
    ("Uniformity of Cell Shape",  "How consistent cell shapes are. Irregular shapes = warning sign."),
    ("Marginal Adhesion",         "How well cells stick together. Cancer cells often separate."),
    ("Single Epithelial Cell Size","Size of individual epithelial cells. Enlargement is a red flag."),
    ("Bare Nuclei",               "Nuclei not surrounded by cytoplasm. Common in malignant cells."),
    ("Bland Chromatin",           "Texture of the nucleus. Coarse texture = more likely malignant."),
    ("Normal Nucleoli",           "Presence of nucleoli. Prominent nucleoli indicate malignancy."),
    ("Mitoses",                   "Rate of cell division. High mitosis = rapid tumor growth."),
]
FEATURE_NAMES = [f[0] for f in FEATURES]

# ── Load models from .sav files ──────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    tree   = pickle.load(open(os.path.join(BASE_DIR, "decision_tree.sav"),  "rb"))
    NB     = pickle.load(open(os.path.join(BASE_DIR, "naive_bayes.sav"),    "rb"))
    nn     = pickle.load(open(os.path.join(BASE_DIR, "neural_network.sav"), "rb"))
    model4 = pickle.load(open(os.path.join(BASE_DIR, "rule_induction.sav"), "rb"))
    scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.sav"),         "rb"))

    models = {
        "Decision Tree":   tree,
        "Naive Bayes":     NB,
        "Neural Network":  nn,
        "Rule Induction":  model4,
    }

    # Hardcoded metrics from your notebook results
    metrics = {
        "Decision Tree":  {"Accuracy": 92.70, "Precision": 91.67, "Recall": 88.00, "F1-Score": 89.80},
        "Naive Bayes":    {"Accuracy": 94.16, "Precision": 92.00, "Recall": 92.00, "F1-Score": 92.00},
        "Neural Network": {"Accuracy": 97.08, "Precision":100.00, "Recall": 92.00, "F1-Score": 95.83},
        "Rule Induction": {"Accuracy": 96.35, "Precision": 97.87, "Recall": 92.00, "F1-Score": 94.85},
    }

    # Extract rules from RuleFit
    rules_data = []
    try:
        for rule_obj in model4.rules_:
            rule_text = rule_obj.rule
            for i, name in enumerate(FEATURE_NAMES):
                rule_text = rule_text.replace(f"X{i}", name)
            coef = rule_obj.args[0] if hasattr(rule_obj, "args") and rule_obj.args else 0
            rules_data.append({"rule": rule_text, "coef": float(coef), "support": float(rule_obj.support)})
        rules_df = pd.DataFrame(rules_data)
        rules_df = rules_df[rules_df["coef"] != 0].sort_values("support", ascending=False)
    except:
        rules_df = pd.DataFrame()

    return models, scaler, metrics, rules_df

# ── PDF text extraction ──────────────────────────────────────────
def extract_values_from_text(text):
    """Try to parse feature values from free text / report."""
    vals = {}
    patterns = {
        "Clump Thickness":            r"clump[\s_]thickness[\s:=]+(\d+(?:\.\d+)?)",
        "Uniformity of Cell Size":    r"cell[\s_]size[\s:=]+(\d+(?:\.\d+)?)",
        "Uniformity of Cell Shape":   r"cell[\s_]shape[\s:=]+(\d+(?:\.\d+)?)",
        "Marginal Adhesion":          r"marginal[\s_]adhesion[\s:=]+(\d+(?:\.\d+)?)",
        "Single Epithelial Cell Size":r"epithelial[\s:=]+(\d+(?:\.\d+)?)",
        "Bare Nuclei":                r"bare[\s_]nuclei[\s:=]+(\d+(?:\.\d+)?)",
        "Bland Chromatin":            r"bland[\s_]chromatin[\s:=]+(\d+(?:\.\d+)?)",
        "Normal Nucleoli":            r"normal[\s_]nucleoli[\s:=]+(\d+(?:\.\d+)?)",
        "Mitoses":                    r"mitoses[\s:=]+(\d+(?:\.\d+)?)",
    }
    text_lower = text.lower()
    for feat, pat in patterns.items():
        m = re.search(pat, text_lower)
        if m:
            v = float(m.group(1))
            vals[feat] = max(1, min(10, int(round(v))))
    return vals

# ── Load models ──────────────────────────────────────────────────
models, scaler, metrics, rules_df = load_models()

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>CancerLens</h1>
  <p>Breast Cancer Classification · Wisconsin Dataset · 4 ML Algorithms</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    algo = st.selectbox("Algorithm", list(models.keys()), index=2)
    st.markdown("---")
    st.markdown("### 📋 About")
    st.caption("This tool uses machine learning to assist in classifying breast tumors as benign or malignant based on 9 clinical features from the Wisconsin Breast Cancer Dataset.")
    st.caption("⚠️ For educational purposes only. Not a medical diagnosis tool.")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🩺 Classify", "📊 Model Comparison"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY
# ════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Enter Patient Data</div>', unsafe_allow_html=True)

        # PDF / report upload
        uploaded = st.file_uploader("📄 Upload lab report (PDF or TXT) — auto-fills sliders", type=["pdf","txt"])
        auto_vals = {}
        if uploaded:
            try:
                if uploaded.type == "application/pdf":
                    import pdfplumber
                    with pdfplumber.open(uploaded) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                else:
                    text = uploaded.read().decode("utf-8", errors="ignore")
                auto_vals = extract_values_from_text(text)
                if auto_vals:
                    st.success(f"✅ Auto-detected {len(auto_vals)}/{len(FEATURES)} features from report")
                else:
                    st.warning("Could not parse feature values — please enter manually below.")
            except Exception as e:
                st.warning(f"Could not read file: {e}. Please enter values manually.")

        st.markdown("**Or enter values manually** (1 = lowest, 10 = highest risk)")
        input_vals = {}
        for feat, desc in FEATURES:
            default = auto_vals.get(feat, 1)
            input_vals[feat] = st.slider(feat, 1, 10, default, help=desc)

        predict_btn = st.button("🔍 Predict", use_container_width=True)
        all_models_btn = st.button("⚡ Run All Models", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

        if predict_btn:
            arr = np.array([[input_vals[f] for f in FEATURE_NAMES]])
            arr_scaled = scaler.transform(arr)
            m = models[algo]
            pred = int(m.predict(arr_scaled)[0])

            # Probability if available
            prob_str = ""
            if hasattr(m, "predict_proba"):
                try:
                    prob = m.predict_proba(arr_scaled)[0]
                    conf = prob[pred] * 100
                    prob_str = f"<p class='result-sub'>Confidence: <b>{conf:.1f}%</b></p>"
                except: pass

            if pred == 0:
                st.markdown(f"""
                <div class="result-benign">
                  <div style="font-size:3rem">✅</div>
                  <div class="result-title" style="color:#4caf8a">Benign</div>
                  <p class="result-sub">Tumor appears non-cancerous</p>
                  {prob_str}
                  <p class="result-sub" style="margin-top:1rem;font-size:0.75rem">Algorithm: {algo}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-malignant">
                  <div style="font-size:3rem">🔴</div>
                  <div class="result-title" style="color:#e05c7a">Malignant</div>
                  <p class="result-sub">Tumor shows malignant characteristics</p>
                  {prob_str}
                  <p class="result-sub" style="margin-top:1rem;font-size:0.75rem">Algorithm: {algo}</p>
                </div>""", unsafe_allow_html=True)

            # Feature risk bars
            st.markdown('<div class="section-header" style="margin-top:1.5rem">Feature Risk Profile</div>', unsafe_allow_html=True)
            for feat in FEATURE_NAMES:
                v = input_vals[feat]
                pct = (v - 1) / 9
                color = "#4caf8a" if pct < 0.4 else "#e0a05c" if pct < 0.7 else "#e05c7a"
                st.markdown(f"""
                <div style="margin-bottom:0.5rem">
                  <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
                    <span>{feat}</span><span style="color:{color};font-weight:600">{v}/10</span>
                  </div>
                  <div style="background:#2a2d3e;border-radius:4px;height:6px">
                    <div style="width:{pct*100:.0f}%;background:{color};height:6px;border-radius:4px;transition:width 0.3s"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Rule Induction section
            if algo == "Rule Induction" and not rules_df.empty:
                st.markdown('<div class="section-header">Fired Rules</div>', unsafe_allow_html=True)
                st.caption("Rules from the model that match your input values:")
                shown = 0
                for _, row in rules_df.head(15).iterrows():
                    direction = "benign" if row["coef"] < 0 else "malignant"
                    label = "🔵 Benign" if direction == "benign" else "🔴 Malignant"
                    st.markdown(f"""
                    <div class="rule-card rule-{direction}">
                      <b>{label}</b> &nbsp;·&nbsp; Support: {row['support']*100:.1f}% &nbsp;·&nbsp; Coef: {row['coef']:.3f}<br>
                      <span style="color:#7a7f9a">{row['rule']}</span>
                    </div>""", unsafe_allow_html=True)
                    shown += 1
                    if shown >= 8: break
        elif all_models_btn:
            arr = np.array([[input_vals[f] for f in FEATURE_NAMES]])
            arr_scaled = scaler.transform(arr)

            st.markdown('<div class="section-header">All Models — Side by Side</div>', unsafe_allow_html=True)

            all_preds = {}
            for name, m in models.items():
                p = int(m.predict(arr_scaled)[0])
                conf = None
                if hasattr(m, "predict_proba"):
                    try:
                        prob = m.predict_proba(arr_scaled)[0]
                        conf = prob[p] * 100
                    except: pass
                all_preds[name] = {"pred": p, "conf": conf}

            # Verdict banner
            votes_malignant = sum(1 for v in all_preds.values() if v["pred"] == 1)
            votes_benign    = len(all_preds) - votes_malignant
            majority = "Malignant" if votes_malignant > votes_benign else "Benign"
            maj_color = "#e05c7a" if majority == "Malignant" else "#4caf8a"
            maj_icon  = "🔴" if majority == "Malignant" else "✅"
            agree = votes_malignant == len(all_preds) or votes_benign == len(all_preds)
            agreement_text = "All models agree" if agree else f"{max(votes_malignant, votes_benign)}/{len(all_preds)} models agree"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a1d27,#21253a);
                        border:2px solid {maj_color};border-radius:14px;
                        padding:1.2rem 1.5rem;text-align:center;margin-bottom:1.2rem">
              <div style="font-size:2rem">{maj_icon}</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:{maj_color}">{majority}</div>
              <div style="color:#7a7f9a;font-size:0.85rem;margin-top:0.3rem">{agreement_text} · {votes_benign} Benign · {votes_malignant} Malignant</div>
            </div>""", unsafe_allow_html=True)

            # Individual model cards
            for name, result in all_preds.items():
                p     = result["pred"]
                conf  = result["conf"]
                label = "✅ Benign" if p == 0 else "🔴 Malignant"
                bcolor = "#4caf8a" if p == 0 else "#e05c7a"
                bg     = "#0d2b1f" if p == 0 else "#2b0d18"
                conf_str = f" · {conf:.1f}% confidence" if conf else ""
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bcolor}44;
                            border-left:4px solid {bcolor};border-radius:10px;
                            padding:0.9rem 1.2rem;margin-bottom:0.6rem;
                            display:flex;justify-content:space-between;align-items:center">
                  <div>
                    <span style="font-weight:600;color:#e8eaf0">{name}</span>
                    <span style="color:#7a7f9a;font-size:0.8rem">{conf_str}</span>
                  </div>
                  <div style="color:{bcolor};font-weight:700;font-size:1rem">{label}</div>
                </div>""", unsafe_allow_html=True)

            # Disagreement note
            if not agree:
                st.markdown("""
                <div style="background:#2a2515;border:1px solid #e0c05c44;border-radius:10px;
                            padding:0.9rem 1.2rem;margin-top:0.5rem;color:#e0c05c;font-size:0.85rem">
                  ⚠️ <b>Models disagree</b> — this case may be ambiguous. 
                  Neural Network and Rule Induction are generally more reliable on this dataset.
                </div>""", unsafe_allow_html=True)

            # Feature risk bars
            st.markdown('<div class="section-header" style="margin-top:1.2rem">Feature Risk Profile</div>', unsafe_allow_html=True)
            for feat in FEATURE_NAMES:
                v = input_vals[feat]
                pct = (v - 1) / 9
                color = "#4caf8a" if pct < 0.4 else "#e0a05c" if pct < 0.7 else "#e05c7a"
                st.markdown(f"""
                <div style="margin-bottom:0.5rem">
                  <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
                    <span>{feat}</span><span style="color:{color};font-weight:600">{v}/10</span>
                  </div>
                  <div style="background:#2a2d3e;border-radius:4px;height:6px">
                    <div style="width:{pct*100:.0f}%;background:{color};height:6px;border-radius:4px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#1a1d27;border:1px dashed #2a2d3e;border-radius:16px;
                        padding:3rem;text-align:center;color:#7a7f9a;margin-top:2rem">
              <div style="font-size:3rem;margin-bottom:1rem">🩺</div>
              <p>Adjust the sliders and click <b>Predict</b> for one model<br>
              or <b>Run All Models</b> to compare all 4 at once</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Algorithm Performance Comparison</div>', unsafe_allow_html=True)

    df_m = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Algorithm"})

    # Metric cards
    cols = st.columns(4)
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for i, lbl in enumerate(metric_labels):
        best_val  = df_m[lbl].max()
        best_algo = df_m.loc[df_m[lbl].idxmax(), "Algorithm"]
        cols[i].markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:#e05c7a">{best_val}%</div>
          <div class="lbl">Best {lbl}</div>
          <div style="color:#7a7f9a;font-size:0.75rem;margin-top:0.3rem">{best_algo}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table
    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    styled = df_m.set_index("Algorithm").style\
        .background_gradient(cmap="RdYlGn", vmin=88, vmax=100)\
        .format("{:.2f}%")
    st.dataframe(styled, use_container_width=True)

    # Insights
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    insights = [
        ("🏆 Best Overall", "Neural Network achieves the highest accuracy (97.08%) with perfect Precision (100%) — zero false positives."),
        ("📖 Most Interpretable", "Rule Induction (96.35%) produces human-readable IF-THEN rules, making it ideal for clinical transparency."),
        ("⚠️ Recall Parity", "All models achieve 92% Recall — meaning 8% of malignant cases are missed across the board. A key area for improvement."),
        ("🌳 Decision Tree", "Lowest accuracy (92.70%) but most visually explainable through its tree structure. Prone to overfitting on small datasets."),
        ("🏥 Clinical Recommendation", "Deploy Neural Network for accuracy + Rule Induction as explainability layer for doctors to validate predictions."),
    ]
    for icon_title, body in insights:
        st.markdown(f"""
        <div style="background:#1a1d27;border:1px solid #2a2d3e;border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:0.6rem">
          <b style="color:#e05c7a">{icon_title}</b><br>
          <span style="color:#b0b3c8;font-size:0.9rem">{body}</span>
        </div>""", unsafe_allow_html=True)
