"""
Thyroid Cancer Risk Prediction — Streamlit Demo App  (UI v3 — Fixed)
────────────────────────────────────────────────────────────────────────
Run:   streamlit run app.py
Needs: thyroid_model.pkl | feature_columns.pkl | best_model_name.pkl

ML LOGIC IS UNCHANGED. Input widgets now use st.container(border=True)
instead of raw HTML divs so they actually render and are interactive.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thyroid Cancer Risk Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp { background: #f0f4f8; }

    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1280px;
    }

    /* ── Header banner ── */
    .app-header {
        background: linear-gradient(135deg, #0f2942 0%, #1a4a7a 60%, #1e6091 100%);
        border-radius: 16px;
        padding: 2rem 2.6rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 24px rgba(15,41,66,0.18);
    }
    .header-left h1 { color:#fff; font-size:1.9rem; font-weight:700; margin:0 0 0.3rem 0; letter-spacing:-0.4px; }
    .header-left p  { color:#a8c8e8; font-size:0.9rem; margin:0; }
    .header-badges  { display:flex; gap:0.9rem; }
    .header-badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 10px;
        padding: 0.65rem 1.1rem;
        text-align: center;
        min-width: 90px;
    }
    .badge-label { color:#a8c8e8; font-size:0.68rem; text-transform:uppercase; letter-spacing:1.1px; display:block; margin-bottom:0.2rem; }
    .badge-value { color:#fff; font-size:0.9rem; font-weight:600; font-family:'DM Mono',monospace; }

    /* ── Section heading label ── */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #1e6091;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }

    /* ── st.container(border=True) card look ── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border-radius: 14px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
        padding: 0.3rem 0.5rem !important;
    }

    /* ── Input labels ── */
    label[data-testid="stWidgetLabel"] p {s
        font-weight: 600;
        font-size: 0.86rem;
        color: #374151;
    }

    /* ── Predict button ── */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #0f2942, #1e6091) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 18px rgba(15,41,66,0.28) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(15,41,66,0.38) !important;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1.5rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #6b7280 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f2942 !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* ── Risk alert ── */
    .risk-alert {
        border-radius: 12px;
        padding: 1.3rem 1.7rem;
        margin: 1.2rem 0;
        border-left: 5px solid;
    }
    .risk-title { font-size:1.05rem; font-weight:700; margin-bottom:0.4rem; }
    .risk-desc  { font-size:0.9rem; line-height:1.6; opacity:0.92; }

    /* ── Results wrapper ── */
    .results-wrapper {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        margin-top: 1.6rem;
    }
    .results-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f2942;
        margin-bottom: 1.4rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #f0f4f8;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #534444; }
    [data-testid="stSidebar"] * { color: #000000 !important; background: #ffffff !important;  }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong { color:#000000 !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ── Disclaimer ── */
    .disclaimer-pill {
        background: #fff8e1;
        border: 1px solid #f9c74f;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;s
        font-size: 0.82rem;
        color: #7c6000;
        margin-top: 1.4rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


# ── Artefact loading (UNCHANGED) ──────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    errors = []
    for fname in ("thyroid_model.pkl", "feature_columns.pkl", "best_model_name.pkl"):
        if not os.path.exists(fname):
            errors.append(fname)
    if errors:
        raise FileNotFoundError(errors)
    model      = joblib.load("thyroid_model.pkl")
    feat_cols  = joblib.load("feature_columns.pkl")
    model_name = joblib.load("best_model_name.pkl")
    return model, feat_cols, model_name


try:
    model, FEATURE_COLUMNS, model_name = load_artefacts()
    artefacts_ok = True
except FileNotFoundError as e:
    artefacts_ok = False
    missing_files = e.args[0] if isinstance(e.args[0], list) else [str(e)]
    st.error(
        "**Model artefacts not found.** Run the training notebook first.\n\n"
        + "\n".join(f"- `{f}`" for f in missing_files)
        + "\n\n1. Open `thyroid_cancer_ML_v2.ipynb`\n"
        "2. Run all cells\n"
        "3. Copy the three `.pkl` files next to `app.py`\n"
        "4. Restart Streamlit"
    )
    st.info("`pip install streamlit scikit-learn pandas numpy joblib lightgbm`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-header">
  <div class="header-left">
    <h1>🔬 Thyroid Cancer Risk Predictor</h1>
    <p>AI-assisted screening tool for Papillary Thyroid Carcinoma &nbsp;·&nbsp; Research &amp; Educational Use Only</p>
  </div>
  <div class="header-badges">
    <div class="header-badge">
      <span class="badge-label">Model</span>
      <span class="badge-value">{model_name}</span>
    </div>
    <div class="header-badge">
      <span class="badge-label">Optimised For</span>
      <span class="badge-value">Recall</span>
    </div>
    <div class="header-badge">
      <span class="badge-label">Features</span>
      <span class="badge-value">{len(FEATURE_COLUMNS)}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Settings")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10, max_value=0.90, value=0.50, step=0.01,
    help="Lower = higher recall (fewer missed malignancies). Default 0.50.",
)
st.sidebar.caption(f"Predicts **Malignant** when P(malignant) ≥ **{threshold:.2f}**")
st.sidebar.markdown("---")
st.sidebar.markdown("## ℹ️ About")
st.sidebar.markdown(f"""
**Model:** `{model_name}`  
**Optimised for:** Recall (sensitivity)  
**Purpose:** PTC risk screening

**Why Recall?**  
Missing a malignancy is far more dangerous than a false positive — the model minimises missed cases.

**Adjusting the threshold:**  
Lower → more sensitive, more false positives.  
Higher → more specific, risks missing cases.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Input features:**")
for f in FEATURE_COLUMNS:
    st.sidebar.markdown(f"  `{f}`")
st.sidebar.markdown("---\n⚠️ *For research & educational use only.*")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: Demographics + Clinical Labs ───────────────────────────────────────
with left_col:

    # Section 1 — Demographics
    st.markdown('<p class="section-label">👤 Patient Demographics</p>', unsafe_allow_html=True)
    with st.container(border=True):
        d1, d2 = st.columns(2, gap="medium")
        with d1:
            age            = st.slider("Age (years)", min_value=10, max_value=100, value=40, step=1)
            family_history = st.selectbox("Family History of Thyroid Cancer", ["No", "Yes"])
        with d2:
            gender    = st.selectbox("Biological Sex", ["Male", "Female"])
            radiation = st.selectbox("Prior Head / Neck Radiation", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2 — Clinical & Lab Values
    st.markdown('<p class="section-label">🧪 Clinical & Laboratory Values</p>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            tsh = st.number_input(
                "TSH Level (µIU/mL)",
                min_value=0.0, max_value=20.0, value=2.5, step=0.1,
                help="Normal range: 0.4 – 4.0 µIU/mL",
            )
            t3 = st.number_input(
                "T3 Level (ng/dL)",
                min_value=0.0, max_value=5.0, value=1.5, step=0.05,
                help="Normal range: 0.8 – 2.0 ng/dL",
            )
        with c2:
            t4 = st.number_input(
                "T4 Level (µg/dL)",
                min_value=0.0, max_value=20.0, value=8.0, step=0.1,
                help="Normal range: 5.0 – 12.0 µg/dL",
            )
            nodule_size = st.number_input(
                "Dominant Nodule Size (cm)",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="Diameter of dominant thyroid nodule on ultrasound",
            )


# ── RIGHT: Lifestyle + Reference panels ──────────────────────────────────────
with right_col:

    # Section 3 — Lifestyle & Metabolic
    st.markdown('<p class="section-label">⚕️ Lifestyle & Metabolic Factors</p>', unsafe_allow_html=True)
    with st.container(border=True):
        lm1, lm2 = st.columns(2, gap="medium")
        with lm1:
            smoking = st.selectbox("Smoking Status", ["No", "Yes"], help="Current or former smoker")
            obesity = st.selectbox("Obesity (BMI > 30)", ["No", "Yes"])
        with lm2:
            diabetes = st.selectbox("Diabetes (Type I or II)", ["No", "Yes"])
            iodine   = st.selectbox("Iodine Deficiency", ["No", "Yes"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Reference ranges table
    st.markdown('<p class="section-label">📋 Reference Ranges</p>', unsafe_allow_html=True)
    with st.container(border=True):
        ref_df = pd.DataFrame({
            "Marker":       ["TSH",       "T3",       "T4",        "Nodule Size"],
            "Normal Range": ["0.4 – 4.0", "0.8 – 2.0","5.0 – 12.0","< 1.0 cm"],
            "Unit":         ["µIU/mL",    "ng/dL",    "µg/dL",     "cm"],
        })
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk threshold guide
    st.markdown('<p class="section-label">🎯 Risk Threshold Guide</p>', unsafe_allow_html=True)
    with st.container(border=True):
        th1, th2, th3 = st.columns(3)
        with th1:
            st.markdown(
                "<div style='background:#d4edda;color:#155724;border-radius:8px;"
                "padding:0.65rem 0.4rem;text-align:center;font-weight:700;font-size:0.82rem;'>"
                "🟢 LOW<br><span style='font-size:0.73rem;font-weight:400;'>&lt; 30%</span></div>",
                unsafe_allow_html=True,
            )
        with th2:
            st.markdown(
                "<div style='background:#fff3cd;color:#856404;border-radius:8px;"
                "padding:0.65rem 0.4rem;text-align:center;font-weight:700;font-size:0.82rem;'>"
                "🟡 MOD<br><span style='font-size:0.73rem;font-weight:400;'>30–70%</span></div>",
                unsafe_allow_html=True,
            )
        with th3:
            st.markdown(
                "<div style='background:#f8d7da;color:#721c24;border-radius:8px;"
                "padding:0.65rem 0.4rem;text-align:center;font-weight:700;font-size:0.82rem;'>"
                "🔴 HIGH<br><span style='font-size:0.73rem;font-weight:400;'>&gt; 70%</span></div>",
                unsafe_allow_html=True,
            )
        st.caption("Monitor · FNA Biopsy · Urgent Referral")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
_, btn_mid, _ = st.columns([1, 1.2, 1])
with btn_mid:
    predict_btn = st.button("🔍  Run Risk Prediction", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC (UNCHANGED from original v2)
# ══════════════════════════════════════════════════════════════════════════════
if predict_btn:

    input_data = {
        "Age":                int(age),
        "Gender":             0 if gender == "Male" else 1,
        "Family_History":     1 if family_history == "Yes" else 0,
        "Radiation_Exposure": 1 if radiation == "Yes" else 0,
        "Iodine_Deficiency":  1 if iodine == "Yes" else 0,
        "Smoking":            1 if smoking == "Yes" else 0,
        "Obesity":            1 if obesity == "Yes" else 0,
        "Diabetes":           1 if diabetes == "Yes" else 0,
        "TSH_Level":          float(tsh),
        "T3_Level":           float(t3),
        "T4_Level":           float(t4),
        "Nodule_Size":        float(nodule_size),
    }

    # Align to training feature order — UNCHANGED
    input_df = pd.DataFrame([input_data]).reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Predict — UNCHANGED
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
        prediction  = int(probability >= threshold)
    else:
        prediction  = int(model.predict(input_df)[0])
        probability = float(prediction)

    # Risk category — UNCHANGED thresholds
    if probability < 0.30:
        risk_label  = "🟢 LOW RISK"
        risk_color  = "#155724"
        risk_bg     = "#d4edda"
        risk_border = "#82c99a"
        risk_msg    = (
            "Routine monitoring recommended. "
            "Repeat ultrasound in 6–12 months. No immediate intervention indicated."
        )
    elif probability < 0.70:
        risk_label  = "🟡 MODERATE RISK"
        risk_color  = "#856404"
        risk_bg     = "#fff3cd"
        risk_border = "#f9c74f"
        risk_msg    = (
            "Further diagnostic workup recommended "
            "(e.g., fine-needle aspiration cytology). "
            "Discuss with an endocrinologist."
        )
    else:
        risk_label  = "🔴 HIGH RISK"
        risk_color  = "#721c24"
        risk_bg     = "#f8d7da"
        risk_border = "#f1aeb5"
        risk_msg    = (
            "Biopsy strongly recommended. "
            "Urgent specialist referral advised. "
            "Consider surgical consultation."
        )

    # ── Results display ───────────────────────────────────────────────────────
    st.markdown('<div class="results-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="results-header">📊 Prediction Results</div>', unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3, gap="large")
    with rc1:
        st.metric("Malignancy Probability", f"{probability:.1%}")
    with rc2:
        st.metric("Classification", "Malignant" if prediction == 1 else "Benign")
    with rc3:
        st.metric("Threshold Applied", f"{threshold:.2f}")

    st.markdown(f"""
<div class="risk-alert" style="background:{risk_bg}; border-left-color:{risk_border}; color:{risk_color};">
  <div class="risk-title">{risk_label}</div>
  <div class="risk-desc">{risk_msg}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-size:0.8rem;font-weight:700;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:0.8px;margin:1rem 0 0.3rem 0;'>"
        f"Malignancy Probability — {probability:.1%}</p>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Model confidence: **{probability:.1%}** · "
        f"Decision threshold: **{threshold:.2f}** · "
        f"Optimised for: **Recall (Sensitivity)**"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Input summary ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🗒️ View Full Input Summary", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("**Demographics & Lab Values**")
            demo_lab = {k: input_data[k] for k in
                        ["Age","Gender","Family_History","Radiation_Exposure",
                         "TSH_Level","T3_Level","T4_Level","Nodule_Size"]
                        if k in input_data}
            st.dataframe(
                pd.DataFrame({"Feature": list(demo_lab.keys()),
                              "Value":   [str(v) for v in demo_lab.values()]}),
                use_container_width=True, hide_index=True,
            )
        with s2:
            st.markdown("**Lifestyle & Metabolic**")
            life = {k: input_data[k] for k in
                    ["Smoking","Obesity","Diabetes","Iodine_Deficiency"]
                    if k in input_data}
            st.dataframe(
                pd.DataFrame({"Feature": list(life.keys()),
                              "Value":   [str(v) for v in life.values()]}),
                use_container_width=True, hide_index=True,
            )

    st.markdown("""
<div class="disclaimer-pill">
  ⚠️ <strong>Medical Disclaimer:</strong> This tool is strictly for research and educational demonstration.
  It must not replace professional medical diagnosis, clinical advice, or treatment decisions.
  Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)
