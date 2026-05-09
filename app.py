"""
app.py  —  Multiple Disease Prediction System
Run with:  streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statistics import mode

# ── Inline predictor (no import needed) ───────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

def _load(filename):
    with open(os.path.join(MODELS_DIR, filename), "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner="Loading ML models …")
def load_artefacts():
    svm_model = _load("svm_model.pkl")
    nb_model  = _load("naivebayes_model.pkl")
    rf_model  = _load("randomforest_model.pkl")
    meta      = _load("meta.pkl")
    return svm_model, nb_model, rf_model, meta

def predict(symptoms_list, svm_model, nb_model, rf_model, meta):
    symptom_index       = meta["symptom_index"]
    predictions_classes = meta["predictions_classes"]

    input_data = np.zeros(len(symptom_index))
    matched, unmatched = [], []

    for s in symptoms_list:
        s = s.strip()
        if s in symptom_index:
            input_data[symptom_index[s]] = 1
            matched.append(s)
        else:
            unmatched.append(s)

    X = input_data.reshape(1, -1)

    svm_pred = predictions_classes[svm_model.predict(X)[0]]
    nb_pred  = predictions_classes[nb_model.predict(X)[0]]
    rf_pred  = predictions_classes[rf_model.predict(X)[0]]

    all_preds = [svm_pred, nb_pred, rf_pred]
    final     = mode(all_preds)

    rf_proba = rf_model.predict_proba(X)[0]
    rf_conf  = round(float(rf_proba.max()) * 100, 1)

    return {
        "svm":               svm_pred,
        "naive_bayes":       nb_pred,
        "random_forest":     rf_pred,
        "final_prediction":  final,
        "agreement":         (all_preds.count(final) / 3) * 100,
        "rf_confidence":     rf_conf,
        "matched_symptoms":  matched,
        "unmatched_symptoms": unmatched,
        "rf_proba":          rf_proba,
    }

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .app-header {
        background: linear-gradient(135deg, #1a6fe8 0%, #0f4b9e 100%);
        padding: 2rem 2.5rem; border-radius: 14px;
        color: white; margin-bottom: 1.8rem;
    }
    .app-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .app-header p  { margin: .4rem 0 0; opacity: .85; font-size: .95rem; }
    .card {
        background: #f8faff; border: 1px solid #dde8ff;
        border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: .8rem;
    }
    .card-title { font-size: .75rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: .06em;
                  color: #6b7280; margin-bottom: .25rem; }
    .card-value { font-size: 1.15rem; font-weight: 700; color: #1e3a5f; }
    .final-box {
        background: linear-gradient(135deg, #1a6fe8 0%, #0f4b9e 100%);
        border-radius: 12px; padding: 1.5rem 2rem;
        color: white; text-align: center; margin-bottom: 1.2rem;
    }
    .final-box .label { font-size: .8rem; opacity: .8;
                        text-transform: uppercase; letter-spacing: .08em; }
    .final-box .disease { font-size: 1.9rem; font-weight: 800; margin: .35rem 0 0; }
    .disclaimer {
        background: #fff8e6; border-left: 4px solid #f59e0b;
        border-radius: 6px; padding: .8rem 1.1rem;
        font-size: .82rem; color: #78350f;
    }
    .pill {
        display: inline-block; background: #e8f0fe; color: #1a56db;
        border-radius: 20px; padding: .18rem .65rem;
        font-size: .8rem; margin: .15rem; font-weight: 500;
    }
    .pill-warn { background: #fef3c7; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
try:
    svm_model, nb_model, rf_model, meta = load_artefacts()
    SYMPTOMS_LIST = sorted(meta["symptom_index"].keys())
except Exception as exc:
    st.error(f"Could not load model files from `models/` folder.\n\n`{exc}`")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🩺 Multiple Disease Prediction System</h1>
  <p>Select your symptoms and get an AI-powered prediction using an ensemble of three ML models.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        "Ensemble of **SVM, Naive Bayes, and Random Forest** trained on "
        "**41 diseases** and **132 symptoms**."
    )
    st.markdown("---")
    st.markdown("## 📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Diseases", "41")
    c2.metric("Symptoms", "132")
    c3, c4 = st.columns(2)
    c3.metric("Train Samples", "4,920")
    c4.metric("Models", "3")
    st.markdown("---")
    st.markdown("## 🛠️ Models")
    st.markdown("- Support Vector Machine\n- Gaussian Naive Bayes\n- Random Forest (200 trees)")
    st.markdown("**Final prediction** = majority vote of all three.")
    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'>⚠️ <strong>Disclaimer:</strong> "
        "For educational use only. Always consult a qualified doctor.</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_predict, tab_explore = st.tabs(["🔍 Predict Disease", "📈 Data Explorer"])

# ── Tab 1: Predict ─────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("### Select Symptoms")
    st.caption("Choose all symptoms you are experiencing. You can search by typing.")

    selected = st.multiselect(
        "Symptoms", options=SYMPTOMS_LIST,
        placeholder="Start typing a symptom …",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🔬 Predict Disease", type="primary", use_container_width=True)

    if btn:
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analysing symptoms …"):
                result = predict(selected, svm_model, nb_model, rf_model, meta)

            # Symptom pills
            st.markdown("**Symptoms used:**")
            pills = "".join(f'<span class="pill">{s}</span>' for s in result["matched_symptoms"])
            if result["unmatched_symptoms"]:
                pills += "".join(f'<span class="pill pill-warn">{s} ⚠</span>'
                                 for s in result["unmatched_symptoms"])
            st.markdown(pills, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Final prediction banner
            st.markdown(
                f'<div class="final-box"><div class="label">Final Prediction (Ensemble Vote)</div>'
                f'<div class="disease">{result["final_prediction"]}</div></div>',
                unsafe_allow_html=True,
            )

            # Per-model cards
            c1, c2, c3 = st.columns(3)
            for col, label, key in [
                (c1, "SVM", "svm"),
                (c2, "Naive Bayes", "naive_bayes"),
                (c3, "Random Forest", "random_forest"),
            ]:
                col.markdown(
                    f'<div class="card"><div class="card-title">{label}</div>'
                    f'<div class="card-value">{result[key]}</div></div>',
                    unsafe_allow_html=True,
                )

            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Model Agreement", f"{result['agreement']:.0f}%")
            m2.metric("RF Confidence",   f"{result['rf_confidence']}%")

            # Probability chart
            proba   = result["rf_proba"]
            classes = meta["predictions_classes"]
            top_idx = proba.argsort()[-8:][::-1]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors  = ["#1a6fe8" if classes[i] == result["final_prediction"]
                       else "#93c5fd" for i in top_idx]
            ax.barh([classes[i] for i in top_idx],
                    [proba[i] * 100 for i in top_idx],
                    color=colors, edgecolor="none")
            ax.set_xlabel("Probability (%)")
            ax.set_title("Top Predicted Diseases (Random Forest)",
                         fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ── Tab 2: Explorer ────────────────────────────────────────────────────────────
with tab_explore:
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Training.csv")
    if not os.path.exists(DATA_PATH):
        st.info("Training.csv not found in data/ folder.")
    else:
        df = pd.read_csv(DATA_PATH).dropna(axis=1)

        st.markdown("### Disease Distribution")
        counts = df["prognosis"].value_counts().reset_index()
        counts.columns = ["Disease", "Count"]
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.bar(counts["Disease"], counts["Count"],
                color="#1a6fe8", edgecolor="none", width=0.7)
        ax2.set_xticklabels(counts["Disease"], rotation=90, fontsize=8)
        ax2.set_ylabel("Samples")
        ax2.set_title("Sample Count per Disease", fontsize=12, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("### Top 30 Symptoms by Frequency")
        feat_cols = df.columns[:-1]
        top_sym   = df[feat_cols].sum().sort_values(ascending=False).head(30)
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.bar(top_sym.index,
                top_sym.values, color="#0f4b9e", edgecolor="none")
        ax3.set_xticklabels(
            [" ".join(w.capitalize() for w in s.split("_")) for s in top_sym.index],
            rotation=90, fontsize=8,
        )
        ax3.set_ylabel("Occurrences")
        ax3.set_title("Top 30 Most Common Symptoms", fontsize=12, fontweight="bold")
        ax3.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(50), use_container_width=True)
