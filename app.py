"""
app.py
------
Streamlit web application for Multiple Disease Prediction.
Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── make utils importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from predictor import load_artefacts, predict

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* global font */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* header strip */
    .app-header {
        background: linear-gradient(135deg, #1a6fe8 0%, #0f4b9e 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.8rem;
    }
    .app-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .app-header p  { margin: .4rem 0 0; opacity: .85; font-size: .95rem; }

    /* result cards */
    .card {
        background: #f8faff;
        border: 1px solid #dde8ff;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: .8rem;
    }
    .card-title { font-size: .75rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: .06em;
                  color: #6b7280; margin-bottom: .25rem; }
    .card-value { font-size: 1.15rem; font-weight: 700; color: #1e3a5f; }

    /* final prediction highlight */
    .final-box {
        background: linear-gradient(135deg, #1a6fe8 0%, #0f4b9e 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: white;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .final-box .label { font-size: .8rem; opacity: .8;
                        text-transform: uppercase; letter-spacing: .08em; }
    .final-box .disease { font-size: 1.9rem; font-weight: 800;
                          margin: .35rem 0 0; }

    /* disclaimer */
    .disclaimer {
        background: #fff8e6;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: .8rem 1.1rem;
        font-size: .82rem;
        color: #78350f;
    }

    /* symptom pill */
    .pill {
        display: inline-block;
        background: #e8f0fe;
        color: #1a56db;
        border-radius: 20px;
        padding: .18rem .65rem;
        font-size: .8rem;
        margin: .15rem;
        font-weight: 500;
    }
    .pill-warn {
        background: #fef3c7;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# ── load artefacts (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models …")
def get_models():
    return load_artefacts()

try:
    svm_model, nb_model, rf_model, encoder, meta = get_models()
    SYMPTOMS_LIST = sorted(meta["symptom_index"].keys())
except Exception as exc:
    st.error(f"Could not load model artefacts. Run `utils/train_models.py` first.\n\n{exc}")
    st.stop()

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🩺 Multiple Disease Prediction System</h1>
  <p>Select your symptoms below and get an AI-powered disease prediction using an ensemble of three ML models.</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar: model info ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        "This system uses an **ensemble of three classifiers** — SVM, Naive Bayes, "
        "and Random Forest — trained on a symptom-to-disease dataset covering "
        "**41 diseases** and **132 symptoms**."
    )

    st.markdown("---")
    st.markdown("## 📊 Dataset Stats")
    col1, col2 = st.columns(2)
    col1.metric("Diseases", "41")
    col2.metric("Symptoms", "132")

    train_path = os.path.join(os.path.dirname(__file__), "data", "Training.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path).dropna(axis=1)
        col3, col4 = st.columns(2)
        col3.metric("Train Samples", f"{len(df):,}")
        col4.metric("Models", "3")

    st.markdown("---")
    st.markdown("## 🛠️ Models Used")
    st.markdown("- Support Vector Machine (SVM)\n- Gaussian Naive Bayes\n- Random Forest (200 trees)")
    st.markdown("**Final prediction** is determined by majority vote across all three models.")

    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'>⚠️ <strong>Medical Disclaimer:</strong> "
        "This tool is for educational purposes only and does not constitute "
        "medical advice. Always consult a qualified healthcare professional.</div>",
        unsafe_allow_html=True,
    )

# ── main layout ────────────────────────────────────────────────────────────────
tab_predict, tab_explore = st.tabs(["🔍 Predict Disease", "📈 Data Explorer"])

# ── TAB 1: Predict ─────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("### Select Symptoms")
    st.caption("Choose all symptoms you are experiencing. You can search by typing in the box.")

    selected_symptoms = st.multiselect(
        "Symptoms",
        options=SYMPTOMS_LIST,
        placeholder="Start typing a symptom …",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔬 Predict Disease", type="primary", use_container_width=True)

    if predict_btn:
        if not selected_symptoms:
            st.warning("Please select at least one symptom before predicting.")
        else:
            with st.spinner("Analysing symptoms …"):
                result = predict(selected_symptoms, svm_model, nb_model, rf_model, meta)

            # ── matched / unmatched pills ──────────────────────────────────────
            st.markdown("**Symptoms used in prediction:**")
            pills_html = "".join(f'<span class="pill">{s}</span>' for s in result["matched_symptoms"])
            if result["unmatched_symptoms"]:
                pills_html += "".join(
                    f'<span class="pill pill-warn">{s} ⚠</span>'
                    for s in result["unmatched_symptoms"]
                )
            st.markdown(pills_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── final prediction banner ────────────────────────────────────────
            st.markdown(
                f"""<div class="final-box">
                      <div class="label">Final Prediction (Ensemble Vote)</div>
                      <div class="disease">{result['final_prediction']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

            # ── individual model cards ─────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="card"><div class="card-title">SVM</div>'
                    f'<div class="card-value">{result["svm"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="card"><div class="card-title">Naive Bayes</div>'
                    f'<div class="card-value">{result["naive_bayes"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="card"><div class="card-title">Random Forest</div>'
                    f'<div class="card-value">{result["random_forest"]}</div></div>',
                    unsafe_allow_html=True,
                )

            # ── confidence metrics ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Model Agreement", f"{result['agreement']:.0f}%")
            m2.metric("RF Confidence", f"{result['rf_confidence']}%")

            # ── RF probability bar chart ───────────────────────────────────────
            proba   = rf_model.predict_proba(
                np.array([
                    [1 if meta["symptom_index"].get(s, -1) == i else 0
                     for i in range(len(meta["symptom_index"]))]
                    for s in [",".join(selected_symptoms)]
                ])[0].reshape(1, -1)
            )[0]
            classes = meta["predictions_classes"]
            top_n   = 8
            top_idx = proba.argsort()[-top_n:][::-1]

            fig, ax = plt.subplots(figsize=(8, 3.5))
            colors  = ["#1a6fe8" if classes[i] == result["final_prediction"]
                       else "#93c5fd" for i in top_idx]
            ax.barh([classes[i] for i in top_idx], [proba[i]*100 for i in top_idx],
                    color=colors, edgecolor="none")
            ax.set_xlabel("Probability (%)")
            ax.set_title("Top Predicted Diseases (Random Forest)", fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ── TAB 2: Explorer ────────────────────────────────────────────────────────────
with tab_explore:
    if not os.path.exists(train_path):
        st.info("Training.csv not found in data/ folder.")
    else:
        df_full = pd.read_csv(train_path).dropna(axis=1)

        st.markdown("### Disease Distribution in Training Data")
        disease_counts = df_full["prognosis"].value_counts().reset_index()
        disease_counts.columns = ["Disease", "Count"]

        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.bar(disease_counts["Disease"], disease_counts["Count"],
                color="#1a6fe8", edgecolor="none", width=0.7)
        ax2.set_xticklabels(disease_counts["Disease"], rotation=90, fontsize=8)
        ax2.set_ylabel("Number of Samples")
        ax2.set_title("Sample Count per Disease", fontsize=12, fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("### Top Symptoms by Frequency")
        feature_cols   = df_full.columns[:-1]
        symptom_totals = df_full[feature_cols].sum().sort_values(ascending=False).head(30)

        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.bar(symptom_totals.index,
                symptom_totals.values,
                color="#0f4b9e", edgecolor="none")
        ax3.set_xticklabels(
            [" ".join(w.capitalize() for w in s.split("_"))
             for s in symptom_totals.index],
            rotation=90, fontsize=8
        )
        ax3.set_ylabel("Occurrences")
        ax3.set_title("Top 30 Most Common Symptoms", fontsize=12, fontweight="bold")
        ax3.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        st.markdown("### Raw Training Data Preview")
        st.dataframe(df_full.head(50), use_container_width=True)
