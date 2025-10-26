import os
import pandas as pd
import streamlit as st

# Custom utilities
from utils.prediction import load_models, predict_supervised, apply_risk_buckets
from utils.feature_engineering_upi import prepare_upi_features
from utils.feature_engineering_cdr import prepare_cdr_features
from utils.reason_generator import generate_reason
from utils.visualization import (
    risk_distribution_bar,
    risk_distribution_pie,
    probability_scatter,
)

# Page config
st.set_page_config(page_title="💳 Unified Fraud Detection Dashboard", layout="wide")

# Welcome header
st.markdown("""
    <style>
        .welcome-box {
            background: linear-gradient(135deg, #0d1b2a, #1b263b);
            color: #ffffff;
            text-align: center;
            padding: 45px;
            border-radius: 15px;
            margin-bottom: 24px;
            box-shadow: 0px 4px 25px rgba(0,0,0,0.35);
        }
        .welcome-title { font-size: 40px; font-weight: 800; color: #00b4d8; }
        .welcome-subtitle { font-size: 18px; color: #d1d1d1; }
    </style>
    <div class="welcome-box">
        <h1 class="welcome-title">👋 Welcome to the Unified UPI & Call Fraud Detection System</h1>
        <p class="welcome-subtitle">Upload your dataset, pick the mode, and review risk insights immediately.</p>
    </div>
""", unsafe_allow_html=True)

# Step 1: upload
st.markdown("### 📂 Step 1: Upload your dataset (CSV)")
uploaded_file = st.file_uploader("", type=["csv"])

# Step 2: mode
st.markdown("### ⚙️ Step 2: Select Data Type")
mode = st.radio("Select Analysis Mode:", ["💰 UPI Transactions", "📞 Call Records"], horizontal=True)
mode_key = "upi" if "UPI" in mode else "cdr"

if uploaded_file:
    # Read
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"✅ File uploaded — {df_raw.shape[0]} rows")

    # Feature engineering
    st.info("⚙️ Preparing features…")
    if mode_key == "upi":
        df = prepare_upi_features(df_raw)
    else:
        df = prepare_cdr_features(df_raw)

    # Load models
    st.info("📦 Loading models…")
    models = load_models(mode_key)

    # Predict and risk bucket
    st.info("🔍 Scoring…")
    df["p_final"] = predict_supervised(models, df_raw, mode=mode_key, normalize=True)
    df["risk_bucket"] = apply_risk_buckets(df["p_final"], mode=mode_key)

    # Reasons
    st.info("🧩 Generating reasons…")
    df["reason"] = df.apply(lambda row: generate_reason(row, mode=mode_key), axis=1)

    # Tabs
    tab1, tab2 = st.tabs(["📊 Overview", "📋 Detailed Transactions"])

    with tab1:
        st.subheader("📈 Risk Overview")

        total = len(df)
        high = df["risk_bucket"].astype(str).str.contains("High", na=False).sum()
        med = df["risk_bucket"].astype(str).str.contains("Medium", na=False).sum()
        low = df["risk_bucket"].astype(str).str.contains("Low", na=False).sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", total)
        c2.metric("🔴 High Risk", high)
        c3.metric("🟠 Medium Risk", med)
        c4.metric("🟢 Low Risk", low)

        # three charts side-by-side
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_distribution_bar(df)
        with col2:
            risk_distribution_pie(df)
        with col3:
            probability_scatter(df)

    with tab2:
        st.subheader("📋 Detailed Transaction Records")

        if mode_key == "upi":
            display_cols = ["amount", "p_final", "risk_bucket", "reason"]
        else:
            display_cols = ["call_duration", "p_final", "risk_bucket", "reason"]

        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

        # download
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{mode_key}_predictions.csv")
        df.to_csv(out_path, index=False)

        st.download_button(
            "⬇️ Download Predictions",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{mode_key}_predictions.csv",
            mime="text/csv",
        )

else:
    st.info("📁 Please upload a dataset to begin.")
