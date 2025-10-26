import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Consistent colors for risk buckets with emoji labels
RISK_COLORS = {
    "🔴 High": "#e63946",
    "🟠 Medium": "#f77f00",
    "🟢 Low": "#2a9d8f",
    "High": "#e63946",
    "Medium": "#f77f00",
    "Low": "#2a9d8f",
}

def _clean_risk_labels(series: pd.Series) -> pd.Series:
    return series.replace({"🔴 High": "High", "🟠 Medium": "Medium", "🟢 Low": "Low"})

def risk_distribution_bar(df: pd.DataFrame):
    data = _clean_risk_labels(df["risk_bucket"])
    counts = data.value_counts()

    fig, ax = plt.subplots(figsize=(4, 4))
    colors = [RISK_COLORS.get(k, "#888") for k in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="white",
        )
    ax.set_title("Risk Distribution", color="white")
    ax.set_xlabel("Risk Level", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    st.pyplot(fig, use_container_width=True)

def risk_distribution_pie(df: pd.DataFrame):
    data = _clean_risk_labels(df["risk_bucket"])
    counts = data.value_counts()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        counts.values,
        labels=[f"{k}" for k in counts.index],
        autopct="%1.1f%%",
        startangle=120,
        colors=[RISK_COLORS.get(k, "#888") for k in counts.index],
        textprops={"color": "white", "fontsize": 10},
    )
    ax.set_title("Risk Composition", color="white")
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig, use_container_width=True)

def probability_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = df["risk_bucket"].map(RISK_COLORS).fillna("#888")
    ax.scatter(df.index, df["p_final"], c=colors, alpha=0.75)
    ax.set_xlabel("Record Index", color="white")
    ax.set_ylabel("Fraud Probability", color="white")
    ax.set_title("Probability Distribution", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    st.pyplot(fig, use_container_width=True)

def render_overview_charts(df: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_distribution_bar(df)
    with col2:
        risk_distribution_pie(df)
    with col3:
        probability_scatter(df)
