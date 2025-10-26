import numpy as np
import pandas as pd

def prepare_cdr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for Call Fraud Detection (CDR).

    Produces the exact features the XGBoost model expects:
    ['call_duration', 'call_cost', 'cost_per_sec', 'call_hour',
     'distinct_callees_last_24h', 'tower_switch_rate',
     'repeated_short_calls_last_1h',
     'type_VoIP', 'type_international', 'type_roaming', 'type_voice']

    Fully defensive — handles missing, malformed, or wrong-type inputs.
    """

    df = df.copy()

    # --- Ensure numeric columns exist and are valid ---
    numeric_cols = {
        "call_duration": 0.0,
        "call_cost": 0.0,
        "cost_per_sec": 0.0,
        "call_hour": 0.0,
        "distinct_callees_last_24h": 0.0,
        "tower_switch_rate": 0.0,
        "repeated_short_calls_last_1h": 0.0,
    }

    for col, default in numeric_cols.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # Clip unrealistic hour values
    df["call_hour"] = df["call_hour"].clip(lower=0, upper=23)

    # --- Handle call_type safely ---
    # Some datasets or Streamlit cache switches can make it a string or missing
    if "call_type" not in df.columns:
        df["call_type"] = ""
    elif not isinstance(df["call_type"], pd.Series):
        df["call_type"] = pd.Series([str(df["call_type"])] * len(df), index=df.index)

    # Clean call_type text
    df["call_type"] = df["call_type"].astype(str).fillna("").str.lower().str.strip()

    # --- Create one-hot encoding for call_type ---
    dummies = pd.get_dummies(df["call_type"], prefix="type")

    # Normalize columns to lowercase (since get_dummies does that)
    dummies.columns = [c.lower() for c in dummies.columns]

    # Expected dummy columns (final trained model’s order)
    expected_dummies = [
        "type_voip", "type_international", "type_roaming", "type_voice"
    ]

    # Add any missing dummy columns
    for col in expected_dummies:
        if col not in dummies.columns:
            dummies[col] = 0

    # Final rename to match model’s training feature case
    rename_map = {
        "type_voip": "type_VoIP",
        "type_international": "type_international",
        "type_roaming": "type_roaming",
        "type_voice": "type_voice"
    }
    dummies = dummies.rename(columns=rename_map)

    # --- Assemble final feature matrix ---
    final_cols = [
        "call_duration", "call_cost", "cost_per_sec", "call_hour",
        "distinct_callees_last_24h", "tower_switch_rate", "repeated_short_calls_last_1h",
        "type_VoIP", "type_international", "type_roaming", "type_voice"
    ]

    # Start with numeric base
    out = pd.DataFrame(index=df.index)
    for col in final_cols[:7]:
        out[col] = df[col].astype(float)

    # Add dummy variables
    for col in final_cols[7:]:
        out[col] = dummies[col].astype(int)

    # Fill missing or NaN values for robustness
    out = out.fillna(0.0)

    return out
