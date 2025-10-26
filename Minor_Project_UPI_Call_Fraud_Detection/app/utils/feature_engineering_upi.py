import numpy as np
import pandas as pd
import yaml
import os


def prepare_upi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely recreate engineered features for UPI Fraud Detection.

    ✅ Works for both raw PaySim-like input and pre-engineered datasets.
    ✅ Automatically handles missing transaction types and placeholder features.
    ✅ Guarantees final feature order matches model training (from YAML).
    """

    df = df.copy()

    # Drop label columns if present
    df.drop(columns=["isFraud", "isFlaggedFraud"], errors="ignore", inplace=True)

    # 1. Core Transformations
   
    if "amount" in df.columns:
        df["amount_log"] = np.log1p(df["amount"])

    if {"oldbalanceOrg", "newbalanceOrig"}.issubset(df.columns):
        df["orig_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    if {"newbalanceDest", "oldbalanceDest"}.issubset(df.columns):
        df["dest_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]

    if {"orig_balance_change", "amount"}.issubset(df.columns):
        df["balance_mismatch_orig"] = ((df["orig_balance_change"] - df["amount"]).abs() > 1e-9).astype(int)

    if {"dest_balance_change", "amount"}.issubset(df.columns):
        df["balance_mismatch_dest"] = ((df["dest_balance_change"] - df["amount"]).abs() > 1e-9).astype(int)

    if {"oldbalanceOrg", "amount"}.issubset(df.columns):
        df["orig_zero_but_amount"] = ((df["oldbalanceOrg"] == 0) & (df["amount"] > 0)).astype(int)

    if {"oldbalanceDest", "amount"}.issubset(df.columns):
        df["dest_zero_but_amount"] = ((df["oldbalanceDest"] == 0) & (df["amount"] > 0)).astype(int)

    if {"newbalanceOrig", "oldbalanceOrg"}.issubset(df.columns):
        df["orig_balance_ratio"] = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1e-9)

    if {"newbalanceDest", "oldbalanceDest"}.issubset(df.columns):
        df["dest_balance_ratio"] = df["newbalanceDest"] / (df["oldbalanceDest"] + 1e-9)

  
    # 2. Transaction Type Dummies
  
    expected_types = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

    if "type" in df.columns:
        dummies = pd.get_dummies(df["type"], prefix="type")
        for col in expected_types:
            if col not in dummies:
                dummies[col] = 0
        df = pd.concat([df, dummies[expected_types]], axis=1)
    else:
        for col in expected_types:
            if col not in df.columns:
                df[col] = 0


    # 3. Placeholder Engineered Features

    placeholder_cols = [
        "sender_amount_mean", "sender_amount_std", "sender_amount_max", "sender_tx_count",
        "sender_balance_mismatch_ratio", "sender_zero_but_ratio",
        "dest_amount_mean", "dest_amount_std", "dest_tx_count", "dest_zero_but_ratio",
        "tx_per_step_orig", "amount_per_step_orig", "avg_amount_per_step_orig",
        "balance_gap_ratio", "relative_amount_to_mean_sender", "amount_to_balance_orig_ratio",
        "amount_balance_gap", "is_large_transfer", "is_same_sender_receiver",
        "is_merchant_dest", "is_customer_dest", "amount_to_sender_mean_ratio",
        "amount_to_balance_gap_ratio", "sender_activity_intensity", "type_encoded"
    ]
    for col in placeholder_cols:
        if col not in df.columns:
            df[col] = 0.0


    # 4. Drop Unnecessary Columns

    df.drop(columns=["step", "nameOrig", "nameDest", "type"], errors="ignore", inplace=True)


    # 5. Enforce Exact Training Column Order
   
    # Find feature_order.yaml (even if run from Streamlit root)
    current_dir = os.path.dirname(__file__)           # .../app/utils
    base_dir = os.path.dirname(current_dir)           # .../app
    config_path = os.path.join(base_dir, "config", "feature_order.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ feature_order.yaml not found at {config_path}")

    with open(config_path, "r") as f:
        training_order = yaml.safe_load(f)["upi_features"]

    # Add missing columns if necessary
    for col in training_order:
        if col not in df.columns:
            df[col] = 0.0

    # Reorder strictly to match model training
    df = df[training_order]


    #  6. Return Ready-to-Predict DataFrame
 
    return df
