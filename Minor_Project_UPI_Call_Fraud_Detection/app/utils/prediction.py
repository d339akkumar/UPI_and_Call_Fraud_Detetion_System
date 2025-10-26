import os
import joblib
import yaml
import numpy as np
from tensorflow.keras.models import load_model
from utils.feature_engineering_upi import prepare_upi_features
from utils.feature_engineering_cdr import prepare_cdr_features  # ✅ added


# Model Loading

def load_models(mode="upi"):
    """
    Load ensemble models dynamically from YAML config.
    Supports both UPI and CDR configurations.
    """
    current_dir = os.path.dirname(__file__)               # .../app/utils
    base_dir = os.path.dirname(current_dir)               # .../app
    config_path = os.path.join(base_dir, "config", "model_paths.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")

    with open(config_path, "r") as f:
        paths = yaml.safe_load(f)[mode]

    project_root = os.path.dirname(base_dir)
    abs_paths = {k: os.path.join(project_root, v) for k, v in paths.items()}

    # Validate existence
    missing = [p for p in abs_paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("⚠️ Missing model files:\n" + "\n".join(missing))

    models = {
        "xgb": joblib.load(abs_paths["xgb"]),
        "rf": joblib.load(abs_paths["rf"]),
        "lr": joblib.load(abs_paths["lr_meta"]),
        "iso": joblib.load(abs_paths["iso"]),
        "ae": load_model(abs_paths["ae"]),
        "scaler_lr": joblib.load(abs_paths["scaler_lr"]),
        "scaler_iso": joblib.load(abs_paths["scaler_iso"]),
        "scaler_ae": joblib.load(abs_paths["scaler_ae"]),
    }

    print(f"✅ {mode.upper()} ensemble models loaded successfully.")
    return models


# Normalization Helper

def normalize_probabilities(probs):
    """Rescale probabilities between 0–1 to avoid saturation bias."""
    probs = np.array(probs)
    if np.allclose(np.min(probs), np.max(probs)):
        return np.zeros_like(probs)
    return (probs - np.min(probs)) / (np.max(probs) - np.min(probs))


# Supervised Ensemble Prediction (mode-aware)

def predict_supervised(models, df_raw, mode="upi", normalize=True):
    """
    Generate ensemble predictions using trained models:
    XGB + RF (weighted) + Meta Logistic Regression (scaled features)
    Supports both UPI and CDR datasets.
    """

    #  Prepare engineered features
    if mode == "upi":
        X = prepare_upi_features(df_raw)
    elif mode == "cdr":
        X = prepare_cdr_features(df_raw)
    else:
        raise ValueError(f"❌ Unknown mode '{mode}'. Expected 'upi' or 'cdr'.")

    #  Base model predictions
    p_xgb = models["xgb"].predict_proba(X)[:, 1]
    p_rf = models["rf"].predict_proba(X)[:, 1]
    ensemble_score = (p_xgb * 0.6) + (p_rf * 0.4)

    #  Meta model refinement
    X_scaled = models["scaler_lr"].transform(X)
    p_meta = models["lr"].predict_proba(X_scaled)[:, 1]

    #  Combine both
    p_final = (ensemble_score * 0.5) + (p_meta * 0.5)

    if normalize:
        p_final = normalize_probabilities(p_final)

    return p_final

# Unsupervised Anomaly Models

def predict_unsupervised(models, df_raw, mode="upi"):
    """
    Compute anomaly scores using IsolationForest + Autoencoder.
    Supports both UPI and CDR feature sets.
    """
    if mode == "upi":
        X = prepare_upi_features(df_raw)
    elif mode == "cdr":
        X = prepare_cdr_features(df_raw)
    else:
        raise ValueError(f"❌ Unknown mode '{mode}'.")

    iso_scores = -models["iso"].score_samples(X)
    p_iso = models["scaler_iso"].transform(iso_scores.reshape(-1, 1)).ravel()

    reconstructions = models["ae"].predict(X, verbose=0)
    reconstruction_error = np.mean(np.square(X - reconstructions), axis=1)
    p_ae = models["scaler_ae"].transform(reconstruction_error.reshape(-1, 1)).ravel()

    return p_iso, p_ae


# Risk Categorization (Mode-Specific Static Thresholds)

def apply_risk_buckets(probabilities, mode="upi", config_path=None):
    """
    Assign risk levels using mode-specific static thresholds.
    

    """

    probs = np.array(probabilities)

    # --- Mode-specific static thresholds ---
    if mode == "upi":
        high_thresh = 0.97
        medium_thresh = 0.95
    elif mode == "cdr":
        high_thresh = 0.70
        medium_thresh = 0.30
    else:
        raise ValueError(f"❌ Unknown mode '{mode}' — expected 'upi' or 'cdr'")

    # --- Categorization ---
    def categorize(p):
        if p >= high_thresh:
            return "🔴 High"
        elif p >= medium_thresh:
            return "🟠 Medium"
        else:
            return "🟢 Low"

    risk_buckets = [categorize(p) for p in probs]

    # --- Optional: Save last thresholds for reference ---
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "last_thresholds.yaml"), "w") as f:
        yaml.dump(
            {mode: {"medium": float(medium_thresh), "high": float(high_thresh)}},
            f,
            default_flow_style=False
        )

    print(f"[{mode.upper()} thresholds] High: {high_thresh}, Medium: {medium_thresh}")
    return risk_buckets


