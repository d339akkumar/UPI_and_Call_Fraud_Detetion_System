import numpy as np

def generate_reason(row, mode="upi"):
    """
    Generate human-readable reasons for high-risk predictions
    based on engineered features.

    Parameters:
        row (pd.Series): A single transaction or call record
        mode (str): 'upi' or 'cdr'

    Returns:
        str: Concise reason summary
    """
    reasons = []

    # =====================================================
    # 💳 UPI Fraud Reasons
    # =====================================================
    if mode == "upi":

        # Transaction amount patterns
        if row.get("amount_log", 0) > 11.5:
            reasons.append("Unusually large transaction amount")
        elif row.get("amount_log", 0) > 10.5:
            reasons.append("Higher-than-average transaction amount")

        # Balance mismatches
        if row.get("balance_mismatch_orig", 0) == 1:
            reasons.append("Mismatch in sender balance update")
        if row.get("balance_mismatch_dest", 0) == 1:
            reasons.append("Mismatch in receiver balance update")

        # Zero or near-zero balances
        if row.get("orig_zero_but_amount", 0) == 1:
            reasons.append("Transfer initiated from zero balance account")
        if row.get("dest_zero_but_amount", 0) == 1:
            reasons.append("Receiver had zero balance before transfer")

        # Rapid or abnormal transaction intensity
        if row.get("sender_tx_count", 0) > 100:
            reasons.append("Unusually high sender activity volume")
        if row.get("sender_amount_std", 0) > 5e5:
            reasons.append("High variation in sender’s transaction amounts")

        # Behavioral outliers
        if row.get("relative_amount_to_mean_sender", 0) > 5:
            reasons.append("Amount deviates sharply from sender’s usual pattern")
        if row.get("balance_gap_ratio", 0) > 0.8:
            reasons.append("Large balance fluctuation detected")

        # Relationship-based flags
        if row.get("is_same_sender_receiver", 0) == 1:
            reasons.append("Sender and receiver accounts are identical")
        if row.get("is_large_transfer", 0) == 1:
            reasons.append("Marked as large-value transfer")

        # Fallback if none match
        if not reasons:
            reasons.append("Transaction shows irregular account or balance behavior")

    # =====================================================
    # 📞 CDR Fraud Reasons
    # =====================================================
    else:
        if row.get("call_duration", 0) < 5:
            reasons.append("Very short call duration pattern")
        if row.get("tower_switch_rate", 0) > 0.6:
            reasons.append("High tower switching frequency detected")
        if row.get("repeated_short_calls_last_1h", 0) > 3:
            reasons.append("Repeated short-duration calls within an hour")
        if row.get("distinct_callees_last_24h", 0) > 50:
            reasons.append("Abnormally high number of distinct callees")
        if row.get("is_international", 0) == 1:
            reasons.append("International call detected")
        if row.get("call_cost", 0) > 200:
            reasons.append("High call cost relative to duration")
        if row.get("tower_switch_rate", 0) > 0.9:
            reasons.append("Extreme cell tower switching, potential SIM-box activity")

        if not reasons:
            reasons.append("Normal call pattern detected")

    # Return combined reasons (limit to 4 for clarity)
    return " + ".join(reasons[:4])
