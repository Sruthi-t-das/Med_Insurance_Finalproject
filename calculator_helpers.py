# calculator_helpers.py
import numpy as np, pandas as pd

ZERO_COUNT_DEFAULTS = {
    "claims_count": 0, "visits_last_year": 0,
    "proc_imaging_count": 0, "proc_lab_count": 0, "proc_surgery_count": 0,
    "proc_consult_count": 0, "proc_physio_count": 0,
    "hospitalizations_last_3yrs": 0, "days_hospitalized_last_3yrs": 0,
    "policy_changes_last_2yrs": 0, "dependents": 0, "medication_count": 0,
}

def derive_bands(row: dict):
    out = {}
    if "age" in row:
        a = float(row["age"])
        out["age_band"] = "u30" if a < 30 else "30s" if a < 40 else "40s" if a < 50 else "50s" if a < 60 else "60+"
    if "bmi" in row:
        b = float(row["bmi"])
        out["bmi_band"] = "underweight" if b < 18.5 else "normal" if b < 25 else "overweight" if b < 30 else "obese"
    return out

def build_full_row_from_user(user_inputs: dict, required_cols: list) -> pd.DataFrame:
    """Return a 1-row DataFrame with all columns model expects."""
    user_inputs = {**user_inputs, **derive_bands(user_inputs)}
    row = {}
    for col in required_cols:
        if col in user_inputs:
            row[col] = user_inputs[col]
        elif col in ZERO_COUNT_DEFAULTS:
            row[col] = ZERO_COUNT_DEFAULTS[col]
        else:
            row[col] = np.nan  # will be imputed (median/"unknown")
    return pd.DataFrame([row])
