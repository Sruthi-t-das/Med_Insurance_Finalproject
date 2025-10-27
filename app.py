# app.py
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb

# ---------- Config: your artifact filenames ----------
PREP_OUT       = "cost_preprocessor.pkl"
FEATS_SAFE_OUT = "cost_feature_names_safe.pkl"
MODEL_OUT      = "cost_xgb_native.json"

st.set_page_config(page_title="Medical Cost Calculator", page_icon="🏥", layout="centered")
st.title("🏥 Medical Cost Calculator")
st.caption("Powered by your XGBoost model with automatic imputation & feature engineering.")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(PREP_OUT)
    feature_names_safe = joblib.load(FEATS_SAFE_OUT)  # list[str]
    booster = xgb.Booster()
    booster.load_model(MODEL_OUT)
    return preprocessor, feature_names_safe, booster

preprocessor, FEATURE_NAMES_SAFE, booster = load_artifacts()

# ---------- Pull raw columns the preprocessor expects ----------
# NOTE: Must match the training order: ("num", ...), ("cat", ...)
try:
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])
except Exception:
    # fallback for newer sklearn shapes
    num_cols = list(preprocessor.named_transformers_["num"].feature_names_in_)
    cat_cols = list(preprocessor.named_transformers_["cat"].feature_names_in_)

RAW_COLS = num_cols + cat_cols

# ---------- Derive bands if model expects them ----------
def derive_bands(row: dict, expected_cols: list) -> dict:
    out = {}
    if "age_band" in expected_cols and "age" in row and row["age"] is not None:
        a = float(row["age"])
        out["age_band"] = (
            "<30" if a < 30 else
            "30-39" if a < 40 else
            "40-49" if a < 50 else
            "50-59" if a < 60 else
            "60-69" if a < 70 else
            "70+"
        )
    if "bmi_band" in expected_cols and "bmi" in row and row["bmi"] is not None:
        b = float(row["bmi"])
        out["bmi_band"] = (
            "under" if b < 18.5 else
            "normal" if b < 25 else
            "over" if b < 30 else
            "obese"
        )
    return out

# ---------- SAFE feature engineering (must match training) ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe FE — matches the training script.
    NO claims_count / premiums / deductible / copay usage here.
    """
    Xd = df.copy()

    # Interactions (safe)
    if {"risk_score", "chronic_count"}.issubset(Xd.columns):
        Xd["risk_chronic_interaction"] = Xd["risk_score"] * Xd["chronic_count"]
    if {"bmi", "age"}.issubset(Xd.columns):
        Xd["bmi_age_interaction"] = Xd["bmi"] * Xd["age"]

    # Prior-period utilization ONLY
    if "visits_last_year" in Xd.columns:
        Xd["visits_per_month"] = Xd["visits_last_year"] / 12.0
    if "hospitalizations_last_3yrs" in Xd.columns:
        Xd["hosp_intensity"] = Xd["hospitalizations_last_3yrs"] / 3.0
    if {"days_hospitalized_last_3yrs", "hospitalizations_last_3yrs"}.issubset(Xd.columns):
        Xd["days_per_hospitalization"] = (
            Xd["days_hospitalized_last_3yrs"] / (Xd["hospitalizations_last_3yrs"] + 1)
        )

    return Xd

# ---------- NA / dtype normalizer ----------
def normalize_for_sklearn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().replace({pd.NA: np.nan})
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_integer_dtype(dt) and df[col].isna().any():
            df[col] = df[col].astype("float64")
        if str(dt) == "boolean":
            df[col] = df[col].astype(object)
        if str(dt) == "string":
            df[col] = df[col].astype(object)
    return df.replace({pd.NA: np.nan})

# ---------- Expected categories helper ----------
def _extract_cat_cols_from_preprocessor(pp) -> list:
    try:
        return list(pp.transformers_[1][2])
    except Exception:
        return list(pp.named_transformers_["cat"].feature_names_in_)

def get_expected_labels_for(preprocessor, col_name: str):
    try:
        cat_cols_local = _extract_cat_cols_from_preprocessor(preprocessor)
        col_idx = cat_cols_local.index(col_name)

        cat_trans = preprocessor.named_transformers_.get("cat", None)
        if cat_trans is None:
            for name, trans, cols in getattr(preprocessor, "transformers_", []):
                if col_name in list(cols):
                    cat_trans = trans
                    break
        if cat_trans is None:
            return None

        if hasattr(cat_trans, "named_steps"):
            encoder = None
            for candidate in ["encoder", "onehot", "ohe"]:
                if candidate in cat_trans.named_steps:
                    encoder = cat_trans.named_steps[candidate]
                    break
            if encoder is None:
                encoder = list(cat_trans.named_steps.values())[-1]
        else:
            encoder = cat_trans

        if hasattr(encoder, "categories_"):
            return list(map(str, encoder.categories_[col_idx]))
    except Exception:
        return None
    return None

def normalize_smoker_value(raw_value: str, expected_labels: list | None) -> str:
    if raw_value is None:
        return raw_value
    v = str(raw_value).strip().lower()
    alias_map = {
        "yes": "current", "y": "current",
        "no": "never", "n": "never",
        "former smoker": "former", "ex-smoker": "former", "ex": "former",
        "non-smoker": "never", "nonsmoker": "never",
    }
    v = alias_map.get(v, v)
    if expected_labels is None:
        return v
    exp = [s.lower() for s in expected_labels]
    if {"never","former","current"}.issubset(exp):
        if v in {"yes","y"}: return "current"
        if v in {"no","n"}:  return "never"
        return v
    if {"yes","no"}.issubset(exp):
        if v == "current": return "yes"
        if v in {"former","never"}: return "no"
        return v
    return v

# ---------- UI ----------
st.subheader("Basic Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
    smoker = st.selectbox("Smoking status", ["never", "former", "current"])
with col2:
    chronic_count = st.number_input("Chronic conditions (count)", min_value=0, max_value=10, value=1, step=1)
    risk_score = st.number_input("Risk score", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    hypertension = st.selectbox("Hypertension", ["no", "yes"])

with st.expander("Advanced (optional)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        visits_last_year = st.number_input("Visits last year", min_value=0, max_value=200, value=0, step=1)
        hospitalizations_last_3yrs = st.number_input("Hospitalizations (last 3 yrs)", min_value=0, max_value=50, value=0, step=1)
        days_hospitalized_last_3yrs = st.number_input("Days hospitalized (last 3 yrs)", min_value=0, max_value=365, value=0, step=1)
    with c2:
        diabetes = st.selectbox("Diabetes", ["no", "yes"])
        asthma = st.selectbox("Asthma", ["no", "yes"])
        copd = st.selectbox("COPD", ["no", "yes"])
    with c3:
        st.markdown("Plan fields removed for new-customer flow.")

# Build user input dict (NO plan/claims fields here)
user_inputs = {
    "age": age,
    "bmi": bmi,
    "smoker": smoker,
    "chronic_count": chronic_count,
    "risk_score": risk_score,
    "hypertension": hypertension,
    # advanced
    "visits_last_year": visits_last_year,
    "hospitalizations_last_3yrs": hospitalizations_last_3yrs,
    "days_hospitalized_last_3yrs": days_hospitalized_last_3yrs,
    "diabetes": diabetes,
    "asthma": asthma,
    "copd": copd,
}

# Derive bands if needed by the model
user_inputs.update(derive_bands(user_inputs, RAW_COLS))

def build_full_row(user_dict: dict, raw_cols: list) -> pd.DataFrame:
    row = {c: user_dict.get(c, np.nan) for c in raw_cols}
    return pd.DataFrame([row])

# ---------- Predict handler ----------
if st.button("Predict Annual Medical Cost", type="primary"):
    try:
        # Align smoker label to what the encoder expects
        expected_smoker_labels = get_expected_labels_for(preprocessor, "smoker") if "smoker" in RAW_COLS else None
        user_inputs["smoker"] = normalize_smoker_value(user_inputs.get("smoker"), expected_smoker_labels)

        # Assemble raw row and add engineered features
        X_raw = build_full_row(user_inputs, RAW_COLS)
        X_fe = add_features(X_raw)

        # -------- ENFORCE TRAIN-TIME SCHEMA (dtypes + order) --------
        if len(num_cols):
            X_fe[num_cols] = X_fe[num_cols].apply(pd.to_numeric, errors="coerce")
        for c in cat_cols:
            if c in X_fe.columns:
                X_fe[c] = X_fe[c].astype(object)
        X_fe = X_fe[num_cols + cat_cols]  # exact train-time order
        X_fe = normalize_for_sklearn(X_fe)
        # ------------------------------------------------------------------

        # # 🔎 Debug panel
        # with st.expander("Debug (inputs sent to model)"):
        #     missing = [c for c in RAW_COLS if c not in X_fe.columns]
        #     nan_rate = float(X_fe.isna().mean().mean())
        #     st.write(f"Missing expected cols: {len(missing)}")
        #     st.write(f"Mean NaN rate before imputation: {nan_rate:.2%}")
        #     st.dataframe(X_fe.head(1).T)

        # Transform and predict
        X_enc = preprocessor.transform(X_fe)
        dnew = xgb.DMatrix(X_enc, feature_names=FEATURE_NAMES_SAFE)
        y_log = booster.predict(dnew)
        pred_cost = float(np.expm1(y_log)[0])

        st.subheader("📊 Prediction")
        st.metric("Predicted Annual Medical Cost", f"${pred_cost:,.2f}")
        st.caption("Note: Missing fields are imputed using the training pipeline. Bands (age/bmi) are auto-derived if required.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Tip: Use the Advanced section for more accurate results. Your model imputes any fields you don’t provide.")
