# train_clean_model.py
import json, re, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# ========= 0) LOAD YOUR DATA (EDIT THIS PATH) =========
df = pd.read_csv(r"C:\Users\sruth\Documents\Greenbootcamps\Med_Insurance_Finalproject\Data\medical_insurance_clean.csv")

# ========= 1) TARGETS =========
COST_COL, PREM_COL = "annual_medical_cost", "annual_premium"
TARGETS = [COST_COL, PREM_COL]

# ========= 2) DROP OBVIOUS IDENTIFIERS =========
df = df.drop(columns=["person_id"], errors="ignore")

# ========= 3) DEFINE LEAKAGE FILTER =========
# Exact columns that leak cost/premium or are post-outcome/admin
EXACT_DROP = {
    COST_COL, PREM_COL,
    "monthly_premium",
    "total_claims_paid", "avg_claim_amount", "claims_count",
    "had_major_procedure", "is_high_risk",
    "deductible", "copay",
    "policy_term_years", "policy_changes_last_2yrs",
    "provider_quality",
    "plan_type", "network_tier",
}

# Pattern-based drops: any column name matching these is removed
# (keeps it simple; you already remove targets explicitly)
PATTERNS = [
    r"claim",             # any claims-derived column
    r"cost",              # cost-derived engineered cols (ok to drop)
    r"payout|paid|reimb|settlement",
    r"proc_.*",           # procedure counts (imaging/surgery/etc.)
    r"premium",           # any other premium fields
]

pat = re.compile("|".join(PATTERNS), re.I)
all_cols = df.columns.tolist()

# Build SAFE feature list = everything not in targets/EXACT_DROP and not matching patterns
safe_features = [
    c for c in all_cols
    if (c not in EXACT_DROP)
    and (c not in TARGETS)
    and (not pat.search(c))
]

# Sanity: ensure targets present
for t in TARGETS:
    if t not in df.columns:
        raise ValueError(f"Missing target column: {t}")

# X, y
X = df[safe_features].copy()
y = df[TARGETS].copy()

# ========= 4) TYPES =========
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"Using {len(safe_features)} safe features "
      f"(Numeric={len(num_cols)}, Categorical={len(cat_cols)})")
print("Examples:", safe_features[:12], "...")

# ========= 5) PREPROCESSORS WITH IMPUTERS =========
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

# IMPORTANT: make OHE dense so RF never gets a sparse matrix
# (scikit-learn >=1.2 supports 'sparse_output=False'; for older versions use 'sparse=False')
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False))
])

pre = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ],
    remainder="drop"
)

# ========= 6) MODEL =========
model = Pipeline([
    ("preprocessor", pre),
    ("regressor", RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=2,
        max_depth=18,          # <- optional: caps tree depth for speed/regularization
        random_state=42,
        n_jobs=-1
    ))
])

# ========= 7) TRAIN / TEST =========
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)

# ========= 8) EVALUATE =========
y_hat = model.predict(X_te)

def eval_one(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    evs  = explained_variance_score(y_true, y_pred)
    return mae, rmse, r2, evs

for i, t in enumerate(TARGETS):
    mae, rmse, r2, evs = eval_one(y_te[t], y_hat[:, i])
    print(f"{t:25} | MAE {mae:,.2f}  RMSE {rmse:,.2f}  R² {r2:.3f}  EVS {evs:.3f}")

# ========= 9) SAVE MODEL + SCHEMA =========
# Keep raw feature order for your app
setattr(model, "raw_feature_names_", X.columns.tolist())
setattr(model, "target_names_", TARGETS)
joblib.dump(model, "rf_clean_model.pkl")
print("✅ Saved rf_clean_model.pkl")

# Save schema so the app can render inputs and defaults
schema = {
    "numeric_cols": num_cols,
    "categorical_cols": cat_cols,
    "numeric_medians": {c: float(np.nanmedian(X[c])) for c in num_cols},
    "categorical_top_values": {
        c: X[c].astype(str).value_counts().head(12).index.tolist() for c in cat_cols
    },
    "safe_features": safe_features
}
with open("rf_clean_model.schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)
print("✅ Saved rf_clean_model.schema.json")
