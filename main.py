# -*- coding: utf-8 -*-
# =============================================================================
#  BANKING TRANSACTION ANOMALY DETECTION - PFE FINAL YEAR PROJECT
#  Author  : [Your Name]
#  Method  : Isolation Forest (Unsupervised) + Mahalanobis Distance (Bonus)
#  Dataset : Data.xlsm  ->  sheet "DETAIL"  (Tunisian Bank SWIFT transfers)
# =============================================================================
#
#  DATA DESCRIPTION (DETAIL sheet)
#  --------------------------------
#  13,004 international wire transfer (SWIFT) transactions
#  Columns used for anomaly detection:
#   - MNT          : Transaction amount (original currency)
#   - CV EN TND    : Amount converted to Tunisian Dinar (TND)
#   - FC___        : Customer ID (Code Client / Fiche Client — anonymized)
#   - DEV          : Currency (EUR, USD, GBP, CAD ...)
#   - FRAIS        : Fee type (OUR=sender pays, SHA=shared, BEN=receiver pays)
#   - 52A_ISO_PAYS : ISO country code of sender bank
#   - Libellé_cpt_ : Account type label
#   - 52A_EXPED    : Correspondent bank BIC code
#   - Date features: month, day, weekday extracted from Date
#   - processing_delay : days between transaction Date and execution (D_EXEC)
#   - value_delay      : days between value date (DATE_VL) and transaction Date
#
#  HOW ISOLATION FOREST WORKS
#  ---------------------------
#  Normal transactions need MANY random splits to be isolated (they blend in).
#  Anomalies (frauds) need VERY FEW splits (they stand out from the crowd).
#  Score:  -1 = ANOMALY (potential fraud)    1 = NORMAL transaction
#
# =============================================================================

import sys
import os
import joblib

# Force UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive: saves PNGs without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import mahalanobis
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")

print("=" * 70)
print("   BANKING TRANSACTION ANOMALY DETECTION SYSTEM")
print("   Dataset: Data.xlsm (DETAIL) -- SWIFT / Wire Transfers -- UIB Tunisia")
print("=" * 70)


# =============================================================================
# SECTION 1 -- CONFIGURATION
# =============================================================================

DATA_FILE      = "Data.xlsm"         # Input Excel macro-enabled workbook
SHEET_NAME     = "DETAIL"            # Raw transaction sheet (not pivot sheet TCD)
CONTAMINATION  = 0.05                # Expected anomaly rate
N_ESTIMATORS   = 200                 # Number of isolation trees
OUTPUT_FILE    = "fraud_detection_results.xlsx"
PLOTS_DIR      = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# High-risk country ISO codes (official FATF lists — updated 2026)
# Source: https://www.fatf-gafi.org/

# FATF BLACKLIST (3 pays — Contre-mesures strictes)
FATF_BLACKLIST = {"KP", "IR", "MM"}  # Corée du Nord, Iran, Myanmar

# FATF GREY LIST (23 pays — Surveillance renforcée)
FATF_GREYLIST = {
    # Afrique
    "DZ", "AO", "CM", "CI", "KE", "NA", "CD", "SS",
    # Amériques
    "BO", "HT", "VG", "VE",
    # Asie, Moyen-Orient, Océanie
    "KW", "LA", "LB", "NP", "PG", "SY", "VN", "YE",
    # Europe
    "BG", "HR", "MC",
}

HIGH_RISK_COUNTRIES = FATF_BLACKLIST | FATF_GREYLIST


# =============================================================================
# SECTION 2 -- LOAD DATA
# =============================================================================
print(f"\n[1/9] Loading '{DATA_FILE}' -> sheet '{SHEET_NAME}' ...")

df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
print(f"  [OK] {df_raw.shape[0]:,} transactions x {df_raw.shape[1]} columns")
print(f"\n  Columns: {list(df_raw.columns)}\n")
print(df_raw.head(3).to_string())
print()


# =============================================================================
# SECTION 3 -- DATA PREPROCESSING
# =============================================================================
print("[2/9] Preprocessing ...")

df = df_raw.copy()

# ── 3a. Handle missing values ─────────────────────────────────────────────────
print(f"  Missing values before: {df.isnull().sum().sum()}")

# ── Create missing-value indicators BEFORE imputing ───────────────────────────
# A missing date or amount is itself suspicious — record this signal
if "ASSOC" in df.columns:
    df["assoc_present"] = df["ASSOC"].notna().astype(int)
    print("  [OK] assoc_present flag")
if "chps_72" in df.columns:
    df["chps_72_present"] = df["chps_72"].notna().astype(int)
    print("  [OK] chps_72_present flag")

MISSING_FLAG_COLS = ["MNT", "CV EN TND"]
for col in MISSING_FLAG_COLS:
    if col in df.columns and df[col].isna().any():
        df[f"{col}_missing"] = df[col].isna().astype(int)
        print(f"  [OK] {col}_missing flag ({df[f'{col}_missing'].sum()} NaN rows)")

# Drop columns that cannot be used for ML:
DROP_COLS = ["ETAT", "ASSOC", "INFO_Champs_70", "chps_72", "chps_57",
             "Nom_pays", "CHRONO", "AGE"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Fill remaining numeric NaN with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical NaN with 'UNKNOWN'
for col in df.select_dtypes(include=["object"]).columns:
    df[col].fillna("UNKNOWN", inplace=True)

print(f"  Missing values after:  {df.isnull().sum().sum()}")

# ── 3b. Date feature engineering ──────────────────────────────────────────────
# Convert to datetime and extract useful sub-features

date_col_pairs = [
    ("Date",    "txn"),       # transaction initiation date
    ("DATE_VL", "vl"),        # value date
    ("D_EXEC",  "exec"),      # execution date
]

for col, prefix in date_col_pairs:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        # Extract ONLY the weekend flag (raw calendar months/days omitted to prevent overfitting)
        df[f"{prefix}_weekend"] = df[col].dt.weekday.isin([5, 6]).astype(int)

print("  [OK] Converted dates and created weekend flags (raw calendar days/months omitted to prevent overfitting)")

# Key delay features -- large delays can indicate fraud / money laundering
if "D_EXEC" in df.columns and "Date" in df.columns:
    df["processing_delay_days"] = (df["D_EXEC"] - df["Date"]).dt.days.clip(lower=0)
    print("  [OK] processing_delay_days = D_EXEC - Date")

if "DATE_VL" in df.columns and "Date" in df.columns:
    df["value_delay_days"] = (df["DATE_VL"] - df["Date"]).dt.days
    print("  [OK] value_delay_days = DATE_VL - Date")

# ── 3b-extra. VELOCITY features (require Date column — computed before drop) ─
# These detect sudden bursts of activity from a sender, country, or account.
# Example: A sender that normally does 2 transfers/week suddenly does 15 in 3 days.
print("  Computing velocity features ...")

def _rolling_count(dataframe, group_col, date_col="Date", window="7D"):
    """Count transactions per group within a rolling time window.
    For each row, counts how many transactions from the same group
    occurred within the specified window ending at that row's date.
    """
    result = pd.Series(1.0, index=dataframe.index)
    valid = dataframe[dataframe[date_col].notna()].copy()
    for _, group in valid.groupby(group_col):
        if len(group) == 0:
            continue
        g_sorted = group.sort_values(date_col)
        g_indexed = g_sorted.set_index(date_col)
        rc = g_indexed.rolling(window, min_periods=0, closed="left")["MNT"].count()
        result.loc[g_sorted.index] = rc.values
    return result.fillna(1).astype(int)

# Feature: txn_count_sender_7d — burst of transfers from same sender bank
if "52A_EXPED" in df.columns:
    try:
        df["txn_count_sender_7d"] = _rolling_count(df, "52A_EXPED")
        print(f"  [OK] txn_count_sender_7d  (rolling 7-day count per sender)")
    except Exception as e:
        df["txn_count_sender_7d"] = df.groupby("52A_EXPED")["MNT"].transform("count")
        print(f"  [OK] txn_count_sender_7d  (fallback: global count) [{e}]")

# Feature: txn_count_country_7d — unusual volume from a specific country
if "52A_ISO_PAYS" in df.columns:
    try:
        df["txn_count_country_7d"] = _rolling_count(df, "52A_ISO_PAYS")
        print(f"  [OK] txn_count_country_7d (rolling 7-day count per country)")
    except Exception as e:
        df["txn_count_country_7d"] = df.groupby("52A_ISO_PAYS")["MNT"].transform("count")
        print(f"  [OK] txn_count_country_7d (fallback: global count) [{e}]")

# NOTE: txn_count_account_7d removed — account-type level is too aggregated
# (groups thousands of different clients) to be individually discriminating.
# Kept: txn_count_customer_7d (individual) and txn_count_sender_7d (BIC level).

# Feature: txn_count_customer_7d — burst of transfers from same customer
# A customer who normally does 1 transfer/month suddenly does 10 in 3 days.
if "FC___" in df.columns:
    try:
        df["txn_count_customer_7d"] = _rolling_count(df, "FC___")
        print(f"  [OK] txn_count_customer_7d (rolling 7-day count per customer)")
    except Exception as e:
        df["txn_count_customer_7d"] = df.groupby("FC___")["MNT"].transform("count")
        print(f"  [OK] txn_count_customer_7d (fallback: global count) [{e}]")

# Drop the original datetime columns (ML needs numbers)
df.drop(columns=["Date", "DATE_VL", "D_EXEC"], errors="ignore", inplace=True)

# ── 3c. Feature engineering -- risk indicators ─────────────────────────────────

# Flag high-risk origin countries (FATF / AML watchlist)
if "52A_ISO_PAYS" in df.columns:
    df["high_risk_country"] = df["52A_ISO_PAYS"].isin(HIGH_RISK_COUNTRIES).astype(int)
    print("  [OK] high_risk_country flag added")

# Flag weekends (unusual for SWIFT transfers)
# (already in txn_weekend above)

# Log-transform of amounts (compresses huge outliers, helps ML)
df["log_MNT"]       = np.log1p(df["MNT"])
df["log_CV_EN_TND"] = np.log1p(df["CV EN TND"])
print("  [OK] log_MNT, log_CV_EN_TND added")

# ── 3c-extra. BEHAVIORAL RATIO features ───────────────────────────────────────
# These compare each transaction's amount to the SENDER's historical average.
# Example: Bank BNPAFRPP usually sends 5,000 TND. Today it sends 150,000 TND.
#          Ratio = 150,000 / 5,000 = 30x → highly unusual → flagged.
print("  Computing behavioral ratio features ...")

if "52A_EXPED" in df.columns and "CV EN TND" in df.columns:
    # Feature: amount_vs_sender_avg — ratio of this amount to sender's average
    sender_avg = df.groupby("52A_EXPED")["CV EN TND"].transform("mean")
    df["amount_vs_sender_avg"] = (df["CV EN TND"] / sender_avg.replace(0, np.nan)).fillna(1.0)
    print("  [OK] amount_vs_sender_avg (ratio to sender mean)")

    # Feature: amount_vs_sender_std — Z-score (how many std deviations from mean)
    sender_std = df.groupby("52A_EXPED")["CV EN TND"].transform("std")
    sender_std = sender_std.fillna(1.0).replace(0.0, 1.0)  # avoid division by zero
    df["amount_vs_sender_std"] = ((df["CV EN TND"] - sender_avg) / sender_std).fillna(0.0)
    print("  [OK] amount_vs_sender_std (Z-score within sender)")

if "52A_ISO_PAYS" in df.columns and "CV EN TND" in df.columns:
    # Feature: amount_vs_country_avg — ratio of this amount to country's average
    country_avg = df.groupby("52A_ISO_PAYS")["CV EN TND"].transform("mean")
    df["amount_vs_country_avg"] = (df["CV EN TND"] / country_avg.replace(0, np.nan)).fillna(1.0)
    print("  [OK] amount_vs_country_avg (ratio to country mean)")

# ── 3c-extra. CUSTOMER BEHAVIORAL features (FC___ = Customer ID) ──────────────
# These compare each transaction to the CUSTOMER's own historical behavior.
# Example: Customer ABIHIDHB usually transfers 3,000 TND. Today: 200,000 TND.
print("  Computing customer behavioral features ...")

if "FC___" in df.columns and "CV EN TND" in df.columns:
    # Feature: amount_vs_customer_avg — ratio of this amount to customer's average
    customer_avg = df.groupby("FC___")["CV EN TND"].transform("mean")
    df["amount_vs_customer_avg"] = (df["CV EN TND"] / customer_avg.replace(0, np.nan)).fillna(1.0)
    print("  [OK] amount_vs_customer_avg (ratio to customer mean)")

    # Feature: amount_vs_customer_std — Z-score within customer's own history
    customer_std = df.groupby("FC___")["CV EN TND"].transform("std")
    customer_std = customer_std.fillna(1.0).replace(0.0, 1.0)
    df["amount_vs_customer_std"] = ((df["CV EN TND"] - customer_avg) / customer_std).fillna(0.0)
    print("  [OK] amount_vs_customer_std (Z-score within customer)")

if "FC___" in df.columns and "52A_EXPED" in df.columns:
    # Feature: customer_sender_diversity — how many different banks this customer uses
    # Normal customers use 1-2 correspondent banks. Money launderers use 5-10.
    df["customer_sender_diversity"] = df.groupby("FC___")["52A_EXPED"].transform("nunique")
    print("  [OK] customer_sender_diversity (unique senders per customer)")

# ── 3c-extra. GEOGRAPHICAL MISMATCH features ─────────────────────────────────
# These detect money being routed through intermediary countries.
# Example: Sender BIC = COMABORJ (Romania), but declared country = FR (France).
#          Mismatch! Money is being laundered through France.
print("  Computing geographical mismatch features ...")

if "52A_EXPED" in df.columns and "52A_ISO_PAYS" in df.columns:
    # Feature: country_mismatch_risk — BIC country ≠ declared origin country
    # BIC code structure: positions 5-6 (index 4:6) = country code
    # e.g. BNPAFRPP → FR, DEUTDEFF → DE, UIBKTNTT → TN
    bic_country = df["52A_EXPED"].astype(str).str[4:6].str.upper()
    declared_country = df["52A_ISO_PAYS"].astype(str).str.strip().str.upper()
    df["country_mismatch_risk"] = (
        (bic_country != declared_country) &
        (bic_country.str.len() == 2) &
        (declared_country.str.len() == 2)
    ).astype(int)
    n_mismatch = df["country_mismatch_risk"].sum()
    print(f"  [OK] country_mismatch_risk ({n_mismatch:,} mismatches found)")

    # Feature: sender_country_diversity — how many different countries this sender claims
    # A legitimate bank always shows 1 country. Shell companies may show 3-5.
    df["sender_country_diversity"] = df.groupby("52A_EXPED")["52A_ISO_PAYS"].transform("nunique")
    print("  [OK] sender_country_diversity (countries per sender BIC)")

if "DEV" in df.columns and "52A_ISO_PAYS" in df.columns:
    # Feature: rare_currency_for_country — unusual currency for this origin country
    # Example: Transaction from France in Iranian Rial → very suspicious.
    common_currency = df.groupby("52A_ISO_PAYS")["DEV"].transform(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    )
    df["rare_currency_for_country"] = (df["DEV"] != common_currency).astype(int)
    n_rare = df["rare_currency_for_country"].sum()
    print(f"  [OK] rare_currency_for_country ({n_rare:,} rare currency transactions)")

# ── 3c-extra. AMOUNT ANOMALY features ─────────────────────────────────────────
print("  Computing amount anomaly features ...")

# Feature: is_round_amount — perfectly round amounts (manual/artificial transfers)
# Legitimate business payments are rarely perfectly round (e.g., 12,847.63).
# Round amounts (e.g., exactly 10,000 or 50,000) suggest manual laundering.
df["is_round_amount"] = ((df["MNT"] % 1000 == 0) & (df["MNT"] > 0)).astype(int)
n_round = df["is_round_amount"].sum()
print(f"  [OK] is_round_amount ({n_round:,} round-amount transactions)")

# Feature: amount_rank_pct — percentile rank of amount in the entire dataset
# Instead of raw amounts, the AI sees "this is in the top 1%" → much easier to learn.
df["amount_rank_pct"] = df["CV EN TND"].rank(pct=True) * 100
print("  [OK] amount_rank_pct (percentile 0-100)")

print(f"\n  >>> Feature engineering complete: {df.shape[1]} total columns")

# Note: ETAT_RPA is the RPA processing status (VALIDE, KO, etc.)
# It is NOT a fraud label — it just says whether the robot succeeded or failed.
# We keep it for informational context only, not for model evaluation.

# ── 3d. Encode categorical variables ──────────────────────────────────────────
label_encoders = {}
# NOTE: ETAT_RPA removed from features — it is a post-processing RPA status,
# not a behavioral feature. Keeping it causes data leakage risk.
LOW_CARD_COLS = ["DEV", "FRAIS"]
HIGH_CARD_COLS = ["52A_ISO_PAYS", "Libellé_cpt_", "52A_EXPED", "chps_52A", "FC___"]

# 1. One-Hot Encoding for Low Cardinality Features
for col in LOW_CARD_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("UNKNOWN").astype(str)
        unique_vals = df[col].unique()
        # Save options as dict to remain compatible with dashboard logic
        label_encoders[col] = {val: 1 for val in unique_vals}
        
df = pd.get_dummies(df, columns=[c for c in LOW_CARD_COLS if c in df.columns], drop_first=False)
# Convert boolean to int
for col in df.columns:
    if col.startswith(tuple(c + "_" for c in LOW_CARD_COLS)):
        df[col] = df[col].astype(int)
print(f"  [OK] One-Hot Encoded: {LOW_CARD_COLS}")

# 2. Frequency Encoding for High Cardinality Features
for col in HIGH_CARD_COLS:
    if col not in df.columns:
        continue
    df[col] = df[col].fillna("UNKNOWN").astype(str)
    freq_map = df[col].value_counts().to_dict()
    df[col + "_freq"] = df[col].map(freq_map)
    label_encoders[col] = freq_map
    df.drop(columns=[col], inplace=True)
    print(f"  [OK] Frequency Encoded: {col}")

print(f"\n  Shape after preprocessing: {df.shape}")


# =============================================================================
# SECTION 4 -- FEATURE SELECTION
# =============================================================================
print("\n[3/9] Selecting features ...")

# Keep only numeric columns that make sense for anomaly detection
EXCLUDE_FROM_FEATURES = [
    # These are identifiers / target-like -- keep for output but not for model
]

numeric_df = df.select_dtypes(include=[np.number])

# Remove near-zero variance features
selector = VarianceThreshold(threshold=0.01)
selector.fit(numeric_df)
selected_columns = numeric_df.columns[selector.get_support()].tolist()
features_df = numeric_df[selected_columns]

print(f"  Features kept ({len(selected_columns)}):")
for i, c in enumerate(selected_columns):
    print(f"    [{i:02d}] {c}")


# =============================================================================
# SECTION 5 -- STANDARDIZATION
# =============================================================================
print("\n[4/9] Scaling (StandardScaler) ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)
print("  [OK] Mean ~ 0, Std ~ 1")


# =============================================================================
# SECTION 6 -- ISOLATION FOREST
# =============================================================================
print(f"\n[5/9] Training Isolation Forest ...")
print(f"  n_estimators  = {N_ESTIMATORS}")
print(f"  contamination = {CONTAMINATION}")
print(f"  max_samples   = 'auto'")
print(f"  random_state  = 42")

iso_forest = IsolationForest(
    n_estimators  = N_ESTIMATORS,
    contamination = CONTAMINATION,
    max_samples   = "auto",
    max_features  = 1.0,
    bootstrap     = False,
    random_state  = 42
)
iso_forest.fit(X_scaled)
print("  [OK] Training complete!")

iso_predictions = iso_forest.predict(X_scaled)    # 1=normal, -1=anomaly
iso_scores      = iso_forest.decision_function(X_scaled)  # lower = more anomalous

n_anomalies = (iso_predictions == -1).sum()
n_normal    = (iso_predictions ==  1).sum()

print(f"\n[6/9] Isolation Forest results:")
print(f"  Normal transactions  ( 1): {n_normal:,}")
print(f"  Anomalous transactions (-1): {n_anomalies:,}")
print(f"  Anomaly rate: {n_anomalies/len(iso_predictions)*100:.2f}%")


# =============================================================================
# SECTION 7 -- BONUS: MAHALANOBIS DISTANCE METHOD
# =============================================================================
print("\n[7/9] Distance-Based detection (Mahalanobis) ...")

try:
    cov_matrix = np.cov(X_scaled.T) + np.eye(X_scaled.shape[1]) * 1e-6
    inv_cov    = np.linalg.inv(cov_matrix)
    mean_vec   = X_scaled.mean(axis=0)

    maha_distances = np.array([
        mahalanobis(row, mean_vec, inv_cov) for row in X_scaled
    ])

    if CONTAMINATION == "auto":
        threshold_maha = np.percentile(maha_distances, 95) # Fallback assumption for distance
    else:
        threshold_maha = np.percentile(maha_distances, 100 * (1 - CONTAMINATION))
        
    dist_predictions = np.where(maha_distances > threshold_maha, -1, 1)

    n_dist_anomalies = (dist_predictions == -1).sum()
    print(f"  Normal  ( 1): {(dist_predictions==1).sum():,}")
    print(f"  Anomaly (-1): {n_dist_anomalies:,}")
    print(f"  Threshold  : {threshold_maha:.4f}")

except np.linalg.LinAlgError:
    print("  [!] Singular matrix -- using Z-score fallback")
    max_z = np.abs(stats.zscore(X_scaled)).max(axis=1)
    if CONTAMINATION == "auto":
        z_thresh = np.percentile(max_z, 95)
    else:
        z_thresh = np.percentile(max_z, 100 * (1 - CONTAMINATION))
    dist_predictions = np.where(max_z > z_thresh, -1, 1)
    maha_distances   = max_z
    threshold_maha   = z_thresh

agreement = np.mean(iso_predictions == dist_predictions) * 100
print(f"\n  Method agreement: {agreement:.1f}%")

ct = pd.crosstab(
    pd.Series(iso_predictions, name="IF"),
    pd.Series(dist_predictions, name="Distance")
)
print(f"\n  Cross-Tab:\n{ct}\n")


# =============================================================================
# SECTION 8 -- ASSEMBLE RESULTS
# =============================================================================
print("[8/9] Assembling results ...")

df_raw["anomaly_IF"]           = iso_predictions
df_raw["anomaly_score_IF"]     = np.round(iso_scores, 5)
df_raw["anomaly_Distance"]     = dist_predictions
df_raw["mahalanobis_distance"] = np.round(maha_distances, 4)
df_raw["status_IF"]            = df_raw["anomaly_IF"].map({1: "Normal", -1: "ANOMALY"})
df_raw["status_Distance"]      = df_raw["anomaly_Distance"].map({1: "Normal", -1: "ANOMALY"})
df_raw["high_confidence_fraud"] = (
    (df_raw["anomaly_IF"] == -1) & (df_raw["anomaly_Distance"] == -1)
).astype(int)
df_raw["log_MNT"]              = np.log1p(df_raw["MNT"])
df_raw["log_CV_EN_TND"]        = np.log1p(df_raw["CV EN TND"])
df_raw["high_risk_country"]    = df_raw["52A_ISO_PAYS"].isin(HIGH_RISK_COUNTRIES).astype(int) \
                                  if "52A_ISO_PAYS" in df_raw.columns else 0

# ── Copy engineered features to output for compliance review ──────────────────
ENGINEERED_FEATURES = [
    "txn_count_sender_7d", "txn_count_country_7d",
    "txn_count_customer_7d",
    "amount_vs_sender_avg", "amount_vs_sender_std", "amount_vs_country_avg",
    "amount_vs_customer_avg", "amount_vs_customer_std", "customer_sender_diversity",
    "country_mismatch_risk", "sender_country_diversity", "rare_currency_for_country",
    "is_round_amount", "amount_rank_pct",
]
for feat in ENGINEERED_FEATURES:
    if feat in df.columns:
        df_raw[feat] = df[feat].values
print(f"  [OK] Copied {sum(1 for f in ENGINEERED_FEATURES if f in df.columns)} engineered features to output")

# Note: ETAT_RPA is NOT a fraud label. It is the RPA processing status.
# We do NOT use it as ground truth for evaluation.
if "ETAT_RPA" in df_raw.columns:
    print(f"  ETAT_RPA values: {df_raw['ETAT_RPA'].value_counts().to_dict()}")

n_high_conf = df_raw["high_confidence_fraud"].sum()
print(f"  High-confidence (IF+Dist): {n_high_conf:,}")

# ── UNSUPERVISED MODEL VALIDATION (no fraud labels needed) ────────────────────
print("\n" + "=" * 70)
print("   MODEL VALIDATION — Unsupervised Approach")
print("=" * 70)

anomalies = df_raw[df_raw["anomaly_IF"] == -1]
normales  = df_raw[df_raw["anomaly_IF"] ==  1]

print("\n  [A] STATISTICAL PROFILE COMPARISON")
print("  " + "-" * 50)
for col, label in [("CV EN TND", "Montant moyen (TND)"),
                    ("MNT", "Montant original moyen")]:
    if col in df_raw.columns:
        m_n = normales[col].mean()
        m_a = anomalies[col].mean()
        ratio = m_a / m_n if m_n > 0 else 0
        print(f"  {label}:")
        print(f"    Normales  : {m_n:>12,.2f}")
        print(f"    Anomalies : {m_a:>12,.2f}  (x{ratio:.1f})")

if "high_risk_country" in df_raw.columns:
    hr_n = normales["high_risk_country"].mean() * 100
    hr_a = anomalies["high_risk_country"].mean() * 100
    print(f"  Pays haut risque (FATF):")
    print(f"    Normales  : {hr_n:>8.1f}%")
    print(f"    Anomalies : {hr_a:>8.1f}%")

if "is_round_amount" in df_raw.columns:
    rn = normales["is_round_amount"].mean() * 100
    ra = anomalies["is_round_amount"].mean() * 100
    print(f"  Montants ronds:")
    print(f"    Normales  : {rn:>8.1f}%")
    print(f"    Anomalies : {ra:>8.1f}%")

if "country_mismatch_risk" in df_raw.columns:
    cm_n = normales["country_mismatch_risk"].mean() * 100
    cm_a = anomalies["country_mismatch_risk"].mean() * 100
    print(f"  BIC/Pays mismatch:")
    print(f"    Normales  : {cm_n:>8.1f}%")
    print(f"    Anomalies : {cm_a:>8.1f}%")

print("\n  [B] CONTAMINATION SENSITIVITY ANALYSIS")
print("  " + "-" * 50)
for cont in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, "auto"]:
    iso_test = IsolationForest(n_estimators=N_ESTIMATORS, contamination=cont, random_state=42)
    preds_t = iso_test.fit_predict(X_scaled)
    n_anom_t = (preds_t == -1).sum()
    marker = " ◄ SELECTED" if cont == CONTAMINATION else ""
    rate_str = f"{cont:.0%}" if isinstance(cont, float) else str(cont)
    print(f"  contamination={rate_str} → {n_anom_t:>5,} anomalies ({n_anom_t/len(preds_t)*100:.1f}%){marker}")

print("\n  [C] METHOD AGREEMENT")
print("  " + "-" * 50)
print(f"  IF + Mahalanobis agree on : {agreement:.1f}% of transactions")
print(f"  High confidence (both)    : {n_high_conf:,} transactions")
only_IF   = ((iso_predictions == -1) & (dist_predictions == 1)).sum()
only_dist = ((iso_predictions == 1) & (dist_predictions == -1)).sum()
print(f"  Only IF flagged           : {only_IF:,}")
print(f"  Only Mahalanobis flagged  : {only_dist:,}")

# =============================================================================
# EVALUATION METRICS -- Saved for dashboard display
# =============================================================================
print("\n" + "=" * 70)
print("   COMPUTING EVALUATION METRICS")
print("=" * 70)

labels_binary = (iso_predictions == -1).astype(int)
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), min(3000, len(X_scaled)), replace=False)

sil_score = silhouette_score(X_scaled[sample_idx], labels_binary[sample_idx])
db_score  = davies_bouldin_score(X_scaled[sample_idx], labels_binary[sample_idx])
ch_score  = calinski_harabasz_score(X_scaled[sample_idx], labels_binary[sample_idx])
print(f"  Silhouette={sil_score:.4f}  Davies-Bouldin={db_score:.4f}  Calinski-Harabasz={ch_score:.2f}")

anom_sc = iso_scores[iso_predictions == -1]
norm_sc = iso_scores[iso_predictions == 1]
score_sep = abs(float(anom_sc.mean()) - float(norm_sc.mean()))

contam_results = {}
for cont in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
    iso_t = IsolationForest(n_estimators=N_ESTIMATORS, contamination=cont, random_state=42)
    preds_t = iso_t.fit_predict(X_scaled)
    contam_results[str(cont)] = int((preds_t == -1).sum())

stability_runs = []
for seed in [42, 123, 0, 7, 99]:
    m = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed)
    p = m.fit_predict(X_scaled)
    stability_runs.append(int((p == -1).sum()))
stability_cv = float(np.std(stability_runs) / np.mean(stability_runs) * 100)
print(f"  Stability CV={stability_cv:.2f}%")

mann_whitney = {}
for col in ["CV EN TND", "MNT"]:
    if col in df_raw.columns:
        a_v = df_raw[df_raw["anomaly_IF"] == -1][col].dropna()
        n_v = df_raw[df_raw["anomaly_IF"] ==  1][col].dropna()
        u, p_val = stats.mannwhitneyu(a_v, n_v, alternative="two-sided")
        mann_whitney[col] = {"u": float(u), "p": float(p_val),
                             "anom_mean": float(a_v.mean()), "norm_mean": float(n_v.mean())}

roc_proxy = pr_proxy = None
if "ETAT_RPA" in df_raw.columns:
    try:
        ko = df_raw["ETAT_RPA"].astype(str).str.startswith("KO").astype(int)
        roc_proxy = float(roc_auc_score(ko, -iso_scores))
        pr_proxy  = float(average_precision_score(ko, -iso_scores))
        print(f"  ROC-AUC(proxy)={roc_proxy:.4f}  PR-AUC(proxy)={pr_proxy:.4f}")
    except Exception:
        pass

evaluation_metrics = {
    "silhouette": float(sil_score), "davies_bouldin": float(db_score),
    "calinski_harabasz": float(ch_score), "score_separation": score_sep,
    "anom_score_mean": float(anom_sc.mean()), "anom_score_std": float(anom_sc.std()),
    "norm_score_mean": float(norm_sc.mean()), "norm_score_std": float(norm_sc.std()),
    "contamination_sensitivity": contam_results, "stability_runs": stability_runs,
    "stability_cv": stability_cv, "method_agreement": float(agreement),
    "n_high_confidence": int(n_high_conf), "only_if": int(only_IF),
    "only_dist": int(only_dist), "mann_whitney": mann_whitney,
    "roc_auc_proxy": roc_proxy, "pr_auc_proxy": pr_proxy,
    "n_total": len(df_raw), "n_anomalies": int(n_anomalies),
    "n_features": len(selected_columns), "contamination": str(CONTAMINATION),
}
print("  [OK] All evaluation metrics computed")

# ── SHAP Values (Model Explainability) ────────────────────────────────────────
try:
    import shap
    print("\n  Computing SHAP values for model explainability ...")
    explainer = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X_scaled)
    
    # Generate Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features_df, show=False)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/08_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {PLOTS_DIR}/08_shap_summary.png")
except ImportError:
    print("\n  [!] 'shap' library not found. Skipping SHAP plots. Run `pip install shap`.")
except Exception as e:
    print(f"\n  [!] Could not generate SHAP values: {e}")


# =============================================================================
# SECTION 9 -- VISUALISATIONS
# =============================================================================
print("\n[9/9] Generating plots ...")

# ── PCA 2D scatter ────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_.sum() * 100

colors_IF   = ["#e74c3c" if p == -1 else "#2ecc71" for p in iso_predictions]
colors_DIST = ["#e74c3c" if p == -1 else "#3498db" for p in dist_predictions]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"SWIFT Transactions — Anomaly Detection\n(PCA: {explained:.1f}% variance)",
             fontsize=14, fontweight="bold")

for ax, colors, title, n_anom in [
    (axes[0], colors_IF,   "Isolation Forest",         n_anomalies),
    (axes[1], colors_DIST, "Mahalanobis Distance", n_dist_anomalies),
]:
    ax.scatter(X_pca[:,0], X_pca[:,1], c=colors, alpha=0.5, s=12, edgecolors="none")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("PCA Component 1"); ax.set_ylabel("PCA Component 2")
    ax.legend(handles=[
        mpatches.Patch(color="#2ecc71", label=f"Normal ({len(iso_predictions)-n_anom:,})"),
        mpatches.Patch(color="#e74c3c", label=f"Anomaly ({n_anom:,})"),
    ], fontsize=9)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_scatter_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK] {PLOTS_DIR}/01_scatter_comparison.png")

# ── Anomaly score distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
sorted_scores  = np.sort(iso_scores)
bar_colors = ["#e74c3c" if s < 0 else "#2ecc71" for s in sorted_scores]
ax.bar(range(len(sorted_scores)), sorted_scores, color=bar_colors, width=1.0, alpha=0.8)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="Decision boundary (0)")
ax.set_title("Isolation Forest — Anomaly Scores (sorted)", fontsize=13, fontweight="bold")
ax.set_xlabel("Transactions"); ax.set_ylabel("Score (lower = more suspicious)")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_anomaly_scores.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK] {PLOTS_DIR}/02_anomaly_scores.png")

# ── Amount distribution — Normal vs Anomaly ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in [
    (axes[0], df_raw["log_MNT"],        "log(MNT)"),
    (axes[1], df_raw["log_CV_EN_TND"],  "log(CV EN TND)"),
]:
    ax.hist(col[iso_predictions == 1],  bins=60, alpha=0.6, color="#2ecc71",
            label="Normal",  density=True)
    ax.hist(col[iso_predictions == -1], bins=60, alpha=0.75, color="#e74c3c",
            label="Anomaly", density=True)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Log Amount"); ax.set_ylabel("Density")
    ax.legend(fontsize=9)
fig.suptitle("Amount Distributions: Normal vs Anomaly", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_amount_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK] {PLOTS_DIR}/03_amount_distributions.png")

# ── Anomalies by Country ───────────────────────────────────────────────────────
if "Nom_pays" in df_raw.columns:
    country_col = "Nom_pays"
elif "52A_ISO_PAYS" in df_raw.columns:
    country_col = "52A_ISO_PAYS"
else:
    country_col = None

if country_col:
    anom_by_country = df_raw[df_raw["anomaly_IF"] == -1][country_col].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(13, 6))
    anom_by_country.plot(kind="barh", ax=ax, color="#e74c3c", alpha=0.85)
    ax.set_title("Top 15 Countries — Anomalous Transactions (Isolation Forest)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Count"); ax.set_ylabel("Country")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_anomalies_by_country.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {PLOTS_DIR}/04_anomalies_by_country.png")

# ── Anomalies by FRAIS type ────────────────────────────────────────────────────
if "FRAIS" in df_raw.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, title, mask in [
        (axes[0], "All Transactions",   slice(None)),
        (axes[1], "Anomalies Only",     df_raw["anomaly_IF"] == -1),
    ]:
        df_raw.loc[mask, "FRAIS"].value_counts().plot(
            kind="bar", ax=ax, color=["#3498db", "#e74c3c", "#2ecc71"], alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fee Type (FRAIS)"); ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("FRAIS Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_frais_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {PLOTS_DIR}/05_frais_distribution.png")

# ── Anomaly Profile Comparison (for jury presentation) ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
profile_features = []
normal_vals = []
anomaly_vals = []

for feat, label in [("CV EN TND", "Montant (TND)"),
                     ("high_risk_country", "Pays haut risque"),
                     ("is_round_amount", "Montant rond"),
                     ("country_mismatch_risk", "BIC/Pays mismatch")]:
    if feat in df_raw.columns:
        profile_features.append(label)
        normal_vals.append(normales[feat].mean())
        anomaly_vals.append(anomalies[feat].mean())

if profile_features:
    x = np.arange(len(profile_features))
    width = 0.35
    ax.bar(x - width/2, normal_vals, width, label="Normal", color="#2ecc71", alpha=0.85)
    ax.bar(x + width/2, anomaly_vals, width, label="Anomaly", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(profile_features, fontsize=10)
    ax.set_title("Anomaly vs Normal — Feature Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Value")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_anomaly_profile.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {PLOTS_DIR}/06_anomaly_profile.png")

# ── Mahalanobis distance histogram ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(maha_distances[dist_predictions ==  1], bins=60, alpha=0.7,
        color="#3498db", label="Normal", density=True)
ax.hist(maha_distances[dist_predictions == -1], bins=60, alpha=0.8,
        color="#e74c3c", label="Anomaly", density=True)
ax.axvline(threshold_maha, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold = {threshold_maha:.2f}")
ax.set_title("Mahalanobis Distance Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Distance"); ax.set_ylabel("Density"); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_mahalanobis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  [OK] {PLOTS_DIR}/07_mahalanobis.png")


# =============================================================================
# EXPORT TO EXCEL (4 sheets)
# =============================================================================
print(f"\n  Exporting -> '{OUTPUT_FILE}' ...")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_raw.to_excel(writer,  sheet_name="All Transactions", index=False)

    (df_raw[df_raw["anomaly_IF"] == -1]
        .sort_values("anomaly_score_IF")
        .to_excel(writer, sheet_name="IF Anomalies", index=False))

    (df_raw[df_raw["high_confidence_fraud"] == 1]
        .to_excel(writer, sheet_name="High Confidence Fraud", index=False))

    summary_rows = [
        ("Total Transactions",                 len(df_raw)),
        ("IF -- Normal",                       n_normal),
        ("IF -- Anomaly",                      n_anomalies),
        ("IF -- Anomaly Rate (%)",             round(n_anomalies/len(df_raw)*100, 2)),
        ("Distance -- Anomaly",                n_dist_anomalies),
        ("High Confidence (both methods)",      n_high_conf),
        ("Method Agreement (%)",               round(agreement, 1)),
        ("Only IF flagged",                    int(only_IF)),
        ("Only Mahalanobis flagged",           int(only_dist)),
    ]

    pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(
        writer, sheet_name="Summary", index=False)

print(f"  [OK] Saved: {OUTPUT_FILE}")


# =============================================================================
# PRINT TOP 10 MOST SUSPICIOUS
# =============================================================================
print("\n" + "=" * 70)
print("   TOP 10 MOST SUSPICIOUS TRANSACTIONS")
print("=" * 70)
top10_cols = ["CHRONO","Date","MNT","DEV","CV EN TND","Nom_pays","52A_ISO_PAYS","FRAIS",
              "ETAT_RPA","anomaly_score_IF","mahalanobis_distance","high_confidence_fraud"]
top10_cols = [c for c in top10_cols if c in df_raw.columns]
top10 = (df_raw[df_raw["anomaly_IF"] == -1]
         .nsmallest(10, "anomaly_score_IF")
         [top10_cols])
print(top10.to_string())


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("   RESULTS SUMMARY")
print("=" * 70)
print(f"  Dataset          : {len(df_raw):,} SWIFT transactions")
print(f"  Features used    : {len(selected_columns)}")
print(f"  IF  Anomalies    : {n_anomalies:,}  ({n_anomalies/len(df_raw)*100:.2f}%)")
print(f"  Dist Anomalies   : {n_dist_anomalies:,}  ({n_dist_anomalies/len(df_raw)*100:.2f}%)")
print(f"  High Confidence  : {n_high_conf:,}")
print(f"  Method Agreement : {agreement:.1f}%")
print(f"  Output Excel     : {OUTPUT_FILE}")
print(f"  Plots folder     : {PLOTS_DIR}/")
print(f"\n  Dashboard        : run -->  streamlit run dashboard.py")

print("""
+------------------------------------------------------------------+
|   HOW TO INTERPRET RESULTS (SWIFT / AML Banking Context)        |
+------------------------------------------------------------------+
|                                                                  |
|  anomaly_IF = 1   --> NORMAL transfer                            |
|     Behaves like majority of SWIFT transfers. No concern.        |
|                                                                  |
|  anomaly_IF = -1  --> [!] SUSPICIOUS TRANSFER                    |
|     Statistically unusual. Could indicate:                       |
|     - Unusually large or tiny amount (MNT >> or << average)      |
|     - Originating from a high-risk / sanctioned country           |
|     - Unusual fee type for the account type                       |
|     - Very long or very short processing delay                    |
|     - Unusual currency for this client segment                   |
|     -> Flag for AML / compliance team review                      |
|                                                                  |
|  anomaly_score_IF (lower = more suspicious):                     |
|     < -0.10  = Highly suspicious                                 |
|     -0.05 to -0.10 = Moderately suspicious                       |
|     > 0      = Clearly normal                                     |
|                                                                  |
|  high_confidence_fraud = 1:                                      |
|     Flagged by BOTH Isolation Forest AND Mahalanobis             |
|     -> Highest priority for compliance review                     |
|                                                                  |
|  Compare with IF results to evaluate model usefulness.           |
+------------------------------------------------------------------+
""")

# =============================================================================
# EXPORT MODEL FOR DASHBOARD SIMULATOR
# =============================================================================
print("\n[10/10] Saving model for dashboard simulator ...")
model_artifacts = {
    "iso_forest": iso_forest,
    "scaler": scaler,
    "selector": selector,
    "label_encoders": label_encoders,
    "features_df_columns": selected_columns,
    "evaluation_metrics": evaluation_metrics,
}
joblib.dump(model_artifacts, "fraud_model.pkl")
print("  [OK] Saved: fraud_model.pkl")
