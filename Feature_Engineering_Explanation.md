# Feature Engineering — Banking Fraud Detection
## PFE Final Year Project — UIB Tunisia

---

## What is Feature Engineering?

Feature Engineering is the process of **creating new, smarter variables** from raw data
to help the machine learning model detect patterns that are impossible to see from
the raw columns alone.

Think of it like this:
- **Raw data** = a security camera showing individual frames
- **Engineered features** = an investigator's report saying "this person visited 15 times
  in 3 days, each time carrying a different passport"

The AI (Isolation Forest) can only work with numbers. Feature engineering transforms
raw banking data into meaningful numerical signals that make fraud **stand out**.

---

## Feature Categories

### 1. 🚀 Velocity Features (Speed / Frequency)

These features detect **sudden bursts of activity**. Fraudsters often act quickly,
making many transfers in a short period before the bank notices.

| Feature | Formula | What It Detects |
|---------|---------|-----------------|
|`txn_count_sender_7d` | Count of transactions from same sender bank (`52A_EXPED`) in rolling 7-day window | Sender that suddenly becomes very active |
| `txn_count_country_7d` | Count of transactions from same country (`52A_ISO_PAYS`) in rolling 7-day window | Unusual volume from a specific country |
| `txn_count_account_7d` | Count of transactions for same account type (`Libellé_cpt_`) in rolling 7-day window | Account type with abnormal spike |

**Real Example:**
- Bank `BNPAFRPP` (BNP Paribas France) normally sends 2 transfers/week to UIB.
- Suddenly, 15 transfers arrive in 3 days from `BNPAFRPP`.
- `txn_count_sender_7d = 15` → The AI flags this as highly unusual.

---

### 2. 📊 Behavioral Ratio Features (Breaking the Habit)

These features compare each transaction to **what is normal for that specific sender**.
A 50,000 TND transfer might be normal for a large corporation but extremely suspicious
for a small personal account.

| Feature | Formula | What It Detects |
|---------|---------|-----------------|
| `amount_vs_sender_avg` | `Transaction Amount ÷ Sender's Historical Average` | Amount wildly different from sender's habit |
| `amount_vs_sender_std` | `(Amount - Sender Mean) ÷ Sender Std Dev` (Z-score) | Statistical outlier for this sender |
| `amount_vs_country_avg` | `Transaction Amount ÷ Country's Historical Average` | Amount unusual for this country's pattern |

**Real Example:**
- Sender `BNPAFRPP` usually sends an average of 5,000 TND.
- Today they send 150,000 TND.
- `amount_vs_sender_avg = 150,000 / 5,000 = 30.0` → 30x the normal amount!
- Meanwhile, sender `DEUTDEFF` (Deutsche Bank) always sends ~150,000 TND.
- Their ratio would be `150,000 / 150,000 = 1.0` → perfectly normal.

**Why this is powerful:** The AI doesn't just flag "large amounts" blindly. It flags
amounts that are **unusual for THAT specific sender**.

---

### 3. 🗺️ Geographical Mismatch Features

These features detect **money laundering through intermediary countries**.
Criminals route money through "clean" countries to hide its true origin.

| Feature | Formula | What It Detects |
|---------|---------|-----------------|
| `country_mismatch_risk` | `1` if BIC country code ≠ declared country (`52A_ISO_PAYS`) | Money routed through intermediary |
| `sender_country_diversity` | Count of unique countries per sender BIC | Shell company operating from many countries |
| `rare_currency_for_country` | `1` if currency is not the most common for that country | Unusual currency for origin (e.g., Iranian Rial from France) |

**Real Example — The Shell Company Hop:**
1. A criminal in **Syria** (sanctioned country) wants to send money to Tunisia.
2. Sending directly from Syria → Tunisia would trigger instant alerts.
3. Instead, they route through a shell company in **France**.
4. The SWIFT message shows: `52A_EXPED = SYRIFRPP` (BIC says Syria "SY" at position 4-5,
   but wait — positions 4-5 say "FR", meaning the bank registered in France).
5. Actually, `52A_ISO_PAYS = SY` (declared origin is Syria).
6. `country_mismatch_risk = 1` if BIC country ≠ declared country.

**How BIC Country Extraction Works:**
- BIC code: `B N P A F R P P`
- Positions:  1 2 3 4 **5 6** 7 8
- Country = positions 5-6 = `FR` (France)

---

### 4. 💰 Amount Anomaly Features

These features detect **suspicious amount patterns** without comparing to history.

| Feature | Formula | What It Detects |
|---------|---------|-----------------|
| `is_round_amount` | `1` if amount is a perfect multiple of 1,000 | Manual/artificial transfers (laundering) |
| `amount_rank_pct` | Percentile rank of amount (0-100) | Relative position in dataset |

**Real Example:**
- Legitimate business payments: 12,847.63 TND, 3,291.50 TND (natural, irregular amounts)
- Money laundering transfers: 10,000.00 TND, 50,000.00 TND (perfectly round)
- `is_round_amount = 1` for the round amounts → flagged as suspicious

---

## Summary of All Engineered Features

| # | Feature | Category | Input Columns Used |
|---|---------|----------|-------------------|
| 1 | `txn_count_sender_7d` | Velocity | `52A_EXPED`, `Date`, `MNT` |
| 2 | `txn_count_country_7d` | Velocity | `52A_ISO_PAYS`, `Date`, `MNT` |
| 3 | `txn_count_account_7d` | Velocity | `Libellé_cpt_`, `Date`, `MNT` |
| 4 | `amount_vs_sender_avg` | Behavioral | `52A_EXPED`, `CV EN TND` |
| 5 | `amount_vs_sender_std` | Behavioral | `52A_EXPED`, `CV EN TND` |
| 6 | `amount_vs_country_avg` | Behavioral | `52A_ISO_PAYS`, `CV EN TND` |
| 7 | `country_mismatch_risk` | Geography | `52A_EXPED`, `52A_ISO_PAYS` |
| 8 | `sender_country_diversity` | Geography | `52A_EXPED`, `52A_ISO_PAYS` |
| 9 | `rare_currency_for_country` | Geography | `DEV`, `52A_ISO_PAYS` |
| 10 | `is_round_amount` | Amount | `MNT` |
| 11 | `amount_rank_pct` | Amount | `CV EN TND` |

---

## Why Python and NOT Excel?

| Criteria | ❌ Excel | ✅ Python |
|----------|---------|----------|
| Raw data safety | Risk of corruption | `Data.xlsm` never modified |
| Automation | Manual formulas each time | `python main.py` → automatic |
| Performance | Freezes with 13K+ rows | Handles millions in seconds |
| Reproducibility | Hard to audit formulas | Code is transparent and shareable |
| Complexity | Rolling 7-day grouped averages nearly impossible | 3 lines of pandas code |

**All features are computed automatically in `main.py` every time the script runs.
The raw Excel file (`Data.xlsm`) is never modified.**
