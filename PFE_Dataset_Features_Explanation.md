# PFE — Dataset Features: Complete Explanation

> **Project**: Fraud Detection in Financial Transactions (PFE Final Year Project)
> **Dataset**: `Data.xlsm` → sheet `DETAIL` — **13,004 SWIFT wire transfers** (UIB Tunisia)
> **Model**: Isolation Forest (unsupervised) + Mahalanobis Distance (validation)

---

## Part 1 — Raw Features (Original Data)

These are the **original columns** present in the Excel file `Data.xlsm`, sheet `DETAIL`. They come directly from the bank's SWIFT transaction system — no transformation has been applied yet.

### 1.1 Transaction Identifiers

| Feature | Definition | Example | Role in Fraud Detection |
|---------|-----------|---------|------------------------|
| `CHRONO` | Unique serial number assigned to each transaction by the banking system | `20230145` | **Dropped** — It is an identifier, not a behavioral feature. It carries no fraud signal. |
| `FC___` | **Customer ID** (Code Client / Fiche Client). Anonymized using letter substitution for privacy. | `ABIHIDHB` | Tracks which customer initiated the transfer. Crucial for detecting customers with abnormal behavior patterns (e.g., sudden burst of transfers). |
| `AGE` | **Branch code** (Code Agence) — identifies which bank branch processed the transaction | `001`, `015` | **Dropped** — Categorical identifier that would add noise if treated as a number. |

### 1.2 Financial Amounts

| Feature | Definition | Example | Role in Fraud Detection |
|---------|-----------|---------|------------------------|
| `MNT` | Transaction amount in the **original currency** (EUR, USD, GBP, etc.) | `15,847.63 EUR` | Core fraud signal. Extremely large or extremely small amounts are suspicious. A transfer of 500,000 EUR stands out from a customer who normally sends 2,000 EUR. |
| `CV EN TND` | Amount **converted to Tunisian Dinar** (TND) using the exchange rate at the time of the transaction | `52,296.18 TND` | Enables **cross-currency comparison**. Without conversion, comparing 10,000 EUR vs 10,000 JPY would be misleading (they differ by ~100x in value). |

### 1.3 Transaction Metadata

| Feature | Definition | Example | Role in Fraud Detection |
|---------|-----------|---------|------------------------|
| `DEV` | **Currency code** (ISO 4217) of the transaction | `EUR`, `USD`, `GBP`, `CAD` | Unusual currencies for a given customer or country are suspicious. A Tunisian client suddenly transacting in Iranian Rial (IRR) raises red flags. |
| `FRAIS` | **Fee structure** — who pays the transfer fees | `OUR` (sender pays all), `SHA` (shared), `BEN` (receiver pays) | `BEN` means the sender wants to hide the true cost. In AML (Anti-Money Laundering) practice, `OUR` on very large cross-border transfers can indicate urgency to move money quickly. |
| `ETAT` | Transaction **state** | Always `TR` (Traitée = Processed) | **Dropped** — Every single row has the same value. Zero variance = zero information for the model. |
| `ETAT_RPA` | **RPA bot processing status** — records whether the automated compliance robot succeeded or failed | `VALIDE`, `KO_MONTANT`, `KO_DEVISE` | Kept for context but **NOT a fraud label**. It only says if the robot could process the transaction, not if the transaction is fraudulent. |

### 1.4 Dates

| Feature | Definition | Example | Role in Fraud Detection |
|---------|-----------|---------|------------------------|
| `Date` | **Transaction initiation date** — when the customer requested the transfer | `2024-03-15` | Transactions on weekends or holidays are unusual for SWIFT. Year-end spikes may indicate window-dressing. |
| `DATE_VL` | **Value date** — when the funds actually become available to the recipient | `2024-03-17` | A value date set *before* the transaction date (backdating) is a classic accounting manipulation technique. |
| `D_EXEC` | **Execution date** — when the bank actually processed and sent the SWIFT message | `2024-03-16` | Long delays between `Date` and `D_EXEC` may indicate compliance holds. Zero delay on a very large amount may mean checks were bypassed. |

### 1.5 Geographic & Banking Information

| Feature | Definition | Example | Role in Fraud Detection |
|---------|-----------|---------|------------------------|
| `52A_ISO_PAYS` | **ISO country code** of the sender (ordering) bank | `FR` (France), `IR` (Iran), `DE` (Germany) | Transactions from FATF-blacklisted countries (Iran, North Korea, Syria) carry inherently higher risk. |
| `Nom_pays` | **Country name** in full text (redundant with ISO code) | `France`, `Allemagne` | **Dropped** — Redundant with `52A_ISO_PAYS`. Keeping both would double-count the same information. |
| `52A_EXPED` | **Correspondent bank BIC code** — identifies the sending bank in the SWIFT network | `BNPAFRPP` (BNP Paribas, France) | Allows tracking sender bank behavior. A small unknown bank sending millions is more suspicious than Deutsche Bank doing the same. |
| `chps_52A` | **Additional SWIFT field 52A data** — extended sender information | Various SWIFT codes | Provides additional sender identification context. |
| `chps_57` | **Receiving bank BIC** — the bank that receives the SWIFT message | Almost always `UIBKTNTTXXX` | **Dropped** — Nearly constant value (all transactions go to UIB Tunisia). Near-zero variance. |

### 1.6 Other Columns

| Feature | Definition | Example | Role in Fraud Detection |
|---------|-----------|---------|------------------------|
| `Libellé_cpt_` | **Account type label** — describes the type of bank account | `CPD` (Compte à Préavis en Devise), `CED`, `CDT PP` | Different account types have different expected behaviors. A savings account suddenly doing 50 international transfers is unusual. |
| `INFO_Champs_70` | **Free-text payment reference** — describes the purpose of the transfer | `INVOICE 2024-0892 PAYMENT` | **Dropped** — Free text cannot be directly processed by ML algorithms without NLP preprocessing. |
| `chps_72` | **Reference codes** — additional identifiers (SSN, invoice refs) | Various codes, 54% missing | **Dropped** — Too sparse (54% missing) to be useful. |
| `ASSOC` | **Association field** — linked transaction reference | Mostly empty | **Dropped** — 98.7% missing. Cannot be meaningfully imputed. |

---

## Part 2 — Encoded Features

### 2.1 Why Is Encoding Necessary?

Machine learning algorithms work with **numbers**, not text. The dataset contains 8 categorical (text) columns that must be converted to numerical values. Without encoding, the model simply cannot process them.

### 2.2 Encoding Method: LabelEncoder

`LabelEncoder` assigns a unique integer to each unique text value. It is used here because Isolation Forest (tree-based) handles ordinal integers well.

### 2.3 All Encoded Features

#### `DEV` → `DEV_enc` (Currency)

| Before (Text) | After (Integer) |
|---------------|----------------|
| `EUR` | `0` |
| `GBP` | `1` |
| `USD` | `2` |
| `CAD` | `3` |

> **Why it helps**: The model can now learn that certain currency codes correlate with anomalies. For example, if currency `7` (say, IRR) appears almost exclusively in flagged transactions, the model captures this pattern.

#### `FRAIS` → `FRAIS_enc` (Fee Type)

| Before | After |
|--------|-------|
| `BEN` | `0` |
| `OUR` | `1` |
| `SHA` | `2` |

> **Why it helps**: Certain fee structures are more common in legitimate vs suspicious transactions. `OUR` on massive transfers can indicate urgency.

#### `52A_ISO_PAYS` → `52A_ISO_PAYS_enc` (Sender Country)

| Before | After |
|--------|-------|
| `FR` | `5` |
| `DE` | `2` |
| `IR` | `8` |
| `US` | `15` |

> **Why it helps**: Country patterns become learnable. The model discovers that transactions from certain country codes tend to cluster with anomalies.

#### `FC___` → `FC____enc` (Customer ID)

| Before | After |
|--------|-------|
| `ABIHIDHB` | `0` |
| `ABAHBBJF` | `1` |
| ... | ... |

> **Why it helps**: Allows the model to learn per-customer patterns. A customer who suddenly appears in unusual clusters gets flagged.

#### Other Encoded Features

| Original Column | Encoded Column | What It Represents |
|----------------|---------------|-------------------|
| `Libellé_cpt_` | `Libellé_cpt__enc` | Account type |
| `52A_EXPED` | `52A_EXPED_enc` | Sender bank BIC |
| `chps_52A` | `chps_52A_enc` | SWIFT field data |
| `ETAT_RPA` | `ETAT_RPA_enc` | RPA processing status |

### 2.4 How Encoding Improves Model Performance

| Without Encoding | With Encoding |
|-----------------|--------------|
| Model cannot read `"EUR"`, `"BNPAFRPP"` | Model reads `0`, `5`, `12` — computable values |
| Categorical features are **completely ignored** | All 8 categorical features become **active inputs** |
| Model uses only 2 raw numeric features | Model uses **39 total features** including encoded ones |

> [!IMPORTANT]
> After encoding, the **original text column is dropped** to avoid duplication. For example, `DEV` is removed and only `DEV_enc` remains.

---

## Part 3 — Engineered Features

These are **new columns created by the Python script** from combinations and transformations of the original data. They capture patterns that raw data alone cannot express.

### 3.1 Missing-Value Flags

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `MNT_missing` | `1` if `MNT` was NaN, else `0` | A transaction with no amount recorded → `MNT_missing = 1` | In SWIFT transfers, a missing amount is a **compliance red flag**. It suggests the transaction was submitted through an irregular channel or was intentionally obscured. |
| `CV EN TND_missing` | `1` if `CV EN TND` was NaN, else `0` | No TND conversion available → `CV EN TND_missing = 1` | Missing conversion could indicate an unrecognized currency or manual override of the system. |

> **Key insight**: These flags are created **before** missing values are filled (imputed). This preserves the signal that "something was missing" even after imputation fills in a median value.

---

### 3.2 Date/Time Features

Extracted from the three date columns (`Date`, `DATE_VL`, `D_EXEC`):

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `txn_month` | `Date.month` (1–12) | March → `3` | Detects **seasonal patterns**. Year-end (December) often shows spikes in fraudulent transfers trying to close before audits. |
| `txn_day` | `Date.day` (1–31) | 15th of the month → `15` | End-of-month transfers may indicate payroll fraud or deadline-driven laundering. |
| `txn_weekday` | `Date.weekday` (0=Monday, 6=Sunday) | Wednesday → `2` | SWIFT transfers on **weekends** (5 or 6) are unusual and suspicious. |
| `txn_weekend` | `1` if Saturday or Sunday, else `0` | Saturday → `1` | Binary flag for weekend transactions. |
| `vl_month`, `vl_day`, `vl_weekday` | Same formulas from `DATE_VL` | — | Value date patterns may reveal backdating manipulation. |
| `exec_month`, `exec_day`, `exec_weekday` | Same formulas from `D_EXEC` | — | Execution date patterns reveal processing anomalies. |

#### Delay Features

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `processing_delay_days` | `(D_EXEC − Date).days` | Transaction on March 15, executed March 22 → `7 days` | **Long delay** (e.g., 15 days) = possible compliance hold. **Zero delay** on a huge amount = checks may have been bypassed. |
| `value_delay_days` | `(DATE_VL − Date).days` | Transaction March 15, value date March 13 → `−2 days` | **Negative values** (backdated) are a classic accounting manipulation technique used in money laundering. |

---

### 3.3 Velocity Features (Rolling 7-Day Windows)

These detect **sudden bursts of activity** — a hallmark of smurfing (splitting large amounts into many small transactions) and automated laundering.

| Feature | Grouped By | Formula | Example | Why Useful |
|---------|-----------|---------|---------|-----------|
| `txn_count_sender_7d` | `52A_EXPED` (sender bank) | Count of transactions from the same sender bank in the past 7 days | Bank BNPAFRPP normally sends 2 transfers/week. This week: 15 transfers → `txn_count_sender_7d = 15` | A sudden burst from one bank suggests coordinated transfer activity. |
| `txn_count_country_7d` | `52A_ISO_PAYS` (country) | Count of transactions from the same country in the past 7 days | Normally 5 transfers/week from France. This week: 40 → value = `40` | Unusual volume from a specific country may indicate a coordinated scheme. |
| `txn_count_account_7d` | `Libellé_cpt_` (account type) | Count per account type in 7 days | CPD accounts normally see 10/week. This week: 50 → value = `50` | Account type with unusual activity burst. |
| `txn_count_customer_7d` | `FC___` (customer) | Count per customer in 7 days | Customer ABIHIDHB normally does 1 transfer/month. This week: 10 → value = `10` | **Most powerful velocity feature.** A customer suddenly doing many transfers is a classic money laundering indicator. |

---

### 3.4 Risk Indicator Features

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `high_risk_country` | `1` if `52A_ISO_PAYS` ∈ {IR, KP, SY, YE, LY, SO, SD, MM, PK, TR, ...} (22 FATF-listed countries) | Transfer from Iran (IR) → `1`. Transfer from France (FR) → `0` | Encodes **real-world AML compliance knowledge**. FATF blacklisted countries are known havens for money laundering and terrorism financing. |
| `log_MNT` | `log(1 + MNT)` | MNT = 500,000 → `log_MNT = 13.12`. MNT = 100 → `log_MNT = 4.62` | **Compresses extreme outliers**. Without log, a 500,000 EUR transfer would dominate. With log, the scale becomes manageable (4 to 13) and the model treats all amounts more fairly. |
| `log_CV_EN_TND` | `log(1 + CV EN TND)` | Same logic for the TND-converted amount | Same benefit: prevents extreme values from distorting the model. |

---

### 3.5 Behavioral Ratio Features

These compare each transaction to the **historical average** of its group. This is critical because raw amounts alone are misleading — 500,000 TND from Deutsche Bank is normal, but 500,000 TND from a small regional bank is extremely suspicious.

#### Sender-Level (per bank)

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `amount_vs_sender_avg` | `CV EN TND / mean(all transfers from this sender)` | Bank BNPAFRPP average = 5,000 TND. This transfer = 150,000 TND → **ratio = 30x** | A ratio of 30x means this transfer is 30 times larger than what this bank normally sends. Highly suspicious. |
| `amount_vs_sender_std` | `(CV EN TND − sender_mean) / sender_std` (Z-score) | Z-score = +4.2 → this transfer is 4.2 standard deviations above the sender's average | Z-score > 3 is statistically extreme (less than 0.1% probability under normal behavior). |

#### Country-Level

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `amount_vs_country_avg` | `CV EN TND / mean(all transfers from this country)` | Average French transfer = 8,000 TND. This one = 80,000 TND → **ratio = 10x** | Detects transfers that are abnormally large relative to their country of origin. |

#### Customer-Level

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `amount_vs_customer_avg` | `CV EN TND / mean(customer's own history)` | Customer ABIHIDHB average = 3,000 TND. Today = 200,000 TND → **ratio = 66x** | The most personalized behavioral feature. Detects when a customer deviates dramatically from their own pattern. |
| `amount_vs_customer_std` | `(CV EN TND − customer_mean) / customer_std` | Z-score = +5.1 within this customer's history | Measures statistical deviation from the customer's own baseline — not the global average. |
| `customer_sender_diversity` | Count of unique sender BICs per customer | Normal customer uses 1–2 banks. Suspect uses 8 different banks → value = `8` | Money launderers use many different correspondent banks to obscure the money trail. Legitimate customers consistently use 1–2 banks. |

---

### 3.6 Geographic Mismatch Features

Designed to catch **money routing through intermediary countries** — a key money laundering technique.

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `country_mismatch_risk` | `1` if the country embedded in the BIC code ≠ declared `52A_ISO_PAYS` | BIC = `COMABORJ` (positions 5–6 = `BO`, Bolivia), but declared country = `FR` (France) → **mismatch = 1** | Indicates the money is being routed through an intermediary. The sender claims France, but the BIC belongs to Bolivia. |
| `sender_country_diversity` | Count of unique countries claimed by the same sender BIC | A legitimate bank always = `1`. A shell company claims 4 different countries → value = `4` | Shell companies and intermediaries often operate under multiple country identities. |
| `rare_currency_for_country` | `1` if the currency used ≠ the most common currency for that origin country | Transfer from France (`FR`) in Iranian Rial (`IRR`) instead of Euro (`EUR`) → `1` | Using an unusual currency for a given country suggests the transaction is not a normal trade payment. |

---

### 3.7 Amount Anomaly Features

| Feature | Formula | Example | Why Useful |
|---------|---------|---------|-----------|
| `is_round_amount` | `1` if `MNT % 1000 == 0` and `MNT > 0` | MNT = 50,000.00 → `1`. MNT = 12,847.63 → `0` | Real business invoices have irregular decimals (€12,847.63). Perfectly round amounts (€50,000.00) suggest **artificial or manually constructed transfers**, common in money laundering. |
| `amount_rank_pct` | Percentile rank of `CV EN TND` × 100 | The largest transfer = `100`. The median = `50`. A small transfer = `5` | Instead of raw amounts, the model sees "this is in the **top 1%**" — much easier to learn patterns from. Transforms a 0–5,000,000 range into a clean 0–100 scale. |

---

## Part 4 — Data Interpretation (Beginner Guide)

### 4.1 How to Read the Dataset — Step by Step

**Step 1 — Open the data.** Open `Data.xlsm`, go to sheet `DETAIL`. You see 13,004 rows (transactions) and 21 columns.

**Step 2 — Understand one row.** Each row = one SWIFT international wire transfer. For example:

| Column | Value | Meaning |
|--------|-------|---------|
| `FC___` | `ABIHIDHB` | Customer "ABIHIDHB" initiated this transfer |
| `Date` | `2024-03-15` | On March 15, 2024 |
| `MNT` | `5,200.00` | For 5,200 euros |
| `DEV` | `EUR` | In euros |
| `CV EN TND` | `17,160.00` | Which equals 17,160 Tunisian Dinars |
| `52A_EXPED` | `BNPAFRPP` | Sent via BNP Paribas (France) |
| `52A_ISO_PAYS` | `FR` | From France |
| `FRAIS` | `SHA` | Fees shared between sender and receiver |

**Step 3 — Check the output columns.** After the model runs, new columns appear:

| Column | Value | Meaning |
|--------|-------|---------|
| `anomaly_IF` | `1` | ✅ Normal — this transaction looks typical |
| `anomaly_score_IF` | `0.08` | Score > 0 = clearly normal |
| `high_confidence_fraud` | `0` | Not flagged by either method |

### 4.2 How to Identify Suspicious Patterns

Look for these **red flags** in the data:

| Red Flag | What to Look For | Why |
|----------|-----------------|-----|
| 🔴 Extreme amounts | `CV EN TND` > 1,000,000 | Amounts 27x above average |
| 🔴 Round amounts | `MNT` = exactly 50,000 or 100,000 | Artificial, manually crafted |
| 🔴 High-risk country | `52A_ISO_PAYS` = `IR`, `KP`, `SY` | FATF blacklisted jurisdictions |
| 🔴 Weekend transfer | `Date` falls on Saturday/Sunday | SWIFT transfers are unusual on weekends |
| 🔴 Customer burst | Same `FC___` appears 10+ times in one week | Normally 1–2 times per month |
| 🔴 BIC/Country mismatch | BIC says Bolivia, country says France | Money routed through intermediary |
| 🔴 Unusual currency | EUR transaction from Iran | Expected currency would be IRR |

### 4.3 Normal vs Suspicious Transaction — Concrete Examples

#### ✅ Example 1 — Normal Transaction

| Feature | Value | Assessment |
|---------|-------|-----------|
| `FC___` | `ABIHIDHB` | Known customer |
| `MNT` | `3,247.85 EUR` | Reasonable amount, non-round |
| `52A_ISO_PAYS` | `FR` (France) | Low-risk country |
| `52A_EXPED` | `BNPAFRPP` | Major international bank |
| `FRAIS` | `SHA` | Standard shared fees |
| `Date` | Wednesday | Normal business day |
| `amount_vs_customer_avg` | `1.1x` | Close to customer's average |
| `txn_count_customer_7d` | `1` | Normal frequency |
| **Model output** | `anomaly_IF = 1` | ✅ **NORMAL** |

**Why it's normal**: Moderate amount, from a well-known bank in a low-risk country, on a business day, consistent with the customer's usual pattern.

---

#### 🔴 Example 2 — Suspicious Transaction

| Feature | Value | Assessment |
|---------|-------|-----------|
| `FC___` | `XKZZMQPW` | Less-known customer |
| `MNT` | `500,000.00 EUR` | ⚠️ Very large AND perfectly round |
| `52A_ISO_PAYS` | `IR` (Iran) | ⚠️ FATF blacklisted country |
| `52A_EXPED` | `COMABORJ` | ⚠️ BIC = Bolivia, but country = Iran |
| `FRAIS` | `OUR` | Sender paying all fees (urgency) |
| `Date` | Saturday | ⚠️ Weekend transfer |
| `amount_vs_customer_avg` | `66x` | ⚠️ 66 times the customer's average |
| `txn_count_customer_7d` | `12` | ⚠️ 12 transfers in one week (normally 1) |
| **Model output** | `anomaly_IF = -1, score = -0.18` | 🔴 **ANOMALY** |

**Why it's suspicious**: Multiple red flags stack up — extreme amount, round number, FATF country, BIC/country mismatch, weekend, and a massive deviation from the customer's historical behavior. The model correctly flags this for compliance review.

---

#### 🟡 Example 3 — Edge Case (Flagged but Potentially Legitimate)

| Feature | Value | Assessment |
|---------|-------|-----------|
| `FC___` | `BCDJEFGH` | Corporate customer |
| `MNT` | `450,000.00 EUR` | Large amount, round |
| `52A_ISO_PAYS` | `DE` (Germany) | Low-risk country |
| `52A_EXPED` | `DEUTDEFF` | Deutsche Bank (major bank) |
| `FRAIS` | `OUR` | Sender pays all |
| `Date` | Monday | Normal business day |
| `amount_vs_customer_avg` | `2.1x` | Slightly above average |
| `txn_count_customer_7d` | `2` | Normal frequency |
| **Model output** | `anomaly_IF = -1, score = -0.04` | 🟡 **Borderline anomaly** |

**Why it's an edge case**: The amount is large and round (red flags), but everything else is clean — reputable bank, low-risk country, business day, normal frequency. This is likely a legitimate large corporate payment. The score of `-0.04` (barely negative) confirms the model is **not very confident** about this anomaly. A compliance officer would likely clear this after review.

### 4.4 Quick Reference — Reading the Model Output

| Column | Value | Meaning |
|--------|-------|---------|
| `anomaly_IF` | `1` | ✅ Normal transaction |
| `anomaly_IF` | `-1` | 🔴 Suspicious — needs review |
| `anomaly_score_IF` | `> 0` | Clearly normal |
| `anomaly_score_IF` | `-0.05 to -0.10` | Moderately suspicious |
| `anomaly_score_IF` | `< -0.10` | **Highly suspicious** |
| `high_confidence_fraud` | `1` | 🔴🔴 Flagged by **BOTH** methods — highest priority |
| `high_confidence_fraud` | `0` | Flagged by at most one method |

---

## Summary — Feature Count Breakdown

| Category | Count | Examples |
|----------|-------|---------|
| Raw numeric features | 2 | `MNT`, `CV EN TND` |
| Encoded categorical features | 8 | `DEV_enc`, `FRAIS_enc`, `FC____enc`, etc. |
| Engineered features | 29 | Velocity, ratios, risk flags, delays, etc. |
| **Total features (after selection)** | **39** | Fed into Isolation Forest |

> [!TIP]
> **For your PFE report**: The 29 engineered features represent the **core intellectual contribution** of this project. They transform raw banking data into meaningful behavioral signals that enable the unsupervised model to detect fraud without labeled examples.
