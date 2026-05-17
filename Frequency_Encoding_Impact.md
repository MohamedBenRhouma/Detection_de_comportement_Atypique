# Frequency Encoding vs. Label Encoding: Impact on Fraud Detection

This document explains the rationale, implementation, and impact of replacing **Label Encoding** with **Frequency Encoding** for high-cardinality categorical variables within the banking fraud detection model.

---

## 1. The Previous State: Label Encoding
Previously, the pipeline used `scikit-learn`'s `LabelEncoder` to convert text-based categories (like Customer IDs, Bank BICs, Currencies, and Countries) into numerical format so the machine learning algorithms could process them.

### How it worked:
`LabelEncoder` takes unique text values and assigns them a random integer based purely on alphabetical order.
* Customer `AAEHCGIJ` ➔ `1`
* Customer `ABAHBBJF` ➔ `2`
* Customer `ZXYWWQ` ➔ `1000`

### The Flaw (Fake Ordinality):
Algorithms like Isolation Forest and Mahalanobis Distance treat these encoded numbers mathematically. Because of Label Encoding, the model assumes that:
1. Customer `1000` is mathematically "larger" or "further away" than Customer `1`.
2. Customer `2` sits exactly halfway between Customer `1` and Customer `3`.

For Bank IDs and Customer IDs, **this mathematical order is completely fake and meaningless.** It creates mathematical "noise" that confuses the model, as it forces the algorithm to find patterns in random alphabetical numbering rather than actual transactional behavior.

---

## 2. The Current State: Frequency Encoding
To eliminate this noise, the code was updated to use **Frequency Encoding**. Instead of assigning a random alphabetical ID, we replace the text with **the exact number of times that entity appears in the dataset.**

### How it works:
* `Bank A` (processes 5,000 transactions) ➔ becomes `5000`
* `Bank B` (processes 2 transactions) ➔ becomes `2`
* `Bank C` (processes 5,000 transactions) ➔ becomes `5000`

### Feature Transformation:
The column transforms from a "Meaningless Identifier" into a highly valuable **"Popularity / Commonness Score"**.

---

## 3. Impact on Model Reasoning and Behavior

By making this change, the AI’s logic and reasoning capabilities were massively upgraded in three specific ways:

### A. Meaningful Splits in Isolation Forest
The Isolation Forest algorithm works by drawing mathematical "lines" to isolate anomalies. 
* **Before:** The model tried to isolate transactions by making splits like *"Is the Customer ID greater than 500?"* — which meant nothing.
* **Now:** The model isolates transactions by making splits like *"Does this transaction involve a correspondent bank that only appears 2 times in our entire database?"* 
Because rare actors are mathematically distinct from common actors (e.g., `2` vs `5000`), the trees can easily isolate and flag one-off shell companies or rare counterparties, which is a massive red flag in anti-money laundering (AML).

### B. Accurate Distance in Mahalanobis
Mahalanobis distance measures how far a transaction is from the "statistical center" of normal behavior.
* **Before:** The variance of the `LabelEncoder` column was just the variance of a random sequence of numbers, skewing the multidimensional distance.
* **Now:** The variance of the feature correctly represents the variance in popularity. "Rareness" becomes a true vector in the multidimensional space, mathematically pushing highly unusual, one-off customers further away from the dense cluster of everyday, high-frequency customers.

### C. Eliminating Distractions (Noise Reduction)
The model no longer wastes its processing depth trying to figure out the meaningless numerical gaps between alphabetical IDs. The noise is removed, allowing the algorithms to focus purely on the features that matter: amounts, processing delays, and real behavioral ratios.

---

## 4. Compliance and Auditability

In a real-world banking and AML compliance context, model interpretability is mandatory. 
* **Label Encoding:** Explaining to an auditor that the model flagged a transaction *"because the customer's alphabetical ID is 532"* is impossible to defend.
* **Frequency Encoding:** Explaining that the model flagged a transaction *"because it involves an entity rarely seen in our historical data profile"* is a standard, highly justifiable AML heuristic. 

This change ensures that when SHAP values extract the top reasons for a flagged anomaly, the explanations are rooted in true banking logic rather than data preprocessing artifacts.
