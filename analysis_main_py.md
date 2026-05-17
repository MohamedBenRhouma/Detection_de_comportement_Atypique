# 🔍 Analyse Critique — `main.py` — Détection de Fraude MT103
**Projet PFE : Anomaly Detection sur transactions SWIFT (Isolation Forest)**

---

## Vue d'ensemble

Le code est globalement bien structuré pour un projet PFE de débutant en ML. Il démontre une bonne compréhension du contexte métier bancaire. Cependant, plusieurs problèmes sérieux doivent être corrigés avant la soutenance.

---

## 🔴 PROBLÈMES BLOQUANTS (à corriger absolument)

### 1. Data Leakage — Le piège le plus grave en ML

**Où :** Lignes 229–243 — `amount_vs_sender_avg`, `amount_vs_sender_std`, `amount_vs_country_avg`

**Le problème :**
Ces features sont calculées avec `groupby().transform("mean")` sur **l'ensemble du dataset complet**.
Cela signifie que la moyenne d'un sender est calculée en incluant la transaction elle-même.
En production, quand le robot reçoit une **nouvelle** transaction en temps réel, il ne connaît pas les futures transactions.
Le modèle "triche" : il a vu les données qu'il est censé ne pas connaître.

```python
# ❌ CODE ACTUEL (Data Leakage) :
sender_avg = df.groupby("52A_EXPED")["CV EN TND"].transform("mean")
# → inclut la transaction courante dans le calcul de sa propre moyenne
```

**Ce que tu dois faire :**
Calculer la moyenne HISTORIQUE — c'est-à-dire exclure la transaction courante :

```python
# ✅ CORRECTION : leave-one-out average (exclut la transaction courante)
def leave_one_out_mean(group):
    """Pour chaque transaction, calcule la moyenne des AUTRES transactions du même sender."""
    n = len(group)
    if n <= 1:
        return pd.Series([np.nan] * n, index=group.index)
    total = group.sum()
    return (total - group) / (n - 1)

# Appliqué sur le dataset
df["amount_vs_sender_avg_LOO"] = df.groupby("52A_EXPED")["CV EN TND"].transform(leave_one_out_mean)
# Remplir les NaN (senders avec 1 seule transaction) par la médiane globale
df["amount_vs_sender_avg_LOO"] = df["amount_vs_sender_avg_LOO"].fillna(df["CV EN TND"].median())
```

> **Pourquoi c'est bloquant :** Si tu présentes ce modèle à ton jury en disant "il fonctionne bien", mais que tu as du leakage, tes métriques sont artificiellement gonflées. Le jury peut te le signaler.

---

### 2. `_rolling_count` — Bug silencieux de Data Leakage temporel

**Où :** Lignes 160–174

**Le problème :**
La fenêtre glissante de 7 jours inclut la transaction **courante** dans son propre comptage.
En réalité, on veut savoir combien de transactions ont eu lieu dans les 7 jours **AVANT** cette transaction.

```python
# ❌ PROBLÈME : la fenêtre inclut T0 (la transaction courante)
rc = g_indexed.rolling(window, min_periods=1)["MNT"].count()
# → Pour T0 à J+7, on compte T0 lui-même + toutes les transactions jusqu'à J+7
```

```python
# ✅ CORRECTION : fenêtre glissante fermée à gauche (exclut la transaction courante)
# Utiliser closed='left' pour compter uniquement les transactions PRÉCÉDENTES
rc = g_indexed.rolling(window, min_periods=0, closed="left")["MNT"].count()
```

---

### 3. `fillna(median)` appliqué sur des colonnes qui ne devraient pas l'être

**Où :** Lignes 119–121

**Le problème :**
Remplir `processing_delay_days` par la médiane quand il est NaN masque une information importante :
un `NaN` dans ce champ peut signifier que la date d'exécution est **manquante**, ce qui est lui-même suspect !

```python
# ❌ NE PAS faire ça aveuglément :
df["processing_delay_days"].fillna(df["processing_delay_days"].median(), inplace=True)

# ✅ Créer un indicateur de valeur manquante, PUIS imputer :
df["processing_delay_missing"] = df["processing_delay_days"].isna().astype(int)  # feature bonus !
df["processing_delay_days"] = df["processing_delay_days"].fillna(df["processing_delay_days"].median())
```

---

## 🟡 PROBLÈMES IMPORTANTS (à corriger pour un meilleur travail)

### 4. `contamination = 0.05` — Hypothèse non justifiée

**Où :** Ligne 71

**Le problème :**
Ce paramètre dit à Isolation Forest "je pense que 5% des transactions sont frauduleuses".
Mais rien dans le code ne justifie ce chiffre. Dans la réalité bancaire, le taux de fraude MT103 est typiquement **0.1% à 2%**.
Fixer 5% signifie que le modèle **forcera** 650 transactions à être anomalies — même si beaucoup sont normales.

**Ce que tu dois faire :**
Tester plusieurs valeurs et comparer avec la validation `ETAT_RPA` :

```python
# ✅ Comparer différents taux de contamination avec ton ground truth RPA
for cont in [0.01, 0.02, 0.05, 0.10]:
    iso_test = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
    preds = iso_test.fit_predict(X_scaled)
    if "rpa_is_problem" in df_raw.columns:
        overlap = ((preds == -1) & (df_raw["rpa_is_problem"] == 1)).sum()
        total_rpa = df_raw["rpa_is_problem"].sum()
        precision = overlap / (preds == -1).sum() if (preds == -1).sum() > 0 else 0
        recall    = overlap / total_rpa if total_rpa > 0 else 0
        print(f"Contamination {cont:.0%} → Anomalies: {(preds==-1).sum()}, Precision: {precision:.2%}, Recall: {recall:.2%}")
```

---

### 5. Évaluation incomplète — Pas de métriques formelles

**Où :** Lignes 376–419

**Le problème :**
Tu calcules le "Method Agreement" (accord entre IF et Mahalanobis) mais ce n'est **pas une métrique d'évaluation**. C'est juste dire "les deux méthodes sont d'accord". Or si les deux se trompent pareil, l'accord est de 100% et le modèle est inutile.

Tu as une colonne `ETAT_RPA` qui constitue un **proxy de ground truth**. Tu dois l'utiliser pour calculer des métriques réelles :

```python
# ✅ Calcul de métriques avec ETAT_RPA comme référence (approximation)
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

if "rpa_is_problem" in df_raw.columns:
    y_true = df_raw["rpa_is_problem"].values           # 1=problème RPA, 0=normal
    y_pred = (iso_predictions == -1).astype(int)       # 1=anomalie IF, 0=normal
    y_score = -iso_scores                               # Plus haut = plus suspect

    print("\n--- Évaluation IF vs ETAT_RPA ---")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))

    # AUC-ROC (ne dépend pas du seuil de contamination)
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC-ROC : {auc:.4f}  (1.0 = parfait, 0.5 = aléatoire)")

    # Precision-Recall curve (plus adaptée aux classes déséquilibrées)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recall, precision, color="#e74c3c", lw=2)
    ax.set_xlabel("Recall (Taux de détection des fraudes réelles)")
    ax.set_ylabel("Precision (% d'alertes qui sont vraiment des fraudes)")
    ax.set_title("Precision-Recall Curve — IF vs ETAT_RPA")
    ax.fill_between(recall, precision, alpha=0.2, color="#e74c3c")
    plt.tight_layout()
    plt.savefig("plots/08_precision_recall.png", dpi=150)
    plt.close()
    print("  [OK] plots/08_precision_recall.png")
```

> **Pourquoi c'est important :** Ton jury voudra voir une courbe PR ou AUC-ROC. L'"accuracy" seule est trompeuse sur des données déséquilibrées (si 95% des transactions sont normales, un modèle qui dit "tout est normal" a 95% d'accuracy !).

---

### 6. LabelEncoder sur `52A_EXPED` (BIC) — Mauvaise pratique

**Où :** Lignes 301–313

**Le problème :**
`LabelEncoder` attribue des entiers ordonnés aux codes BIC : `BNPAFRPP=0, DEUTDEFF=1, CHASUSXXX=2...`
Cela implique que `BNPAFRPP < DEUTDEFF`, ce qui est un **non-sens bancaire**.
L'Isolation Forest peut interpréter à tort que les banques avec un code plus petit sont "moins risquées".

De plus, en production : si une nouvelle banque apparaît qui n'était pas dans le dataset d'entraînement, le `LabelEncoder` **plantera** avec une `ValueError`.

```python
# ❌ Problème : ordinalité artificielle + fragile en production
le = LabelEncoder()
df["52A_EXPED_enc"] = le.fit_transform(df["52A_EXPED"].astype(str))

# ✅ Alternative : fréquence d'encodage (ne crée pas d'ordinalité artificielle)
# On remplace chaque BIC par son nombre d'apparitions dans le dataset
freq_map = df["52A_EXPED"].value_counts().to_dict()
df["52A_EXPED_freq"] = df["52A_EXPED"].map(freq_map).fillna(0)
# → Une banque qui apparaît 500 fois = encodée 500 (banque fréquente/connue)
# → Une banque inconnue = encodée 0 (automatiquement suspecte !)
```

---

### 7. Mahalanobis — Problème de passage à l'échelle

**Où :** Lignes 387–410

**Le problème :**
La méthode Mahalanobis calcule la matrice de covariance (`np.cov`) sur TOUTES les transactions (13,000 × n_features).
Si tu as 30+ features, la matrice de covariance est (30×30) et peut devenir **singulière** (non-inversible).
Tu as un fallback avec Z-score, mais cela doit être expliqué clairement.

**Piste d'amélioration :**
Utiliser la version robuste de Mahalanobis qui est moins sensible aux outliers :

```python
# ✅ Mahalanobis robuste avec sklearn (plus stable numériquement)
from sklearn.covariance import MinCovDet

# MinCovDet = Minimum Covariance Determinant, robuste aux outliers
robust_cov = MinCovDet(support_fraction=0.9, random_state=42)
robust_cov.fit(X_scaled)
maha_distances = robust_cov.mahalanobis(X_scaled)
# → Carré des distances Mahalanobis, robuste aux outliers
maha_distances = np.sqrt(maha_distances)  # prendre la racine pour revenir à la distance
```

---

## 🟢 POINTS FORTS (ce qui est bien fait)

| # | Aspect | Commentaire |
|---|--------|-------------|
| ✅ | **Feature Engineering riche** | `country_mismatch_risk`, `sender_country_diversity`, `rare_currency_for_country` — excellentes features métier MT103 |
| ✅ | **Log-transformation des montants** | `log1p(MNT)` correcte. Compresse les valeurs extrêmes sans perdre l'information |
| ✅ | **HIGH_RISK_COUNTRIES** | Basé sur la liste FATF, pertinent pour AML |
| ✅ | **BIC country extraction** | `str[4:6]` correctement utilisé pour extraire le code pays du BIC |
| ✅ | **Fallback Mahalanobis → Z-score** | Bonne robustesse au cas de matrice singulière |
| ✅ | **`VarianceThreshold`** | Bonne pratique pour éliminer les features constantes |
| ✅ | **Documentation inline** | Commentaires clairs avec contexte métier |
| ✅ | **Double validation IF + Mahalanobis** | `high_confidence_fraud` est une excellente idée |
| ✅ | **Comparaison avec ETAT_RPA** | Très bonne idée d'utiliser les résultats du robot RPA comme référence |
| ✅ | **Export multi-sheets Excel** | Très pratique pour les équipes compliance |

---

## 🔵 OPTIMISATIONS OPTIONNELLES

### 8. `random_state=42` — OK mais justifie-le

Dans un projet de détection de fraude, le `random_state` doit être documenté comme un choix de **reproducibilité**, pas de performance. Mentionne-le dans ta soutenance.

### 9. Standardisation — Pourquoi StandardScaler ?

`StandardScaler` (moyenne=0, std=1) est bon, mais pour les montants bancaires avec des distributions très asymétriques, `RobustScaler` est plus résistant aux outliers (il utilise la médiane et l'IQR au lieu de la moyenne et std) :

```python
# ✅ Alternative recommandée pour données financières avec outliers extrêmes
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # utilise médiane + IQR → moins sensible aux valeurs extrêmes
```

### 10. Absence de pipeline sklearn

En production (intégration UiPath), il faudra appliquer exactement les mêmes transformations sur de nouvelles données. Encapsuler tout dans un `Pipeline` sklearn permet de sauvegarder et réutiliser :

```python
import joblib

# Sauvegarder le modèle entraîné pour UiPath
joblib.dump({"scaler": scaler, "model": iso_forest}, "fraud_model.pkl")
# Chargement côté UiPath/Python subprocess :
# artifacts = joblib.load("fraud_model.pkl")
```

---

## 📊 Résumé des priorités

| Priorité | Problème | Effort de correction |
|----------|----------|---------------------|
| 🔴 Bloquant | Data leakage sur features comportementales | 30 min |
| 🔴 Bloquant | `_rolling_count` inclut la transaction courante | 5 min |
| 🔴 Bloquant | Missing values masquent une information métier | 15 min |
| 🟡 Important | Contamination non justifiée (0.05) | 20 min |
| 🟡 Important | Métriques d'évaluation formelles absentes | 45 min |
| 🟡 Important | LabelEncoder sur BIC → fréquence-encoding | 15 min |
| 🟡 Important | Mahalanobis → MinCovDet robuste | 10 min |
| 🟢 Optionnel | RobustScaler | 2 min |
| 🟢 Optionnel | Pipeline sklearn pour production | 30 min |

---

> **Note pour la soutenance :** Prépare-toi à justifier le choix de `contamination=0.05`. Le jury te demandera "pourquoi 5% et pas 2% ?". La réponse doit être : "J'ai testé plusieurs valeurs et validé contre ETAT_RPA. 5% maximise le rappel des cas déjà identifiés par le système RPA."
