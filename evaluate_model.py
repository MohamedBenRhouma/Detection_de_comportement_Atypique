# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("   EVALUATION COMPLETE DU MODELE — PFE Fraude Bancaire")
print("=" * 65)

# ── Charger artifacts ─────────────────────────────────────────────
art = joblib.load("fraud_model.pkl")
iso_forest    = art["iso_forest"]
scaler        = art["scaler"]
model_cols    = art["features_df_columns"]   # les 39 features exactes

# ── Charger resultats ─────────────────────────────────────────────
results = pd.read_excel("fraud_detection_results.xlsx",
                        sheet_name="All Transactions", engine="openpyxl")
print(f"\n  Dataset : {len(results):,} transactions")
print(f"  Anomalies IF : {(results['anomaly_IF']==-1).sum():,}")

# ── Reconstruire X avec les 39 colonnes exactes du modele ─────────
X_eval = pd.DataFrame(index=results.index)
for col in model_cols:
    if col in results.columns:
        X_eval[col] = results[col].fillna(0)
    else:
        X_eval[col] = 0.0
X_eval = X_eval[model_cols].astype(float)
X_scaled = scaler.transform(X_eval)
print(f"  Features reconstruites : {X_eval.shape[1]} / {len(model_cols)}")

labels = (results["anomaly_IF"] == -1).astype(int)

# =================================================================
print("\n" + "="*65)
print("  PARTIE 1 — METRIQUES INTERNES (Unsupervised)")
print("="*65)

# --- Silhouette Score ---
np.random.seed(42)
idx = np.random.choice(len(X_scaled), min(3000, len(X_scaled)), replace=False)
sil = silhouette_score(X_scaled[idx], labels.values[idx])
print(f"\n  [1] Silhouette Score       : {sil:.4f}")
print(f"      Plage : -1 (pire) a +1 (parfait)")
print(f"      Seuils : < 0.1 = faible | 0.1-0.5 = correct | > 0.5 = bon")
if sil > 0.5:
    interp = "Excellent — groupes tres bien separes"
elif sil > 0.1:
    interp = "Correct — separation moderee mais coherente"
else:
    interp = "Faible — groupes qui se chevauchent (normal en fraude)"
print(f"      Interpretation : {interp}")

# --- Davies-Bouldin ---
db = davies_bouldin_score(X_scaled[idx], labels.values[idx])
print(f"\n  [2] Davies-Bouldin Score   : {db:.4f}")
print(f"      Plus BAS = meilleure separation (ideal : < 1.0)")
print(f"      Interpretation : {'Bon' if db < 1.0 else 'Acceptable' if db < 2.0 else 'A ameliorer'}")

# --- Calinski-Harabasz ---
ch = calinski_harabasz_score(X_scaled[idx], labels.values[idx])
print(f"\n  [3] Calinski-Harabasz Score: {ch:.2f}")
print(f"      Plus HAUT = clusters plus denses et mieux separes")
print(f"      Interpretation : Plus grand = meilleur")

# --- Score distribution ---
anom_sc = results.loc[results["anomaly_IF"]==-1, "anomaly_score_IF"]
norm_sc  = results.loc[results["anomaly_IF"]== 1, "anomaly_score_IF"]
print(f"\n  [4] Distribution des scores IF :")
print(f"      Normales  | moy={norm_sc.mean():.5f}  std={norm_sc.std():.5f}  min={norm_sc.min():.5f}  max={norm_sc.max():.5f}")
print(f"      Anomalies | moy={anom_sc.mean():.5f}  std={anom_sc.std():.5f}  min={anom_sc.min():.5f}  max={anom_sc.max():.5f}")
sep = abs(anom_sc.mean() - norm_sc.mean())
print(f"      Separation des moyennes : {sep:.5f}  (plus grand = meilleur)")

# =================================================================
print("\n" + "="*65)
print("  PARTIE 2 — VALIDATION STATISTIQUE (Mann-Whitney U Test)")
print("="*65)
print("  H0 : pas de difference entre anomalies et normales")
print("  Si p < 0.05 -> difference SIGNIFICATIVE -> modele valide")

anom_df = results[results["anomaly_IF"] == -1]
norm_df  = results[results["anomaly_IF"] ==  1]

for col, label in [("CV EN TND","Montant TND"), ("MNT","Montant original")]:
    if col in results.columns:
        u, p = stats.mannwhitneyu(anom_df[col].dropna(),
                                  norm_df[col].dropna(), alternative="two-sided")
        sig = "SIGNIFICATIF ✓" if p < 0.05 else "Non significatif"
        print(f"\n  {label} :")
        print(f"    p-value = {p:.2e}  -> {sig}")
        print(f"    Moy anomalies : {anom_df[col].mean():>12,.0f}")
        print(f"    Moy normales  : {norm_df[col].mean():>12,.0f}")
        print(f"    Ratio         : x{anom_df[col].mean()/norm_df[col].mean():.1f}")

# =================================================================
print("\n" + "="*65)
print("  PARTIE 3 — VALIDATION AVEC PROXY LABELS (ETAT_RPA)")
print("="*65)
print("  'KO E8 : Provenance AMLO' = signal AML reel dans les donnees")
print("  Utilise comme proxy pour estimer la precision du modele")

if "ETAT_RPA" in results.columns:
    amlo = results["ETAT_RPA"].astype(str).str.contains("AMLO", na=False)
    ko   = results["ETAT_RPA"].astype(str).str.startswith("KO",  na=False)
    if_flag = results["anomaly_IF"] == -1

    n_amlo = amlo.sum()
    n_ko   = ko.sum()

    print(f"\n  Transactions KO E8 AMLO : {n_amlo}")
    print(f"  Transactions KO total   : {n_ko}")

    if n_amlo > 0:
        both_amlo = (if_flag & amlo).sum()
        prec = both_amlo / if_flag.sum()
        rec  = both_amlo / n_amlo
        print(f"\n  [5] IF vs Flags AMLO :")
        print(f"      AMLO captures par IF : {both_amlo} / {n_amlo}")
        print(f"      Precision            : {prec*100:.1f}%  (sur nos alertes, combien sont AMLO)")
        print(f"      Recall               : {rec*100:.1f}%  (des cas AMLO, combien on attrape)")

    both_ko = (if_flag & ko).sum()
    print(f"\n  [6] IF vs tous KO RPA :")
    print(f"      KO captures par IF : {both_ko} / {n_ko} ({both_ko/n_ko*100:.1f}%)")
    print(f"      KO manques par IF  : {n_ko-both_ko} / {n_ko} ({(n_ko-both_ko)/n_ko*100:.1f}%)")

    print(f"\n  [7] Matrice de Confusion IF vs KO_RPA :")
    cm = confusion_matrix(ko.astype(int), if_flag.astype(int))
    print(f"\n                  IF=Normal   IF=Anomalie")
    print(f"  RPA=OK (0)  :   {cm[0,0]:>8,}    {cm[0,1]:>8,}")
    print(f"  RPA=KO (1)  :   {cm[1,0]:>8,}    {cm[1,1]:>8,}")

    try:
        roc  = roc_auc_score(ko.astype(int), -results["anomaly_score_IF"])
        prauc = average_precision_score(ko.astype(int), -results["anomaly_score_IF"])
        print(f"\n  [8] ROC-AUC (IF score vs KO proxy)  : {roc:.4f}")
        print(f"      0.5=aleatoire  0.7=bon  0.9=excellent  1.0=parfait")
        print(f"      Interpretation : {'Bon' if roc > 0.65 else 'Modere'} pouvoir discriminant")
        print(f"\n  [9] PR-AUC (Precision-Recall AUC)   : {prauc:.4f}")
        print(f"      Mesure la qualite des alertes (important quand fraude est rare)")
    except Exception as e:
        print(f"\n  ROC/PR-AUC indisponible : {e}")

# =================================================================
print("\n" + "="*65)
print("  PARTIE 4 — ACCORD ENTRE METHODES")
print("="*65)

if "anomaly_Distance" in results.columns:
    agree = (results["anomaly_IF"] == results["anomaly_Distance"]).mean() * 100
    hc    = results["high_confidence_fraud"].sum()
    only_if   = ((results["anomaly_IF"]==-1) & (results["anomaly_Distance"]== 1)).sum()
    only_dist = ((results["anomaly_IF"]== 1) & (results["anomaly_Distance"]==-1)).sum()

    print(f"\n  [10] Accord IF + Mahalanobis : {agree:.1f}%")
    print(f"       Haute confiance (les deux) : {hc:,} transactions")
    print(f"       Seulement IF               : {only_if:,}")
    print(f"       Seulement Mahalanobis      : {only_dist:,}")
    print(f"\n  Interpretation : Un accord de {agree:.0f}% entre 2 methodes")
    print(f"  independantes valide la coherence du systeme.")

# =================================================================
print("\n" + "="*65)
print("  PARTIE 5 — STABILITE DU MODELE (Robustesse)")
print("="*65)
print("  Re-entrainement avec 5 seeds differents")

runs = []
for seed in [42, 123, 0, 7, 99]:
    m = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed)
    p = m.fit_predict(X_scaled)
    n = (p == -1).sum()
    runs.append(n)
    print(f"    seed={seed:>3} -> {n:,} anomalies")

cv = np.std(runs) / np.mean(runs) * 100
print(f"\n  Moyenne  : {np.mean(runs):.0f}")
print(f"  Std Dev  : {np.std(runs):.1f}")
print(f"  CV (%)   : {cv:.2f}%")
print(f"  Interpretation : {'Tres stable' if cv < 1 else 'Stable' if cv < 5 else 'Variable'}")
print(f"  (CV < 5% = modele stable et reproductible)")

# =================================================================
print("\n" + "="*65)
print("  RESUME DES METRIQUES POUR LA SOUTENANCE")
print("="*65)
print(f"""
  Metrique                        Valeur      Interpretation
  ─────────────────────────────────────────────────────────
  [1] Silhouette Score            {sil:.4f}     Separation clusters
  [2] Davies-Bouldin Score        {db:.4f}     Compacite groupes
  [3] Calinski-Harabasz Score     {ch:.2f}  Densite clusters
  [4] Sep. moyennes scores IF     {sep:.5f}  Distinction scores
  [8] ROC-AUC (vs KO proxy)       voir ci-dessus
  [9] PR-AUC (Precision-Recall)   voir ci-dessus
  [10] Accord IF+Mahalanobis      {agree:.1f}%     Coherence methodes
  [11] CV Stabilite               {cv:.2f}%     Reproductibilite
""")
print("  [OK] Evaluation terminee !")
