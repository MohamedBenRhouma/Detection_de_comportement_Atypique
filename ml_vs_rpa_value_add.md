# 🚀 Valeur Ajoutée du Modèle ML par rapport au RPA (UiPath)

Ce document explique clairement pourquoi le robot RPA existant ne suffit pas pour détecter la fraude complexe, et comment le modèle de Machine Learning (Isolation Forest) comble ces lacunes. **C'est l'argument principal de votre PFE.**

---

## 🤖 1. Les limites du système RPA actuel (Règles rigides)

Le robot RPA (UiPath) actuel est programmé avec des règles "dures" (hardcoded). Il agit comme un garde-frontière avec une checklist stricte.

| Règle RPA (Checklist) | Logique d'évaluation | La faille (Limite) |
|-----------------------|----------------------|--------------------|
| **Montant > Seuil** | Si `MNT > 500 000` → KO | Le seuil est fixe. Un virement de 499 999 TND passe sans alerte. |
| **Pays à risque** | Si `Pays ∈ [Liste Noire]` → KO | La liste est figée. Un pays non blacklisté mais inhabituel passe. |
| **État du compte** | Si `Compte = Fermé/Bloqué` → KO | Vérification binaire (Oui/Non), aucune analyse comportementale. |
| **Données manquantes**| Si `Chrono absent` → KO | C'est une erreur technique/format, pas de la détection de fraude. |

**Problème fondamental du RPA :** Il raisonne de manière binaire (OUI/NON) sur chaque critère **isolément**, et n'a **aucune mémoire** des transactions précédentes.

---

## 🧠 2. Ce que le modèle ML (Isolation Forest) fait EN PLUS

Le modèle de Machine Learning agit comme un analyste expérimenté doté d'une mémoire statistique parfaite. Voici ses 5 super-pouvoirs que le RPA n'a pas :

### A. La détection par combinaison de "signaux faibles" 🔴 Impossible en RPA
Le robot vérifie chaque règle séparément. Le ML regarde **les 36 dimensions en même temps**.

> **Exemple de fraude qui trompe le robot :**
> - Montant = 400 000 TND (sous le seuil RPA) ✅
> - Pays = Tunisie (pas sur liste noire) ✅
> - Compte = Ouvert et valide ✅
> 
> **Décision RPA : VALIDE** (L'humain ne verra jamais cette transaction).
> 
> **MAIS le ML remarque :** C'est un montant très rond (400k pile) + les frais sont à la charge du bénéficiaire (rare pour ce type de compte) + c'est le 3ème virement de cet expéditeur en 2 jours + le montant habituel de cet expéditeur est de 5 000 TND.
> 
> **Décision ML : ANOMALIE 🚨** (Combinaison de 4 signaux faibles = hautement suspect).

### B. Le profilage comportemental historique 🔴 Impossible en RPA
Le RPA ne connaît que la transaction à l'instant T. Le ML compare la transaction à l'historique.

| Feature ML | Ce qu'elle détecte | Pourquoi le RPA est aveugle |
|------------|--------------------|-----------------------------|
| `amount_vs_sender_avg` | Ce client envoie normalement 5 000 TND. Aujourd'hui il envoie 400 000 TND (**Ratio x80**). | Le RPA n'a pas de mémoire pour calculer la moyenne historique du client. |
| `amount_vs_sender_std` | La transaction dévie massivement de la variance habituelle du client (Z-score). | Le RPA ne calcule pas de statistiques. |

### C. L'analyse de vélocité (Spike Detection) 🔴 Impossible en RPA
Le ML détecte des "rafales" d'activité suspectes dans le temps.

| Feature ML | Ce qu'elle détecte |
|------------|--------------------|
| `txn_count_sender_7d` | Un expéditeur fait 12 virements en 7 jours (alors que sa moyenne est de 1 par mois). |
| `txn_count_country_7d` | Un volume anormal et soudain de transactions provenant d'un pays spécifique. |

### D. La détection d'incohérences géographiques 🟡 Partiellement possible en RPA
Le ML croise les données pour trouver des mensonges ou du blanchiment par pays rebond.

| Feature ML | Ce qu'elle détecte | Pourquoi le RPA est aveugle |
|------------|--------------------|-----------------------------|
| `country_mismatch_risk` | Le code BIC correspond à une banque en Roumanie, mais le pays déclaré est la France. | Le RPA vérifie juste si le pays est blacklisté, il ne croise pas l'origine réelle du BIC avec le pays déclaré. |
| `sender_country_diversity` | Un même code BIC expéditeur apparaît avec 5 pays d'origine différents (typique des sociétés écrans). | Le RPA ne fait pas d'agrégation par BIC. |

### E. Les patterns statistiques invisibles 🔴 Impossible en RPA

| Feature ML | Ce qu'elle détecte |
|------------|--------------------|
| `is_round_amount` | Un montant de 970 000,00 USD pile est suspect dans le monde B2B (où les factures ont des centimes). Le ML l'apprend. |
| `amount_rank_pct` | Le ML sait immédiatement si une transaction est dans le "Top 1%" des montants globaux. |

---

## 🎯 3. Résumé "Elevator Pitch" pour le Jury (À mémoriser)

> *"Le robot RPA actuel agit comme une barrière de péage : il applique des règles binaires et isolées (ex: le montant dépasse-t-il le plafond ?). Mon modèle d'Intelligence Artificielle agit comme un détective financier : il analyse 36 dimensions simultanément, croise les incohérences géographiques, et compare chaque transaction au comportement historique de l'expéditeur. Le ML ne remplace pas le RPA, il comble ses angles morts en détectant des fraudes sophistiquées qui respectent parfaitement les règles de format."*
