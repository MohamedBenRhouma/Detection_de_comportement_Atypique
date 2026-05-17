# Dictionnaire des Caractéristiques (Features) : Détection de Comportements Atypiques

Ce document détaille l'ensemble des caractéristiques (*features*) ingénierisées pour le modèle d'Intelligence Artificielle. Plutôt que de fournir des données brutes au modèle, nous avons créé des "méta-variables" regroupées par concepts comportementaux. C'est cette architecture qui permet au modèle de détecter l'atypisme.

---

## 1. Groupe 1 : Les Variables Temporelles et Délais (Frictions Opérationnelles)
*Ce groupe analyse les anomalies dans le temps de traitement des transactions SWIFT.*

*   **`processing_delay_days`** *(D_EXEC - Date)* : Le nombre de jours entre la création de la transaction et son exécution réelle. Un délai anormalement long peut indiquer une transaction bloquée manuellement par la conformité, ou un problème administratif atypique.
*   **`value_delay_days`** *(DATE_VL - Date)* : Décalage entre la date de transaction et la date de valeur.
*   **`txn_weekend`** : Indicateur binaire (0 ou 1) si la transaction a été effectuée ou exécutée un week-end. Les virements SWIFT d'entreprise (B2B) se font massivement en semaine ; une activité le dimanche est un signal d'atypisme très fort.

## 2. Groupe 2 : Les Variables de Vélocité (Rythme à Court Terme)
*Ce groupe mesure les "rafales" d'activité (bursts). Il agit comme la mémoire à court terme du modèle (fenêtre glissante de 7 jours).*

*   **`txn_count_sender_7d`** : Le nombre de transactions effectuées par cette banque correspondante spécifique (`52A_EXPED`) au cours des 7 derniers jours.
*   **`txn_count_country_7d`** : Le volume de transactions provenant de ce pays spécifique sur 7 jours. Détecte une soudaine vague de fonds depuis une région donnée.
*   **`txn_count_customer_7d`** : Le nombre de transactions initiées par le client (`FC___`) sur 7 jours. C'est un indicateur direct de changement soudain de rythme (ex: technique du "Smurfing" ou fractionnement).

## 3. Groupe 3 : Les Ratios Historiques (Déviations à Long Terme)
*Ce groupe compare la transaction actuelle à l'historique complet (all-time) de l'entité. Il agit comme la mémoire à long terme du modèle.*

*   **`amount_vs_sender_avg`** : Le ratio entre le montant actuel et la moyenne historique de l'expéditeur. *(Exemple : Si le ratio est de 30x, cela signifie que la banque envoie 30 fois plus d'argent que sa moyenne habituelle).*
*   **`amount_vs_sender_std`** : Le Z-score (écart-type) du montant par rapport à l'historique de l'expéditeur. Mesure mathématiquement à quel point le montant s'éloigne de la "zone de confort" habituelle de cet expéditeur.
*   **`amount_vs_country_avg`** : Le ratio du montant par rapport à la moyenne habituelle des virements venant de ce pays.

## 4. Groupe 4 : Le Profilage d'Acteur (Popularité et Rareté)
*Ces variables utilisent l'Encodage par Fréquence (Frequency Encoding) pour mesurer à quel point un acteur ou un attribut est commun ou rare dans l'écosystème global de la banque.*

*   **`FC____freq`** (Client) & **`52A_EXPED_freq`** (Banque) : Remplace l'identifiant texte par le nombre total d'apparitions de ce client/banque. Permet au modèle de faire la différence entre une multinationale quotidienne (très commun) et un individu "one-shot" (très rare).
*   **`52A_ISO_PAYS_freq`**, **`DEV_freq`**, **`FRAIS_freq`** : Fréquence d'utilisation du pays, de la devise et du type de frais. Permet de détecter si une transaction utilise une devise exotique ou un schéma de frais inhabituel pour la banque.

## 5. Groupe 5 : Les Indicateurs de Risque et de Forme (Signaux Forts)
*Ce groupe repère les anomalies structurelles ou légales de la transaction.*

*   **`log_MNT`** & **`log_CV_EN_TND`** : Valeur logarithmique du montant brut et du montant converti en dinars. Cela écrase les valeurs extrêmes (ex: des milliards) pour permettre au modèle de comparer logiquement les ordres de grandeur.
*   **`is_round_amount`** : Indicateur binaire (1 si le montant est un compte rond parfait, sans centimes). Dans le commerce légitime B2B, les factures comportent souvent des centimes. Les montants parfaitement ronds sont fréquents dans le transfert de patrimoine atypique ou les tests de blanchiment.
*   **`country_mismatch_risk`** : Indicateur de contradiction. Le pays dérivé du code BIC de la banque est-il différent du pays d'origine déclaré ?
*   **`high_risk_country`** : Indicateur de conformité. Le pays fait-il partie des listes de surveillance internationales (GAFI / FATF) ?

## 6. Groupe 6 : Les Signaux de Données Manquantes (Rareté Structurelle)
*Ce groupe analyse la structure du message SWIFT lui-même.*

*   **`assoc_present`** & **`chps_72_present`** : Indicateurs binaires précisant si ces champs optionnels SWIFT sont remplis. Le champ ASSOC, par exemple, n'est rempli que dans 1.3% des cas. Son utilisation est donc en soi un acte statistiquement atypique que le modèle doit prendre en compte.
