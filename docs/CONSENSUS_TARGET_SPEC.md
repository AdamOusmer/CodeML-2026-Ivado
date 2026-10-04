# Cible de consensus, base et résidu TabM

Décidé après cinq revues d'équité indépendantes (`docs/reviews/CONSENSUS.md`). À lire avec `HARNESS_SPEC.md` et
`JURY_SPEC.md`.

## 1. Pourquoi

- L'étiquette du comité n'est identifiée qu'à `γ = a − d` près (différence régionale légitime moins discrimination) ;
  « comité moins pénalité » garde la récompense du revenu, que quatre examinateurs sur cinq rejettent.
- Le seuil d'heures à 10 h n'est pas identifié (retenu dans 48 % des rééchantillonnages) : crédit linéaire.
- Les jurés entraînés reproduisent la règle d'entraînement (AUC hors échantillon ≈ 1) et n'ajoutent aucun signal
  indépendant ; les cinq règles des examinateurs sont des définitions réellement différentes du mérite.

## 2. Cible

- `Config.target = "consensus"` : les étiquettes d'entraînement sont les `k` meilleurs (k = round(part × n), part lue
  dans l'historique) de la règle de consensus sur l'historique, standardisée par la moyenne et l'écart-type de
  l'historique. Le modèle du comité reste ajusté pour le diagnostic (pénalité, IC, test des pentes).
- La règle est exprimée en unités de cote R : `z(R) + h × z(heures) + w × z(log revenu)`.
  - `h` = `DECLARED_HOURS_WEIGHT` = 0,185 : **choix de modélisation déclaré, retenu par essais**, proche du rapport
    |coefficient heures| / |coefficient R| du comité (≈ 0,1835), qui reste calculé à chaque exécution pour le diagnostic
    (« heures/R du comité » dans `decision_record.json`). Défini dans `src/policy/references.py`.
  - `w` = `DECLARED_INCOME_WEIGHT` = +0,025 : **choix de modélisation déclaré, retenu par essais, dans la plage de
    désaccord des examinateurs** (0 à +0,19 ; besoin −0,05). Défini à un seul endroit, `src/policy/references.py`.
- Règles de référence (`src/policy/references.py`) : `consensus` (heures 0,185, revenu +0,025), et les cinq examinateurs avec leur
  propre poids du revenu lu dans `docs/reviews/consensus.json` (mérite 0, besoin −0,05, juridique 0, processus de données
  +0,19, régional 0), heures dérivées du comité.

## 3. Jury en panel

- Jurés `ref_merit`, `ref_need`, `ref_legal`, `ref_data_process`, `ref_regional` : chacun note le lot avec sa règle.
  Aveugles à la région.
- Configuration « consensus panel » : panel des cinq règles, quorum 0,8 (4 sur 5) ; échanges appariés.
- Configuration **déclarée** : même panel, en **mode audit** (`audit_only`) : les votes et les échanges contestés sont
  enregistrés, aucun n'est appliqué. Le raisonnement (`ReasoningSettings`) est aussi en mode audit (traces seulement).

## 4. Résidu TabM

- Appris sur l'historique seulement, hors machine (`kaggle/train.py`, GPU Kaggle gratuit) : TabM borné, entrées cote R
  et heures seulement, au-dessus d'un modèle linéaire figé ; pénalité du résidu choisie par la perte logarithmique
  hors échantillon (5 plis × 3 graines) ; grille de monotonie en R et en heures.
- Artefact `models/tabm_residual/` : `residuals.csv` (`id_candidat, residual_rsd`, unités de cote R) et `manifest.json`
  (SHA-256 des entrées, graines, plis, sélection de la pénalité, grille de monotonie, versions).
- Score déclaré = base + `DECLARED_RESIDUAL_BLEND` × résidu, avec un mélange de 1 : **choix de modélisation déclaré,
  retenu par essais**, justifié par l'amélioration de la perte logarithmique historique hors échantillon (manifeste).
- Chargé par `src/adapters`, SHA-256 vérifié contre le manifeste et contre les deux CSV décidés ; sans artefact, la
  configuration déclarée échoue clairement.

## 5. Corrections de garde

1. L'EO de la surveillance vs mérite est signé : g = TPR centre − TPR éloigné ; OK si −0,09 ≤ g ≤ 0,05.
2. La fuite régionale du jury utilise l'AUC sans orientation, max(AUC, 1 − AUC).
3. L'effet d'équité du jury regarde chaque référence disponible (mérite, corrigé, consensus), alerte sur la pire.
4. Le choix du décalage compare les écarts non arrondis : faisabilité exacte d'abord, puis plus petit |δ|.
5. Au retrait du jury par la garde, le décalage est recalculé sans jury avant le ré-audit.
6. Les limites strictes du consensus sont informatives (conjointement infaisables au budget fixé).
7. Garde forte : planchers relatifs à la règle de consensus **déclarée** (revenu +0,025), plancher absolu 0,90 sur le
   ratio d'impact régional.

## 6. Terminé quand

- Acceptation : constantes de référence égales à `consensus.json` ; poids du revenu déclaré dans la plage des
  examinateurs ; étiquettes = top-k de la règle ; décisions déclarées = top-k de base + résidu ; artefact vérifié.
- `decide` publie la configuration déclarée par le harnais complet.
