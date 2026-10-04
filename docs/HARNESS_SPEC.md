# Spécification du harnais de décision (v6, pipeline « historique seulement »)

Contrat du contrôleur qui enveloppe le pipeline d'équité, décide un lot, l'audite et refuse de publier une décision
dangereuse. « DOIT » = contrat. « ÉCART : » = le code diffère du contrat ; le code fait foi.

## 1. Objectifs et non-objectifs

Objectifs
- Décisions respectant le budget pour un lot, sous une seule configuration déclarée.
- Ne publier que les lots dont l'audit final n'est pas en ALERTE.
- Enregistrement rejouable de ce qui a été décidé et pourquoi.
- `model_corrige.py` et `src/main.py` sont de minces racines de composition au-dessus de modules partagés.

Non-objectifs : aucune validation humaine dans le chemin de décision ; aucun changement de politique automatique au
moment de la décision, sauf le décalage borné `ADJUST_OFFSET` ; aucun apprentissage en ligne (I7) ; aucun réglage en
direct (I8).

## 2. Provenance : tout vient de l'historique

Aucune valeur de la chaîne ne provient d'un score externe ni d'une expérience qui en dépend. Les entrées sont les deux CSV fournis. Les constantes déclarées sont de deux sortes :

- **dérivées des données à chaque exécution** : la pénalité régionale, la part d'octrois et, pour le diagnostic, le
  rapport heures / cote R du comité (≈ 0,1835) ;
- **choix de modélisation déclarés, retenus par essais, dans la plage de désaccord des examinateurs** (0 à +0,19 pour le
  revenu) : le poids des heures (`DECLARED_HOURS_WEIGHT` = 0,185, proche du 0,1835 du comité), le poids du revenu
  (`DECLARED_INCOME_WEIGHT` = +0,025, en unités de cote R) et le mélange du résidu (`DECLARED_RESIDUAL_BLEND` = 2,5).
  Ces constantes sont dans `src/policy`, en un seul endroit chacune.

Les poids du revenu des cinq examinateurs (`REVIEWER_INCOME_WEIGHTS` : mérite 0, besoin −0,05, juridique 0, processus de
données +0,19, régional 0) sont lus dans `docs/reviews/consensus.json` ; un contrôle d'acceptation les compare au fichier.

## 3. Pipeline (l'ordre est normatif)

1. **Prétraitement** : `validate_frames` ; le score ne lit que `cote_r`, `log_revenu`, `heures_travail`. La région reste
   pour l'audit, jamais pour le score.
2. **Comité (diagnostic)** : `CommitteeModel` ajusté sur l'historique (8 critères + indicateur « éloigné »). Il donne la
   pénalité, son IC et le rapport heures / cote R. `Config.target = "consensus"` : les étiquettes d'entraînement sont les
   `k` meilleurs de la règle de consensus sur l'historique (`k = round(part × n)`).
3. **Base** : score de consensus en unités de cote R, `z(R) + 0,185 × z(heures) + 0,025 × z(log revenu)`,
   normalisé par la moyenne et l'écart-type de l'historique. Une régression logistique à une variable, ajustée sur ce
   score et les étiquettes de consensus, ne sert qu'à exprimer le score en probabilité (déclencheur `low_confidence`,
   jurés, explications).
4. **Résidu TabM** (`Config.residual_blend`, déclaré à 2,5 ; `Config.residual_artefact`) : artefact `models/tabm_residual_ensemble_rh/` (`residuals.csv` :
   `id_candidat, residual_rsd` ; `manifest.json`), moyenne de 4 réseaux TabM (R + heures) appris sur l'historique seulement ; le manifeste liste les 4 membres, leurs SHA-256 sources, leurs gains hors échantillon et leur monotonie. Score final =
   base + mélange × résidu, en unités de cote R. L'artefact est lu par `src/adapters` ; le SHA-256 de `residuals.csv`
   est comparé au manifeste, et les SHA-256 des deux CSV décidés aux `input_hashes` du manifeste ; toute divergence, un
   identifiant manquant ou une valeur non finie lève `InputError` (sortie 1, aucun fichier). Sans artefact, la
   configuration déclarée échoue clairement ; les autres configurations tournent.
5. **Proposition** : les `k = round(part × n)` meilleurs candidats (tri stable).
6. **Jury en mode audit** (`JurySettings.audit_only`) : le panel des cinq règles de référence vote sur les cas
   déclenchés (`near_cutoff`, `low_confidence`, `disagreement`) ; votes, déclencheurs et échanges contestés sont
   enregistrés, **aucun échange n'est appliqué**.
7. **Raisonnement en mode audit** (`ReasoningSettings`, `apply_moves=False`) : pour chaque candidat examiné, une trace
   variable par variable (cote R, heures, règle de consensus) est écrite dans `explanations.csv` ; les contradictions
   sont comptées dans `deliberation`, aucune décision ne bouge.
8. **Garde-fous** puis publication ou BLOCAGE (section 6).

Règle du budget : `part = moyenne(decision_octroi)` lue dans les données (39,94 %), dans `BUDGET_BOUNDS = (0,36 ; 0,44)`
sinon `InputError`. `decide` lève aussi `InputError` pour un `id_candidat` manquant ou dupliqué.

## 4. Découpage en modules

| Chemin | Rôle |
|---|---|
| `src/policy/regions.py`, `schema.py`, `core.py` | régions, critères, allocation, `budget_share`, `eo_gap` |
| `src/policy/references.py` | poids déclarés du revenu, règles des examinateurs, étiquettes de consensus, poids dérivés |
| `src/policy/label_correction.py` | `CommitteeModel`, correction des étiquettes, rapport de pénalité |
| `src/policy/jury.py`, `jurors.py`, `reasoning.py` | jury (avec `audit_only`), jurés modèles, raisonnement |
| `src/policy/models.py` | `Config`, `DECLARED_CONFIG`, `FairPipeline`, `reference_labels` |
| `src/policy/thresholds.py` | bandes et limites des examinateurs (`docs/reviews/CONSENSUS.md`) |
| `src/monitoring/checks.py` | contrôles de surveillance (EO signé, dérive) |
| `src/harness/` | `controller.decide`, `postprocessing`, `consensus` (garde du consensus), `jury_guard`, `strong_guard`, `reasoning_gate`, `output_guard`, `record` |
| `src/evaluation/` | candidats du front de Pareto, tuner hors ligne |
| `src/adapters/files.py` | toute l'E/S : lecture des CSV, de l'artefact résidu (`read_residual`), écriture du dossier |
| `src/pipelines/decision.py` | composition : lecture, validation, résidu, décision, écriture ; branche Pareto concurrente |
| `kaggle/` | entraînement du résidu TabM (`train.py`), générateur de carnet Kaggle |
| `scripts/acceptance.py`, `scripts/build_audit_report.py` | contrôles d'acceptation, génération de `audit_rapport.ipynb` |

`src/policy/__init__.py` n'importe pas sklearn à l'import : il exporte par `__getattr__` (PEP 562).

## 5. Frontières

Chaque paquet expose son API dans `__init__.py` ; on importe `from src.<paquet> import nom`, jamais
`src.<paquet>.<module>` hors du paquet. Sens des dépendances :

```
main.py, model_corrige.py -> pipelines, adapters, harness, evaluation, monitoring, preprocessing, policy, common
pipelines -> adapters, harness, evaluation, preprocessing, policy, common
adapters  -> harness (types), common
harness   -> monitoring, explain, policy, common
monitoring, evaluation, explain, preprocessing -> policy (preprocessing aussi common)
policy    -> numpy, pandas, scipy, scikit-learn seulement
```

Les E/S de fichiers ne sont permises que dans `src/adapters/`, `src/main.py`, `src/common/logging` et
`src/preprocessing/validation.py`. Appliqué par `scripts/acceptance.py` (analyse AST, prouvée par injection).

## 6. Garde-fous et actions

Quatre couches de garde (détail dans `MONITORING_PLAN.md`) : contrôles de surveillance (EO signé vs mérite, dérive),
garde du jury, garde du consensus, **garde forte** (consensus tout OK ; ratio d'impact régional ≥ max(0,90 ; règle de
consensus déclarée − 0,012) ; écart de sous-groupe, écart de mérite sur cinq régions et écart à chaque référence au plus
égaux à ceux de la règle de consensus déclarée + 0,012 (`TOLERANCE`, tolérance déclarée de la garde forte pour les régressions par rapport à la règle de consensus) ; coût du revenu ≤ 5 % de la pénalité retirée). La règle de
consensus de comparaison est la règle **déclarée** (revenu +0,025), recalculée sur l'historique à chaque exécution.

| Action | Paramètres | Quand |
|---|---|---|
| `SELECT_CONFIG` | `{config}` | une fois, en premier |
| `ADJUST_OFFSET` | `{offset, moved, offset_moved, alerts}` | ALERTE corrigible de la surveillance ou du consensus, `fit_offset` ≠ 0 |
| `REVERT_JURY` | `{checks, offset}` | ALERTE de la garde du jury : jury retiré, décalage recalculé |
| `BLOCK` | `{checks, suggestion}` | verdict final, garde du consensus ou garde forte en ALERTE, ou problème de sortie |

```
AJUSTER -> DÉCIDER(décalage 0) -> AUDIT --OK/WARN--> PUBLIER
                                 '--ALERTE corrigible--> ADJUST_OFFSET -> AUDIT --OK/WARN--> PUBLIER
                                 '--autre ALERTE-------> BLOCK             '--ALERTE--> BLOCK
```

- Au plus un décalage par lot (|δ| ≤ `OFFSET_BOUND` = 0,10, grille de 41 valeurs). Une ALERTE après correction mène à
  BLOCK (sortie 3), sans politique de repli, hors `REVERT_JURY`.
- `fit_offset` choisit par la clé `(max(violation, 0), |δ|, −δ)` ; une métrique non calculable est une ALERTE non
  corrigible.
- La garde de sortie (`output_guard`) passe en dernier : identifiants, valeurs binaires, exactement `k` octrois.
- Un lot bloqué n'écrit jamais `predictions.csv` (I5) ; `decision_record.json` et `explanations.csv` sont toujours écrits.
- En mode audit, `jury_moved_ids` est vide et `jury.applied` vaut `false` ; l'effet d'équité et le volume d'échanges de
  la garde du jury mesurent les échanges appliqués.

## 7. Invariants

| Id | Invariant | Appliqué par |
|---|---|---|
| I1 | octrois = round(part × n), 0,36 ≤ part ≤ 0,44 | `budget_share`, `allocate`, contrôle de budget |
| I2 | le score ne lit jamais région, code postal, distance ; le résidu ne prend que R et les heures | `FairPipeline`, entraînement du résidu |
| I3 | la région n'entre que par `ADJUST_OFFSET` (|δ| ≤ 0,10, enregistré) | `OFFSET_BOUND`, `fit_offset` |
| I4 | même historique + lot + artefact => mêmes décisions, scores, enregistrement | aucun aléa dans `decide`, tris stables |
| I5 | un lot bloqué n'écrit pas `predictions.csv` | `write_decision` |
| I6 | toute action est une `ActionKind` présente dans `record.actions` | `controller.decide` |
| I7 | les décisions ne servent jamais de données d'entraînement | `FairPipeline.fit(history)` |
| I8 | le tuner ne modifie jamais la configuration vivante | sortie CSV seulement |
| I9 | le résidu est vérifié par SHA-256 contre le manifeste et les deux CSV décidés | `read_residual` |

## 8. Contrat de surveillance

`run_checks(history, batch, decisions, reviewed=None, eo_gap_alert=EO_GAP_ALERT, *, corrected_enforced=True,
graph=MONITORING, workers=4) -> list[Check]` ; `Check.excess` est la distance au-delà de la limite d'ALERTE.
Statut = maximum des contrôles (OK < WARN < ALERT). Le contrôle d'EO vs mérite est signé, g = TPR(centre) − TPR(éloigné),
en ALERTE hors [−0,09 ; +0,05]. L'écart vs comité corrigé n'est que consultatif quand la cible est « consensus »
(cette référence conserve la récompense du revenu). Les références viennent de `policy.reference_labels` : `corrected`,
`merit`, `consensus`, les cinq examinateurs, et `historical` si le lot est étiqueté.

## 9. Évaluation (hors ligne)

- `pareto` : `default_candidates()` sur K partitions 70/30 stratifiées (10 par défaut) : RF de production, retrait des
  proxys, ThresholdOptimizer, balayage ExpGrad, balayage de la pénalité du comité, balayage de la bande du jury, jury de
  validation, sans revenu, et la **base de consensus en mode audit** (sans résidu : le résidu n'existe que pour les
  candidats).
- `tune` : parcourt `SEARCH_SPACE` (bandes × quorums × jurés × mélange, plus le panel de consensus et sa variante en
  mode audit) ; `worst_case = min(score_r)` sur les références ; sortie `resultats_tuner.csv`. Ni `tune` ni `pareto`
  n'alimente `decide`.

## 10. Explications et enregistrement

`explain` produit : `id_candidat, decision, proposed_decision, score, final_rank, merit_vote, model_vote, offset,
validated, trigger_reasons, juror_votes, jury_outcome` (`not_reviewed | confirmed | contested_unpaired | contested_out |
contested_in` en mode audit), `reasoning_outcome`, `reasoning_trace`, `factor_1..3`. Avec le résidu, les facteurs sont les
contributions de la base (cote R, heures, revenu) et le terme `residual_rsd`, en unités de cote R.

`DecisionRecord` : configuration (dont `residual_blend`), part, statut, décisions, scores, décalage, verdicts, actions,
`output_issues`, identifiants déplacés, explications, taux par région, `input_hashes` (SHA-256 des deux CSV et de
`tabm_residual_ensemble_rh/residuals.csv`), `jury`, `strong_guard`, `deliberation`, `label_correction`, `training_labels`.

## 11. Points d'entrée et codes de sortie

- `check-data`, `monitor`, `decide --history --batch --out-dir [--config NOM] [--residual-dir DIR] [--json]`, `pareto`,
  `tune`.
- Sortie : 0 OK ou publié ; 3 ALERTE de surveillance ou lot bloqué ; 1 erreur (dont artefact résidu absent ou
  incohérent) ; 2 arguments invalides ; 130 interruption.
- `decide(history, batch, input_hashes, config=DECLARED_CONFIG, residual=None, ...)` est pur : aucune E/S.

## 12. Contrôles d'acceptation

`OMP_NUM_THREADS=2 uv run python scripts/acceptance.py` : 79 contrôles, tous PASS. Ils couvrent le budget,
les doublons, les alertes corrigibles et non corrigibles, la dérive, le rejeu, l'invariance régionale, les frontières de
paquets, le jury (échanges symétriques, déterminisme), la correction des étiquettes, la garde de sortie, la garde du
jury, la garde du consensus, la garde forte, le raisonnement, ainsi que les contrôles propres à cette version :
`residual_artefact` (SHA-256 contre le manifeste et les données ; valeurs, hachage et lot altérés refusés ; quatre membres avec gain hors échantillon positif et monotonie GRIDPASS ; SHA-256 sources déclarés),
`residual_required` (la configuration déclarée échoue clairement sans artefact, les autres tournent),
`declared_pipeline` (poids du revenu dans la plage des examinateurs ; décisions = top-k de base + résidu ; mode audit
sans échange ni déplacement ; résidu aligné par identifiant), `consensus_constants` (poids du revenu des examinateurs égaux
à `consensus.json`) et `no_derived_constants` (aucune constante ou référence d'origine externe dans `src`, `kaggle`,
`models`). Les contrôles de mécanique du harnais décident avec « consensus panel » ; ceux du décalage forcé avec
« income-blind, no jury ».

## 13. Risques ouverts

- La référence cachée est inconnue ; l'équité mesurée est relative aux règles des cinq examinateurs.
- Le poids du revenu et le mélange du résidu sont des choix de modélisation déclarés : à valider par un comité humain.
- Chaque membre du résidu améliore à peine la perte logarithmique historique (voir le manifeste) : l'étiquette historique mesure les
  décisions du comité, pas le mérite.
- Les votes en percentile dépendent de la composition du lot.
- Un nœud corrompu (longueur ou NaN) lève avant la construction du BLOCK : échec fermé (sortie 1), pas un BLOCK enregistré.
