# Plan de surveillance en production

Système entièrement automatisé : aucune file de révision humaine dans le chemin de décision. Les humains agissent
uniquement sur un `BLOCK` et lors des audits planifiés. Source de vérité : `src/monitoring/checks.py`,
`src/harness/controller.py`, `docs/HARNESS_SPEC.md` (le code prime sur la documentation en cas d'écart).

## 1. Objet et périmètre

| Quoi | Quand | Commande | Sortie |
|---|---|---|---|
| Validation des trames (schéma, identifiants, catégories, bornes, codes postaux) | chaque lot, avant toute décision | `decide` (nœud `frame_report`), `check-data` | erreur → arrêt, code 1, aucun fichier ; avertissement → journal |
| Audit d'équité et de dérive (9 contrôles ci-dessous) | chaque lot, avant publication | `decide` (fonction `audit`) | `decision_record.json` (`verdicts`), code 0 publié / 3 bloqué |
| Ré-audit d'un lot déjà décidé, avec ou sans échantillon revu à l'aveugle | audit planifié | `monitor [--reviewed]` | rapport ; code 3 si ALERT |
| Ré-évaluation des configurations (Pareto, tuner) | trimestriel, hors ligne | `pareto`, `tune` | `resultats_pareto.csv`, `resultats_tuner.csv` (jamais lus par `decide`) |

Surveillé : budget d'octroi, écarts centre vs régions éloignées (`REMOTE_REGIONS` = Bas-Saint-Laurent, Côte-Nord,
Gaspésie–Îles-de-la-Madeleine), dérive des caractéristiques de notation (`cote_r`, `log_revenu`, `heures_travail`)
et des variables catégorielles par région. Les colonnes région, code postal et distance servent uniquement à mesurer
et corriger, jamais à noter (invariant I2).

## 2. Tableau des contrôles

Statuts : OK < WARN < ALERT. Le statut global est le maximum. Une métrique non calculable (sous-groupe vide, NaN) devient
une ALERT non corrigible, donc bloque. Seuils tels que codés dans `src/monitoring/checks.py`.

| Métrique (nom dans le code) | Définition | WARN | ALERT | Corrigible | Action automatique | Propriétaire | Action humaine attendue | Cadence |
|---|---|---|---|---|---|---|---|---|
| Validation des trames (`validate_frames`) | colonnes, valeurs manquantes, `id_candidat` (`C\d{6}`, doublons, chevauchement historique/lot), catégories, `code_postal_3` (`[A-Z]\d[A-Z]`), binaires, bornes (R 15–40, revenu > 0, heures 0–168, distance ≥ 0), transformations finies | avertissement journalisé : code postal vu dans plusieurs régions ; lot hors plage historique | toute erreur → `DataValidationError` | non | arrêt avant décision, code 1, rien n'est écrit | responsable données | corriger la source, relancer ; lire les avertissements du journal | chaque lot |
| `grant rate within budget` | part d'octrois du lot | — (pas de palier WARN) | hors `BUDGET_BOUNDS` [36 %, 44 %] | non | BLOCK | responsable modèle | vérifier `budget_share` de l'historique et la taille du lot | chaque lot |
| `demographic parity gap (centre vs remote)` | \|taux centre − taux éloigné\| | > 0.10 | — (WARN seulement) | — | aucune (journal) | comité d'éthique | examen au rapport trimestriel ; décision de politique (art. 86 Charte) si persistant | chaque lot |
| `impact ratio, lowest/highest region` | taux min / taux max parmi les 5 régions | < 0.80 | — | — | aucune (journal) | comité d'éthique | idem ; identifier la région la plus basse (`detail`) | chaque lot |
| `opportunity gap vs merit reference` | \|TPR centre − TPR éloigné\|, méritants = meilleurs scores R au même budget | > 0.03 | > 0.05 (`EO_GAP_ALERT`) | oui | ADJUST_OFFSET (une fois, \|offset\| ≤ 0.10) puis ré-audit ; sinon BLOCK | responsable modèle | si BLOCK : analyser la dérive, envisager la politique « mérite seul » (`MERIT_ONLY_SUGGESTION`) | chaque lot |
| `opportunity gap vs corrected committee` | idem, méritants = règle du comité sans pénalité régionale | > 0.03 | > 0.05 | oui | idem | responsable modèle | idem | chaque lot |
| `opportunity gap vs human-reviewed sample` | idem, méritants = étiquettes `merite` d'une revue à l'aveugle | > 0.03 | > 0.05 | oui (drapeau), mais hors `decide` : seul `monitor --reviewed` l'exécute | aucune automatique | audit interne | constituer l'échantillon revu, lancer `monitor --reviewed`, consigner | trimestriel |
| `largest intersectional gap (centre vs remote)` | pire écart centre/éloigné parmi première génération et terciles de revenu | > 0.15 | — | — | aucune (journal) | comité d'éthique | examen trimestriel du sous-groupe nommé dans `detail` | chaque lot |
| `proxy drift: region predictability (AUC change)` | AUC(lot) − AUC(historique) d'une régression logistique prédisant la région à partir des 3 caractéristiques de notation (CV 3) | > +0.05 (même seuil que l'ALERT : pas de palier WARN effectif) | > +0.05 | non | BLOCK | responsable modèle | les caractéristiques encodent davantage la région : revoir les proxys, ré-estimer la pénalité | chaque lot |
| `feature drift, max PSI` | PSI max sur `cote_r`, `log_revenu`, `heures_travail`, par groupe centre/éloigné, 10 déciles historiques | > 0.10 | > 0.25 | non | BLOCK | responsable données | confirmer la dérive source ; décider d'un rafraîchissement de l'historique (jamais avec les décisions, I7) | chaque lot |
| `categorical drift, max PSI` | PSI max de `programme_etudes` et `premiere_generation_universitaire`, par région | > 0.10 | > 0.25 | non | BLOCK | responsable données | idem ; `detail` nomme la région et la catégorie déplacée | chaque lot |

Règle de correction (`controller.decide`) : une ALERT n'est corrigée que si **toutes** les ALERT sont corrigibles
(`Verdict.correctable`). `fit_offset` choisit sur 41 valeurs de [−0.10, 0.10] l'offset minimisant le pire des deux
écarts d'opportunité ; si le meilleur offset est 0, pas d'`ADJUST_OFFSET`, BLOCK direct
(« no offset in the allowed grid lowers the gap »). Au plus une correction par lot.

## 3. Valeurs actuelles sur le lot évalué (`decision_record.json`, 4 000 candidats, statut `published`)

| Contrôle | Valeur | Seuil | Statut | Détail |
|---|---|---|---|---|
| Validation des trames | 0 erreur | — | OK | historique 10 000 lignes, lot 4 000 |
| Taux d'octroi | 0.3995 (1 598 / 4 000) | 36 %–44 % | OK | budget historique 0.3994 |
| Écart de parité | 0.057 | warn > 0.10 | OK | centre 42.3 %, éloigné 36.5 % |
| Ratio d'impact | 0.819 | warn < 0.80 | OK | Côte-Nord 35.0 % vs Capitale-Nationale 42.8 % |
| Écart d'opportunité vs mérite | 0.011 | warn > 0.03, alert > 0.05 | OK | |
| Écart d'opportunité vs comité corrigé | 0.018 | warn > 0.03, alert > 0.05 | OK | |
| Écart intersectionnel | 0.064 | warn > 0.15 | OK | première génération |
| Dérive des proxys (ΔAUC) | +0.008 | alert > +0.05 | OK | historique 0.845, lot 0.853 |
| PSI numérique max | 0.012 | warn > 0.10, alert > 0.25 | OK | `log_revenu` (centre) |
| PSI catégoriel max | 0.038 | warn > 0.10, alert > 0.25 | OK | `programme_etudes` (Gaspésie), Génie +7.98 pts |

Offset 0 ; actions : `SELECT_CONFIG` seulement ; jury : 207 déclenchés, 36 échanges.

## 4. Boucle de décision automatisée

```
lire CSV ─▶ valider trames ──erreur──▶ ARRÊT (code 1, rien écrit)
               │ ok (avertissements journalisés)
               ▼
   SELECT_CONFIG ─▶ ajuster sur l'historique ─▶ décider (offset 0) ─▶ jury validateur ─▶ AUDIT
                                                                                            │
                     ┌──── OK / WARN ─────────────────────────────────────────────────────────┤
                     │                                                                        │ ALERT
                     │                        toutes corrigibles ? ── non ──▶ BLOCK (code 3)  │
                     │                               │ oui                                    │
                     │                        fit_offset ── 0 ──▶ BLOCK                       │
                     │                               │ ≠ 0                                    │
                     │                        ADJUST_OFFSET ─▶ jury ─▶ AUDIT ── ALERT ──▶ BLOCK
                     │                                               │ OK / WARN
                     ▼                                               ▼
                 PUBLIER : predictions.csv + decision_record.json + explanations.csv
                 (BLOCK : decision_record.json + explanations.csv seulement, predictions.csv intact)
```

| Invariant | Énoncé | Garanti par |
|---|---|---|
| I1 | octrois = round(part × n), 0.36 ≤ part ≤ 0.44 | `budget_share`, `allocate`, contrôle budget |
| I2 | la notation par défaut ne lit jamais région, code postal, distance ; l'offset ne déplace que le classement | `scoring_features`, `FairPipeline.decide` |
| I3 | la région n'entre qu'au travers d'`ADJUST_OFFSET`, \|offset\| ≤ 0.10, consigné (borne tenue par `OFFSET_GRID` seulement, divergence documentée) | `fit_offset`, `DecisionRecord` |
| I4 | même historique + même lot ⇒ décisions, scores et dossier identiques (pas d'horodatage) | aucun aléa, tris stables |
| I5 | un lot bloqué n'écrit jamais `predictions.csv` | `write_decision` |
| I6 | chaque action est un `ActionKind` (`SELECT_CONFIG`, `ADJUST_OFFSET`, `BLOCK`) consigné dans `actions` | `controller.decide` |
| I7 | les décisions ne sont jamais réinjectées comme données d'entraînement | `FairPipeline.fit(history)` |
| I8 | le tuner ne modifie jamais la configuration vivante ; `DECLARED_CONFIG` est éditée par l'équipe | `tune` écrit un CSV, rien ne le relit |

## 5. Rôles et responsabilités (RACI)

| Activité | Responsable modèle | Responsable protection des RP (Loi 25) | Comité d'éthique | Audit interne | Direction |
|---|---|---|---|---|---|
| Exécution de `decide` par lot, lecture du dossier | R | I | I | I | A |
| Traitement d'un BLOCK (diagnostic, remise en service) | R | C | C | I | A |
| Suivi des WARN (parité, impact, intersectionnel) | C | I | R | I | A |
| Rapport d'équité trimestriel | R | C | A | C | I |
| Ré-estimation de la pénalité, `tune` / `pareto`, édition de `DECLARED_CONFIG` | R | I | A | C | I |
| Échantillon revu à l'aveugle, `monitor --reviewed` | C | I | C | R | A |
| Information des personnes, principaux facteurs, recours | C | R | C | I | A |
| Audit externe annuel | C | C | C | R | A |

R = réalise, A = approuve, C = consulté, I = informé.

## 6. Cadence

| Fréquence | Activité | Preuve |
|---|---|---|
| Chaque lot | validation des trames, `decide`, 9 contrôles, archivage de `decision_record.json` (hachages SHA-256 des entrées) | `decision_record.json`, journal `logs/` |
| Chaque BLOCK | diagnostic humain avant toute remise en service ; aucune politique de repli automatique | note de diagnostic jointe au dossier |
| Trimestriel | rapport d'équité (tendance des 9 métriques par lot) ; ré-estimation de la pénalité régionale (−1.90 log-odds, IC 95 % ≈ [−2.07, −1.73]) sur l'historique à jour ; `tune` puis `pareto` ; revue du comité d'éthique | `resultats_tuner.csv`, `resultats_pareto.csv`, `pareto_front.png` |
| Annuel | audit externe : reproduction d'un lot (I4), contrôle des invariants, revue des seuils et de la liste `REMOTE_REGIONS` | rapport d'audit |

## 7. Règles de changement

| Règle | Mécanisme |
|---|---|
| Jamais de ré-entraînement sur ses propres décisions (I7) | l'historique d'entraînement est un fichier séparé ; `decide` ne lit `predictions.csv` d'aucun lot antérieur |
| Les boutons (`removal`, `band`, `conf`, `disagree`, `quorum`, `jurors`, seuils) ne changent que par l'équipe : `tune` → lecture de la table → édition manuelle de `DECLARED_CONFIG` ou des constantes → revue | I8 ; aucune valeur de configuration n'est lue à l'exécution depuis un fichier modifiable |
| Toute modification (code, seuil, configuration, historique) relance `scripts/acceptance.py` (30 contrôles au dernier passage : 30/30) avant mise en service | preuve d'acceptation archivée avec la version |
| Toute dérive d'historique (rafraîchissement) exige un nouveau `check-data`, une ré-estimation de la pénalité et un `pareto` | tableau de la section 6 |
| Pas de changement de politique au moment de la décision hormis `ADJUST_OFFSET` borné | HARNESS_SPEC §1, §6 |

## 8. Transparence et recours

| Élément | Support | Contenu |
|---|---|---|
| Information de la personne : décision fondée exclusivement sur un traitement automatisé | lettre de décision | mention explicite du traitement automatisé et du droit de présenter des observations (Loi 25, art. 12.1 — « à vérifier » : l'institution relevant du secteur public, la disposition applicable pourrait être l'art. 65.2 de la Loi sur l'accès) |
| Principaux facteurs | `explanations.csv` : `factor_1..3` (« caractéristique ±contribution »), `score`, `merit_vote`, `model_vote`, `validated`, `trigger_reasons`, `juror_votes`, `jury_outcome`, `offset` | fournis à chaque personne sur demande ; contributions = coefficient × valeur standardisée, somme vérifiée à 1e-9 |
| Piste d'audit | `decision_record.json` : configuration, part budgétaire, statut, actions, verdicts avant/après correction, `moved_ids`, taux par région, hachages des entrées, comptes du jury | conservé pour chaque lot, rejouable (I4) |
| Recours | observations de la personne traitées par le responsable protection des RP ; réexamen humain hors système (le harnais n'a pas de voie de dérogation individuelle) | registre des recours, revu au trimestre |

## 9. Limites et risques résiduels

| Risque | Nature | Atténuation |
|---|---|---|
| Références de mérite = proxys dérivés du comité audité (règle corrigée, score R) ; pas la référence cachée | les écarts d'opportunité mesurent l'équité contre nos propres hypothèses ; l'objectif du tuner est partiellement circulaire | échantillon revu à l'aveugle (`monitor --reviewed`) ; audit externe annuel |
| Le comité récompense le revenu (× 3.40 d'odds par doublement) et les heures travaillées (× 2.70 pour +5 h) | choix de politique conservé, pas une erreur de modèle ; revenu et heures sont des proxys opposés et doivent rester ensemble | documenté ; à trancher par le comité d'éthique, pas par le modèle |
| Votes du jury par percentile dans le lot | un lot atypique (petit, déséquilibré) déplace les seuils de vote ; la fenêtre de faible confiance ne signale que des refus | contrôles de dérive (PSI, ΔAUC) ; taille minimale de lot à fixer |
| Portée de l'offset faible (+0.10 ≈ 11 octrois éloignés) | une grande dérive d'équité n'est pas corrigible et bloque | BLOCK volontaire : la remise en service passe par les humains |
| Borne \|offset\| ≤ 0.10 tenue par la grille seulement, pas dans `FairPipeline` | un appelant hors `fit_offset` pourrait dépasser | divergence documentée (HARNESS_SPEC I3) ; contrôle d'acceptation |
| Région = 5 régions administratives fixes, « éloigné » = 3 régions | un redécoupage ou une nouvelle région invalide `REMOTE_REGIONS` | revue annuelle de la liste ; validation des catégories bloque toute valeur inconnue |

Écarts code / documentation relevés : `docs/HARNESS_SPEC.md` §7 ne liste pas le contrôle `categorical drift, max PSI`
(présent dans `CHECK_NODES`) ; le contrôle de dérive des proxys a `warn == alert == 0.05`, donc aucun palier WARN
réel ; le contrôle sur échantillon revu n'est jamais exécuté par `decide`, seulement par `monitor --reviewed`.
