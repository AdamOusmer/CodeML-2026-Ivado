# Plan de surveillance en production

Système automatisé : aucune file de révision humaine dans le chemin de décision. Les humains agissent sur un `BLOCK`
et lors des revues planifiées. Sources de vérité : `src/monitoring/checks.py`, `src/harness/jury_guard.py`,
`src/harness/consensus.py`, `src/harness/postprocessing.py`, `src/harness/controller.py`, `src/policy/thresholds.py`
`src/harness/strong_guard.py` (le code prime sur la documentation). Configuration déclarée : base de consensus + résidu TabM, jury et raisonnement en mode audit (`docs/CONSENSUS_TARGET_SPEC.md`).

## 1. Les quatre couches de garde

| Couche | Module | Question posée | Effet d'une ALERT |
|---|---|---|---|
| 1. Contrôles de surveillance | `src/monitoring/checks.py` (`run_checks`) | budget, équité (EO signé vs mérite), dérive des données | corrigible : décalage ; sinon BLOCK |
| 2. Garde du jury | `src/harness/jury_guard.py` (`jury_guard`) | le jury aide-t-il ou nuit-il ? | `REVERT_JURY` (jury retiré), jamais BLOCK à lui seul |
| 3. Garde du consensus | `src/harness/consensus.py` (`consensus_guard`) | la décision respecte-t-elle les bandes et limites des cinq examinateurs ? | corrigible : décalage ; sinon BLOCK |
| 4. Garde forte | `src/harness/strong_guard.py` (`strong_guard`) | la décision reste-t-elle au moins aussi équitable que la règle de consensus déclarée ? | BLOCK |

Statuts : OK < WARN < ALERT ; le statut global est le maximum. Une métrique non calculable (NaN) devient une ALERT non
corrigible. Les niveaux WARN des couches 1 (EO) et 3 se calculent par `warn_level(règle, limite) = règle + 0,9 × (limite −
règle)` (`thresholds.py`, `WARN_FRACTION = 0,9`), où « règle » est la valeur de la règle consensus elle-même sur le lot :
une décision qui fait mieux que la règle des examinateurs ne déclenche rien.

### 1.1 Couche 1 : contrôles de surveillance (`checks.py`)

| Contrôle (nom dans le code) | WARN | ALERT | Corrigible |
|---|---|---|---|
| `grant rate within budget` | — | hors [36 %, 44 %] (`BUDGET_BOUNDS`) | non |
| `demographic parity gap (centre vs remote)` | > 0,10 | — | — |
| `impact ratio, lowest/highest region` | < 0,80 | — | — |
| `opportunity gap vs merit reference` (signé g = TPR centre − TPR éloigné ; mérite = meilleurs R au même budget) | hors [−`overshoot_warn`, `gap_warn`] (voir `warn_level` ; valeurs calculées sur le lot) | hors [−0,09 ; +0,05] (`max_eo_overshoot_vs_merit`, `EO_GAP_ALERT`) | oui |
| `opportunity gap vs corrected committee` | > 0,03 | > 0,05 seulement si la cible est « corrected » ; en « consensus » : WARN seul (`corrected_enforced = False`) | oui si appliqué |
| `opportunity gap vs human-reviewed sample` | > 0,03 | > 0,05 | hors `decide` : seul `monitor --reviewed` l'exécute |
| `largest intersectional gap` | > 0,15 | — | — |
| `proxy drift: region predictability (AUC change)` | > +0,05 | > +0,05 (pas de palier WARN) | non |
| `feature drift, max PSI` (R, log revenu, heures, par groupe) | > 0,10 | > 0,25 | non |
| `categorical drift, max PSI` (programme, première génération, par région) | > 0,10 | > 0,25 | non |

Le signe de g compte : un g négatif (régions éloignées mieux servies que le mérite seul) est attendu à cause du crédit
d'heures ; la règle consensus affiche elle-même environ −0,063. Un plafond bilatéral de 0,05 échouerait aux règles des
examinateurs (CONSENSUS.md). Validation des trames (`validate_frames`) : toute erreur arrête avant décision (code 1).

### 1.2 Couche 2 : garde du jury (`jury_guard.py`)

| Contrôle | WARN | ALERT |
|---|---|---|
| `juror quality` (AUC hors échantillon des jurés modèles contre leurs étiquettes) | < 0,90 | < 0,85 |
| `juror region leakage` (AUC régional sans orientation, max(AUC, 1 − AUC), juré − modèle principal) | > 0,03 | > 0,05 |
| `jury fairness effect` (EO décisions − EO propositions, pire des références mérite / corrigée / consensus) | > 0,005 | > 0,01 |
| `jury swap volume` (échanges / candidats) | > 3 % | > 5 % |
| `juror agreement: split share of triggered cases` (part des cas déclenchés où les jurés se divisent) | > 0,5 | — |

Dans la configuration déclarée, les jurés sont les cinq règles de référence (sans modèle entraîné) : les contrôles de
qualité et de fuite ne s'appliquent qu'aux jurés modèles (`model_jurors`). Un jury sans juré renvoie OK. Le jury est en
mode audit : l'effet d'équité et le volume d'échanges mesurent les échanges **appliqués** (zéro) ; les échanges contestés
sont comptés dans `jury.overturned_out` du dossier.

### 1.3 Couche 3 : garde du consensus (`consensus.py`, constantes dans `thresholds.py`)

| Contrôle | ALERT (limites `LIMITS`) | Limite stricte (information seulement) |
|---|---|---|
| Taux d'octroi global (non corrigible) / centre / éloigné | [0,36 ; 0,44] / [0,38 ; 0,43] / [0,35 ; 0,42] | [0,38 ; 0,42] / [0,42 ; 0,43] / [0,36 ; 0,38] |
| Écart EO vs mérite, désavantage éloigné | > 0,05 | 0,04 |
| Écart EO vs mérite, dépassement (avantage éloigné) | > 0,09 | 0,07 |
| Écart EO vs règle consensus | > 0,06 | 0,03 |
| Écart de parité | > 0,08 | 0,06 |
| Ratio d'impact éloigné/centre | < 0,85 | 0,90 |
| Écart intersectionnel (cellules ≥ 100) | > 0,10 | 0,08 |
| Accord moyen avec les cinq règles (non corrigible) | < 0,93 | 0,96 |
| Accord minimum avec une règle (non corrigible) | WARN < 0,92, ALERT < 0,90 (`MIN_AGREEMENT_*`) | — |
| Forme des heures : plus grand saut entre fenêtres de 2 h (par décile de R) | WARN > 1,4 × saut de la règle consensus, ALERT > 1,7 × (`JUMP_*_RATIO`) | — |
| Forme des heures : plus grande baisse quand les heures montent | WARN > baisse max des règles + 0,02 (`DROP_SLACK`), ALERT > 3 × (`DROP_ALERT_RATIO`) | — |

Les limites strictes ne bloquent jamais (conjointement infaisables au budget fixé) ; elles sont notées dans `detail`.
Les limites EO vs consensus (0,06) et d'accord moyen (0,93) sont des plafonds tirés des cinq règles des examinateurs
(CONSENSUS.md, « Choice of the two evidence-based limits »).

### 1.4 Couche 4 : garde forte (`strong_guard.py`)

Compare la décision à la règle de consensus **déclarée** (revenu +0,025), recalculée sur le lot et l'historique.

| Contrôle | ALERT |
|---|---|
| contrôles de la garde du consensus non OK | au moins un |
| ratio d'impact régional (plus basse / plus haute région) | < max(0,90 ; règle − 0,01) |
| pire écart de sous-groupe (programme, première génération) | > règle + 0,01 |
| écart de mérite sur les cinq régions | > règle + 0,01 |
| pire écart à une référence au-delà de la règle de consensus | > 0,01 |
| coût du revenu en part de la pénalité retirée | > 5 % |

## 2. Qui déclenche quoi (`postprocessing.py`, `controller.decide`)

| Situation | Action | Consigne dans `actions` |
|---|---|---|
| Toutes les ALERT des couches 1 et 3 sont corrigibles | `fit_offset` : décalage régional unique parmi 41 valeurs de [−0,10 ; 0,10] (`OFFSET_GRID`), écarts non arrondis : faisabilité exacte d'abord, puis plus petit \|δ\| ; ré-audit | `ADJUST_OFFSET` |
| Au moins une ALERT non corrigible (budget, dérive PSI / AUC, accord, forme des heures) | aucun décalage ; BLOCK | `BLOCK` |
| Aucun décalage de la grille ne réduit la violation (δ = 0 optimal) | BLOCK (« no offset in the allowed grid lowers the gap ») | `BLOCK` |
| Garde du jury en ALERT | jury désactivé, décalage recalculé sans jury, ré-audit des couches 1 et 3 | `REVERT_JURY` (repli déclaré, `HARNESS_SPEC.md`) |
| ALERT restante après correction (couche 1, 3 ou 4), ou rejet par la garde de sortie (k octrois, identifiants) | BLOCK : `decision_record.json` + `explanations.csv`, `predictions.csv` intact | `BLOCK` |
| WARN seulement | publication ; WARN journalisés | — |

Au plus un décalage par lot. Une ALERT de la garde du jury ne bloque pas par elle-même : elle retire le jury (qui ne
peut alors plus aggraver l'équité) ; c'est l'audit sur la décision publiée qui peut bloquer. Statut `blocked` si
verdict final ALERT, garde du consensus ALERT, garde forte ALERT ou problème de sortie.

## 3. Valeurs sur le lot évalué (`decision_record.json`, 4 000 candidats, statut `published`)

Taux d'octroi 0,3995 (1 598), centre 40,2 % / éloigné 39,6 %, décalage 0, jury en mode audit : 200 examinés, 14 échanges
contestés, 0 appliqué.

| Contrôle | Valeur | Statut |
|---|---|---|
| Parité centre / éloigné | 0,0056 | OK |
| Ratio d'impact régions (plus basse / plus haute) | 0,916 | OK |
| EO signé vs mérite | −0,065 | OK (marge de 0,025 avant le dépassement de 0,09) |
| EO vs comité corrigé | 0,046 | WARN (attendu : conserve la récompense du revenu) |
| Écart intersectionnel (couche 1) | 0,021 | OK |
| ΔAUC proxys / PSI numérique max / PSI catégoriel max | +0,008 / 0,012 / 0,038 | OK |
| Jury : effet d'équité / volume d'échanges appliqués | 0,0 / 0,0 | OK |
| Jury : part de cas examinés où les jurés se divisent | 0,78 | WARN |
| Consensus : EO vs règle consensus / accord moyen / accord minimum | 0,001 / 0,980 / 0,946 | OK |
| Consensus : intersectionnel / ratio d'impact éloigné-centre | 0,050 / 0,986 | OK |
| Forte : écart de sous-groupe / écart de mérite sur cinq régions / pire écart à une référence / coût du revenu | 0,039 / 0,079 / 0,005 / 3,6 % | OK |

## 4. Surveillance en production : cadence et propriétaires

| Fréquence | Activité | Propriétaire | Preuve |
|---|---|---|---|
| Chaque lot | validation des trames, `decide` (quatre couches), vérification SHA-256 de l'artefact résidu contre le manifeste et les données, archivage de `decision_record.json` (hachages SHA-256 des entrées et du résidu) | responsable modèle | `decision_record.json`, journal `logs/` |
| Chaque lot | **dérive** : PSI numérique et catégoriel, ΔAUC des proxys ; ALERT = BLOCK ; WARN consigné | responsable données | rapport de dérive |
| Chaque BLOCK | diagnostic humain avant remise en service ; aucun repli automatique hors `REVERT_JURY` | responsable modèle, protection des RP consultée | note de diagnostic au dossier |
| **Chaque cycle** (trimestriel) | **ré-estimation du diagnostic du comité** sur l'historique à jour : pénalité (−1,90 logit, IC 95 % [−2,07 ; −1,73]), test des pentes, part du revenu (29 %) ; **ré-entraînement du résidu TabM** (`kaggle/train.py`) et contrôle de son manifeste (perte logarithmique hors échantillon, grille de monotonie, hachages) ; `tune`, `pareto` ; revue des WARN récurrents | responsable modèle ; comité d'éthique approuve | `resultats_tuner.csv`, `resultats_pareto.csv` |
| Trimestriel | `monitor --reviewed` sur un échantillon revu à l'aveugle (EO vs échantillon humain) | audit interne | rapport |
| **Annuel** | **revue des signes de légitimité par un comité humain** (R +, heures +, première génération 0) : confirmer ou modifier ; l'amplitude des heures est ré-dérivée des données ; examen du poids déclaré du revenu (+0,025) et du mélange du résidu (1) dans la plage de désaccord des examinateurs (besoin −0,05, processus de données +0,19) | comité d'éthique ; direction approuve | procès-verbal, `docs/reviews/CONSENSUS.md` mis à jour |
| Annuel | audit externe : reproduction d'un lot (déterminisme, I4), invariants, liste `REMOTE_REGIONS`, seuils de `thresholds.py` | audit interne | rapport d'audit |

Toute modification de code, de seuil, de signe ou d'historique relance `scripts/acceptance.py` (79/79 au dernier passage)
avant mise en service. Les décisions ne sont jamais réinjectées comme données d'entraînement (I7) ; le tuner ne modifie
jamais la configuration vivante (I8). Les seuils du consensus ne changent qu'avec une nouvelle revue d'examinateurs.

### Rôles (R = réalise, A = approuve, C = consulté, I = informé)

| Activité | Resp. modèle | Resp. RP (Loi 25) | Comité d'éthique | Audit interne | Direction |
|---|---|---|---|---|---|
| `decide` par lot, lecture du dossier | R | I | I | I | A |
| Traitement d'un BLOCK | R | C | C | I | A |
| Suivi des WARN | C | I | R | I | A |
| Ré-estimation trimestrielle, `tune` / `pareto` | R | I | A | C | I |
| Revue annuelle des signes de légitimité | C | C | R | C | A |
| Échantillon revu à l'aveugle | C | I | C | R | A |
| Information des personnes, recours | C | R | C | I | A |
| Audit externe annuel | C | C | C | R | A |

## 5. Transparence (Loi 25) et recours

| Élément | Support | Contenu |
|---|---|---|
| Information de la personne : décision fondée exclusivement sur un traitement automatisé | lettre de décision | mention du traitement automatisé et du droit de faire des observations. Loi 25, art. 12.1 (secteur privé) : **à vérifier** ; pour un organisme public, l'art. 65.2 de la Loi sur l'accès : **à vérifier** |
| Principaux facteurs | `explanations.csv` : `factor_1..3` (« caractéristique ±contribution »), `score`, `merit_vote`, `model_vote`, `validated`, `trigger_reasons`, `juror_votes` (vote de chacune des cinq règles), `jury_outcome` (mode audit : `contested_*`), `reasoning_trace`, `offset` | fournis sur demande ; la contribution est le poids de la règle × la valeur standardisée (cote R, heures, revenu), plus le terme du résidu TabM |
| Piste d'audit | `decision_record.json` : configuration, statut, actions, verdicts, `jury_guard`, `consensus_guard`, `moved_ids`, taux par région, hachages des entrées | conservé par lot, rejouable |
| Recours | observations traitées par le responsable de la protection des RP ; réexamen humain hors système | registre des recours, revu au trimestre |

Fondement juridique de l'équité (à vérifier avant citation) : Charte des droits et libertés de la personne, art. 10
(« condition sociale », la région n'est pas un motif énuméré) et art. 86 (programmes d'accès à l'égalité).

## 6. Limites et risques résiduels

| Risque | Atténuation |
|---|---|
| La référence cachée est inconnue ; les « méritants » sont les règles des cinq examinateurs, donc l'équité mesurée est relative à nos hypothèses | échantillon revu à l'aveugle ; audit externe annuel ; revue annuelle des signes |
| Désaccord sur le revenu (besoin −0,05 ; processus de données +0,19) ; le poids déclaré +0,025 et le mélange du résidu sont des choix de modélisation | comité humain, section 4 |
| Résidu TabM : gain de perte logarithmique minime, étiquette historique = décisions du comité, monotonie vérifiée sur grille seulement | blocage si le hachage ne correspond pas ; ré-entraînement et revue du manifeste à chaque cycle |
| Marge de l'EO signé (−0,065 contre −0,09) | suivi de tendance par lot ; WARN avant ALERT |
| Part de division des jurés élevée (0,78) : les cinq règles divergent sur les cas limites ; le jury est en mode audit (aucun échange appliqué) | acceptable : aucun risque d'aggravation ; les échanges contestés restent consignés |
| Portée du décalage faible (\|δ\| ≤ 0,10) : une grande dérive d'équité n'est pas corrigible | BLOCK volontaire |
| Bornes de l'offset tenue par la grille seulement, pas dans `FairPipeline` | divergence documentée, contrôle d'acceptation |
| Région = 5 régions fixes, « éloigné » = 3 régions | revue annuelle de `REMOTE_REGIONS` |
