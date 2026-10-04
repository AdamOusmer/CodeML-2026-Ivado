# Constats et correction (configuration déclarée : base de consensus + résidu TabM, audit du jury)

Public : jury. Chiffres sur les 10 000 demandes historiques et les 4 000 candidats évalués, sauf mention. Les chiffres
du résultat final viennent de `decision_record.json` (acceptation 79/79, `scripts/acceptance.py`). Toute la
chaîne est construite à partir de l'historique : aucune valeur n'est réglée sur un score externe.

## 1. Diagnostic : le comité pénalise la région

- **Pénalité.** Le modèle logistique du comité (8 critères + région éloignée) donne un terme « éloigné »
  de −1,90 logit (IC 95 % [−2,07 ; −1,73] ; test des mêmes pentes p = 0,82 : une pénalité plate, pas des règles
  différentes par région). Taux historique d'octroi : centre 48,4 % contre régions éloignées 27,3 %.
- **Même cote R, issue différente.** À R entre 28 et 30 : 70,9 % d'octrois au centre contre 40,8 % en région éloignée.
- **Proxys de la région** (éloignée = Bas-Saint-Laurent, Côte-Nord, Gaspésie) : code postal 1,0, distance 0,998,
  heures travaillées 0,805, revenu 0,69, cote R 0,56 (AUC). Retirer la colonne région ne suffit donc pas.
- **Le revenu est une pénalité cachée.** Le revenu (log) des candidats éloignés est inférieur de 0,68 écart-type.
  Dans le comité, la récompense du revenu coûte en moyenne −0,54 logit aux candidats éloignés, soit **29 %** de la
  pénalité explicite. Les heures travaillées les aident (+0,80 logit, 42 % de la pénalité).

## 2. Définition de l'équité retenue

Égalité des chances (« equal opportunity ») : à mérite égal, même probabilité d'obtenir la bourse, quelle que soit la
région. Ce n'est pas la parité des taux : la cote R des candidats éloignés est plus basse de 0,66 point, ce qui justifie
légitimement environ 7 points d'écart de taux (R seul : centre 42,8 % / éloigné 35,8 %).

Le « mérite » est défini par **consensus** de cinq examinateurs indépendants (mérite, besoin, juridique, processus de
données, équité régionale ; `docs/reviews/CONSENSUS.md`). Les **signes de légitimité sont déclarés par des humains**
(R +, heures +, première génération 0) ; l'**amplitude des heures vient des données** (rapport heures/R = 0,1835, dérivé
de l'ajustement du comité à chaque exécution, non réglé à la main).

Le revenu est le seul désaccord : mérite, juridique, régional 0 ; besoin −0,05 ; processus de données +0,19. Les données
seules ne le tranchent pas. Le poids retenu, **+0,025 en unités de cote R**, et le mélange du résidu, **1**, sont des
**choix de modélisation déclarés, retenus par essais, dans la plage de désaccord des examinateurs** (0 à +0,19). Ils
sont définis chacun en un seul endroit de `src/policy`.

## 3. Correction

1. **Diagnostic** : le modèle du comité reste ajusté pour estimer la pénalité et le rapport heures/R ; il n'est pas la cible.
2. **Base** : règle de consensus `z(R) + 0,1835 × z(heures) + 0,025 × z(log revenu)` en unités de cote R ; les étiquettes
   d'entraînement sont les k meilleurs sur l'historique (k = round(part × n), part = 39,94 % lue dans les données).
3. **Résidu TabM** : un TabM borné (entrées : cote R et heures seulement) est appris sur l'historique hors machine
   (`kaggle/train.py`, 5 plis × 3 graines, pénalité choisie par la perte logarithmique hors échantillon) et livré comme
   artefact `models/tabm_residual/`, vérifié par SHA-256 (manifeste et données décidées). Score = base + 1 × résidu.
   La perte logarithmique historique hors échantillon passe de 0,25884 (modèle linéaire figé) à 0,25869 : un gain
   minime mais de bon sens ; la grille de monotonie en R et en heures est respectée (`manifest.json`).
4. **Les k meilleurs** reçoivent la bourse.
5. **Jury et raisonnement en mode audit** : le panel des cinq règles de référence vote sur les cas limites et le
   raisonnement trace chaque candidat examiné ; votes, traces et contradictions sont enregistrés, **aucun échange ni
   déplacement n'est appliqué**.
6. **Quatre garde-fous** (détail dans `MONITORING_PLAN.md`) : surveillance (EO signé), garde du jury, garde du consensus,
   garde forte ; au plus un décalage borné (|δ| ≤ 0,10) ; garde de sortie ; publication ou BLOCAGE.

## 4. Résultats (4 000 candidats)

Publié, 1 598 octrois (39,95 %), décalage 0. Jury en mode audit : 200 cas examinés, 14 échanges contestés, 0 appliqué ;
raisonnement : 5 déplacements proposés, 44 contradictions conservées, 0 appliqué.

| Indicateur | Comité (historique) | Maintenant |
|---|---|---|
| Taux centre / éloigné | 48,4 % / 27,3 % | **40,2 % / 39,6 %** |
| Écart de parité | 0,21 | 0,006 |
| Écart EO signé vs mérite (centre − éloigné) | n.d. | −0,065 (OK : plage −0,09 à +0,05) |
| Écart d'opportunité vs règle de consensus | n.d. | 0,001 |
| Ratio d'impact plus basse/plus haute région | n.d. | 0,916 (Bas-Saint-Laurent 37,9 % ; Gaspésie 41,4 %) |
| Accord moyen / minimal avec les cinq règles | n.d. | 0,980 / 0,946 (processus de données) |
| Pire écart de sous-groupe (programme, première génération) | n.d. | 0,039 |
| Garde forte | n.d. | OK (ratio d'impact 0,916 ≥ plancher ; coût du revenu 3,6 % de la pénalité retirée, plafond 5 %) |

Lecture de l'EO signé : −0,065 signifie que les candidats éloignés méritants (R seul) sont un peu **plus** servis que
ceux du centre, effet voulu du crédit d'heures ; la limite de dépassement est 0,09.
Avertissements (WARN) : écart vs comité corrigé 0,046 (attendu : cette référence garde la récompense du revenu, que
quatre examinateurs sur cinq rejettent) ; part des cas examinés où les jurés se divisent : 0,78 (WARN au-delà de 0,5).

Front de Pareto (10 partitions 70/30 de l'historique ; le résidu n'existe que pour les candidats, le point « déclaré » est
donc la base en mode audit) : écart EO moyen contre les cinq références 0,017 et accord moyen 0,983, contre 0,064 et
0,948 pour le comité sans pénalité (jury de validation : 0,073 et 0,952) ; la base déclarée est sur le front.

## 5. Expériences rejetées

| Piste | Preuve du rejet |
|---|---|
| Modèles plus gros, MLP | la cible est un seuil linéaire : jurés entraînés AUC ≈ 1,0 sur l'étiquette, ils répètent le modèle principal (1 échange sur 293 révisions) |
| Gros modèles contrefactuels (splines, boosting 2 000 arbres, forêt 1 000 arbres, MLP 256-256-128) sur les vraies décisions du comité | validation croisée 5 plis : aucun ne bat la régression logistique (perte log 0,2597 contre 0,2623 à 0,2734) ; le comité est linéaire |
| Axes « mérite » / « effort » (`merit_effort_axes`) | reparamétrage d'une même régression (coefficients implicites 9,504 / 1,382 contre 9,506 / 1,382 ; Δp max 5e−4) |
| « Plan équitable » (`fair_plane`, projection sur le noyau) | retire surtout les heures (u : R −0,144, revenu −0,465, heures +0,874) ; accord avec les étiquettes 0,997 → 0,974 |
| Harnais par étapes (`stepwise`) et veto à double jury | écartés pour la complexité qu'ils ajoutent ; leur code et leurs contrôles sont retirés de cette version |
| Ancien jury validateur | 35 échanges : écart EO de 0,002 à 0,016 (désormais bloqué par la garde du consensus) |
| Seuil d'heures à 10 h | non identifié (retenu dans 48 % des rééchantillonnages) ; remplacé par un crédit linéaire |
| Normalisation du revenu par région | supprime le signal régional (AUC 0,693 → 0,502) mais le revenu n'est pas un critère de la référence : retiré de la cible plutôt que normalisé |

## 6. Limites et incertitudes

- **Référence cachée inconnue.** Les « méritants » sont des hypothèses issues des examens ; l'accord moyen avec les cinq
  règles est un substitut, non une mesure. Si la référence est celle du processus de données (revenu +0,19), le
  consensus ne s'accorde avec elle qu'à environ 95 %.
- **Désaccord sur le revenu.** Les données seules ne fixent pas ce poids : +0,025 et le mélange 1 sont des choix de
  modélisation déclarés, à faire valider par un comité humain. Coût en équité mesuré sur les données : revenu = 3,6 % de la
  pénalité retirée, garde du consensus OK, ratio d'impact régional 0,916.
- **Résidu TabM.** Le gain de perte logarithmique est minime et l'étiquette historique mesure les décisions du comité, pas le
  mérite. La monotonie n'est vérifiée que sur une grille numérique, pas globalement.
- Mesure en partie circulaire pour le point déclaré : l'accord avec le consensus optimise la cible qui sert à le mesurer ;
  d'où les cinq références prises séparément.
- Données synthétiques ; pénalité estimée sur l'historique, à ré-estimer à chaque cycle.
- Le revenu pèse peu dans le score, mais ses effets passent encore par R et les heures (proxys résiduels).
- Fondements juridiques « à vérifier » avant citation : Charte québécoise art. 10 (« condition sociale »), art. 86 ;
  Loi 25 art. 12.1.

## 7. Reproduction

Valeurs lues dans `decision_record.json` (verdicts, jury_guard, consensus_guard, strong_guard) et `docs/reviews/*.md`.
Relance du lot : `OMP_NUM_THREADS=2 .venv/bin/python -m src.main decide --plain --no-log-file`. Entraînement du résidu :
`kaggle/train.py` (GPU Kaggle gratuit) ; le carnet Kaggle se génère par `kaggle/build_notebook.py`. Le rapport
`audit_rapport.ipynb` se régénère par `scripts/build_audit_report.py`.
