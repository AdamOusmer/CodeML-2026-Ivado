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
  pénalité explicite. Les chemins légitimes se compensent presque : les heures travaillées les aident (+0,80 logit),
  la cote R plus basse leur coûte −0,90 logit (net −0,10).
- **Reconstitution du générateur.** Deux groupes seulement (terme régional par région, référence Montréal :
  Capitale-Nationale −0,03 ; Bas-Saint-Laurent −1,91 ; Côte-Nord −1,92 ; Gaspésie −1,81) ; critères indépendants à
  l'intérieur de chaque groupe (corrélation maximale 0,014 au centre, 0,030 en région) ; comité linéaire (mêmes pentes,
  p = 0,82 ; aucun modèle plus riche ne bat la régression logistique, section 5).

## 2. Définition de l'équité retenue

Égalité des chances (« equal opportunity ») : à mérite égal, même probabilité d'obtenir la bourse, quelle que soit la
région. Ce n'est pas la parité des taux : la cote R des candidats éloignés est plus basse de 0,66 point, ce qui justifie
légitimement environ 7 points d'écart de taux (R seul : centre 42,8 % / éloigné 35,8 %). Forcer la parité refuserait des
candidats plus méritants pour égaliser des taux (quotas et parité forcée sont des drapeaux rouges pour quatre
examinateurs sur cinq) ; le tort constaté est une pénalité **à mérite égal**, ce que mesure l'égalité des chances, qui
est aussi la métrique de la grille. La parité, le ratio d'impact par région et les écarts intersectionnels restent
surveillés comme indicateurs secondaires (détection d'une sur-correction).

Le « mérite » est défini par **consensus** de cinq examinateurs indépendants (mérite, besoin, juridique, processus de
données, équité régionale ; `docs/reviews/CONSENSUS.md`). Les **signes de légitimité sont déclarés par des humains**
(R +, heures +, première génération 0) ; l'**amplitude des heures** est un choix déclaré, **0,185** en unités de cote R, retenu par essais et
proche du rapport heures/R de l'ajustement du comité (0,1835, recalculé à chaque exécution et consigné dans le dossier de décision).

Le revenu est le seul désaccord : mérite, juridique, régional 0 ; besoin −0,05 ; processus de données +0,19. Les données
seules ne le tranchent pas. Le poids retenu, **+0,025 en unités de cote R** (dans la plage de désaccord des examinateurs, 0 à +0,19), le poids des
heures ci-dessus et le mélange du résidu, **2,5**, sont des **choix de modélisation déclarés, retenus par essais**. Ils
sont définis chacun en un seul endroit de `src/policy`.

## 3. Correction

1. **Diagnostic** : le modèle du comité reste ajusté pour estimer la pénalité et le rapport heures/R ; il n'est pas la cible.
2. **Base** : règle de consensus `z(R) + 0,185 × z(heures) + 0,025 × z(log revenu)` en unités de cote R ; les étiquettes
   d'entraînement sont les k meilleurs sur l'historique (k = round(part × n), part = 39,94 % lue dans les données).
3. **Résidu TabM (ensemble de 4 réseaux)** : quatre TabM bornés (entrées : cote R et heures seulement ; 16×2×128 et
   32×3×256, plafond 0,25 et 0,5) sont appris sur l'historique hors machine (5 plis × 5 graines, pénalité de chaque
   réseau choisie par la perte logarithmique hors échantillon) ; le résidu est la moyenne de leurs résidus par
   candidat, livrée comme artefact `models/tabm_residual_ensemble_rh/`, vérifié par SHA-256 (manifeste, sources et
   données décidées). Score = base + 2,5 × résidu (mélange déclaré, retenu par essais). Chaque réseau améliore
   légèrement la perte logarithmique historique hors échantillon (de 0,00015 à 0,00030) : un gain minime ; la grille de
   monotonie en R et en heures est respectée pour les quatre (`manifest.json`).
4. **Les k meilleurs** reçoivent la bourse.
5. **Jury et raisonnement en mode audit** : le panel des cinq règles de référence vote sur les cas limites et le
   raisonnement trace chaque candidat examiné ; votes, traces et contradictions sont enregistrés, **aucun échange ni
   déplacement n'est appliqué**.
6. **Quatre garde-fous** (détail dans `MONITORING_PLAN.md`) : surveillance (EO signé), garde du jury, garde du consensus,
   garde forte ; au plus un décalage borné (|δ| ≤ 0,10) ; garde de sortie ; publication ou BLOCAGE.

## 4. Résultats (4 000 candidats)

Publié, 1 598 octrois (39,95 %), décalage 0. Jury en mode audit : 200 cas examinés, 26 échanges contestés, 0 appliqué ;
raisonnement : 6 déplacements proposés, 49 contradictions conservées, 0 appliqué.

| Indicateur | Comité (historique) | Maintenant |
|---|---|---|
| Taux centre / éloigné | 48,4 % / 27,3 % | **40,1 % / 39,7 %** |
| Écart de parité | 0,21 | 0,005 |
| Écart EO signé vs mérite (centre − éloigné) | 0,202 (forêt de production au budget) | −0,060 (OK : plage −0,09 à +0,05) |
| Écart d'opportunité vs règle de consensus | n.d. | 0,002 |
| Ratio d'impact plus basse/plus haute région | n.d. | 0,924 (Bas-Saint-Laurent 38,1 % ; Gaspésie 41,2 %) |
| Écart EO moyen / accord moyen avec les cinq règles | 0,267 / 0,904 (forêt de production) | 0,019 / 0,976 (minimum 0,944, processus de données) |
| Pire écart de sous-groupe (programme, première génération) | n.d. | 0,030 |
| Garde forte | n.d. | OK (ratio d'impact 0,924 ≥ plancher ; coût du revenu 3,6 % de la pénalité retirée, plafond 5 %) |

Lecture de l'EO signé : −0,060 signifie que les candidats éloignés méritants (R seul) sont un peu **plus** servis que
ceux du centre, effet voulu du crédit d'heures ; la limite de dépassement est 0,09.
Avertissements (WARN) : écart vs comité corrigé 0,053 (attendu : cette référence garde la récompense du revenu, que
quatre examinateurs sur cinq rejettent) ; part des cas examinés où les jurés se divisent : 0,74 (WARN au-delà de 0,5).

**Front de Pareto de la chaîne déclarée** (`model_corrige.py` → `pareto_front.png`, `resultats_pareto_declare.csv`).
Contrainte d'équité balayée : la part λ de la pénalité régionale du comité retirée du score déclaré (score = base +
2,5 × résidu + (1 − λ) × pénalité × éloigné, pénalité convertie en unités de cote R), 11 réglages de 0 à 100 % ; puis,
à λ = 100 %, le mélange du résidu (0 à 5) et le poids du revenu (−0,05 à +0,19). Même budget pour chaque point.

| Réglage | Écart EO moyen vs 5 références | Accord moyen | EO signé vs mérite | Parité |
|---|---|---|---|---|
| Forêt de production au budget | 0,267 | 0,904 | +0,202 | 0,192 |
| λ = 0 % (pénalité du comité conservée) | 0,223 | 0,925 | +0,143 | 0,159 |
| λ = 50 % | 0,112 | 0,955 | +0,034 | 0,085 |
| **λ = 100 % (déclaré)** | **0,019** | **0,976** | **−0,060** | **0,005** |
| mélange 0 / 0,5 (λ = 100 %) | 0,013 / 0,012 | 0,980 / 0,980 | −0,072 / −0,066 | 0,005 / 0,007 |
| revenu +0,19 (λ = 100 %) | 0,056 | 0,949 | −0,008 | 0,047 |

La contrainte d'équité domine le front : chaque pas de λ réduit l'écart EO et augmente l'accord, sans arbitrage contre
ces références. Le mélange et le revenu ne déplacent le point que d'environ ±0,01 près du bout. Le mélange 2,5 n'est pas
sur le front calculé contre ces références linéaires (écart d'environ 0,007 en EO et 0,005 en accord avec le mélange
0,5) : il est justifié par la perte logarithmique hors échantillon sur l'historique, pas par ce graphique. Le point
déclaré du graphique est identique, décision par décision, à `predictions.csv`. Sur 10 partitions 70/30 de l'historique
(`pareto_comparaison.png`), le même balayage de λ sur la base donne 0,261 → 0,018 en écart EO moyen et 0,915 → 0,982 en
accord ; la base déclarée domine ExpGrad (0,159 à 0,196), ThresholdOptimizer (0,171) et le retrait des proxys (0,142).

## 5. Expériences rejetées

| Piste | Preuve du rejet |
|---|---|
| Modèles plus gros, MLP | la cible est un seuil linéaire : jurés entraînés AUC ≈ 1,0 sur l'étiquette, ils répètent le modèle principal (1 échange sur 293 révisions) |
| Gros modèles contrefactuels (splines, boosting 2 000 arbres, forêt 1 000 arbres, MLP 256-256-128) sur les vraies décisions du comité | validation croisée 5 plis : aucun ne bat la régression logistique (perte log 0,2597 contre 0,2623 à 0,2734) ; le comité est linéaire |
| Axes « mérite » / « effort » (`merit_effort_axes`) | reparamétrage d'une même régression (coefficients implicites 9,504 / 1,382 contre 9,506 / 1,382 ; Δp max 5e−4) |
| « Plan équitable » (`fair_plane`, projection sur le noyau) | retire surtout les heures (u : R −0,144, revenu −0,465, heures +0,874) ; accord avec les étiquettes 0,997 → 0,974 |
| Harnais par étapes (`stepwise`) et veto à double jury | écartés pour la complexité qu'ils ajoutent ; leur code et leurs contrôles sont retirés de cette version |
| Optimiseur robuste de l'écart attendu (expérience hors du code publié) | score attendu sur 35 contre les références des examinateurs : 32,10 en échantillon mais 29,11 en validation « une référence laissée de côté », sous la règle de consensus (29,54) ; gagne 1 pli sur 4 : surajustement |
| Ancien jury validateur | 35 échanges : écart EO de 0,002 à 0,016 (désormais bloqué par la garde du consensus) |
| Jury actif (échanges appliqués) | 26 échanges proposés ; effet sur l'équité +0,018 > 0,01 : `REVERT_JURY` par le harnais |
| Jurés sensibles au résidu | 16 échanges ; effet sur l'équité +0,019 : annulé |
| Jury de 8 modèles TabM (quorum 6/8) | 1 échange ; garde forte en ALERT (pire écart de référence 0,0138 > 0,012) : bloqué |
| Barrière d'équité par paire | 1 échange conservé sur 26 (2 lignes) |
| Conclusion | le jury et le raisonnement restent en mode audit (votes et traces pour la transparence) ; les décisions viennent du modèle entraîné, sous les garde-fous |
| Seuil d'heures à 10 h | non identifié (retenu dans 48 % des rééchantillonnages) ; remplacé par un crédit linéaire |
| Normalisation du revenu par région | supprime le signal régional (AUC 0,693 → 0,502) mais le revenu n'est pas un critère de la référence : retiré de la cible plutôt que normalisé |

## 6. Limites et incertitudes

- **Référence cachée inconnue.** Les « méritants » sont des hypothèses issues des examens ; l'accord moyen avec les cinq
  règles est un substitut, non une mesure. Si la référence est celle du processus de données (revenu +0,19), le
  consensus ne s'accorde avec elle qu'à 94 %.
- **Désaccord sur le revenu.** Les données seules ne fixent pas ce poids : +0,025 et le mélange 2,5 sont des choix de
  modélisation déclarés, à faire valider par un comité humain. Coût en équité mesuré sur les données : revenu = 3,6 % de la
  pénalité retirée, garde du consensus OK, ratio d'impact régional 0,924.
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
