# Pitch ÉquiAlgo (5 minutes)

Dix diapositives, environ 730 mots prononcés (environ 150 mots par minute). Les figures sont dans
`docs/pitch_figures/` et se régénèrent avec `OMP_NUM_THREADS=4 uv run python scripts/pitch_figures.py`.

| # | Diapositive | Durée | Cumul |
|---|---|---|---|
| 1 | Même cote R, pas la même bourse | 0:30 | 0:30 |
| 2 | Diagnostic : une pénalité plate, mesurée | 0:35 | 1:05 |
| 3 | Retirer la région ne suffit pas | 0:30 | 1:35 |
| 4 | Notre définition : l'égalité des chances | 0:35 | 2:10 |
| 5 | Un modèle appris sur l'historique seulement | 0:40 | 2:50 |
| 6 | Résultat : l'écart régional est refermé | 0:30 | 3:20 |
| 7 | Plus équitable sans perdre en utilité | 0:25 | 3:45 |
| 8 | Les garde-fous ont fait leurs preuves | 0:30 | 4:15 |
| 9 | Ce que nous avons rejeté | 0:20 | 4:35 |
| 10 | Gouvernance, Loi 25 et limites | 0:25 | 5:00 |

---

## 1. Même cote R, pas la même bourse (0:30)

- Comité : 48,4 % d'octrois aux grands centres, 27,3 % aux régions éloignées
- Cote R moyenne : 28,0 contre 27,3, soit 0,66 point seulement
- À cote R 28–30 : 70,9 % contre 40,8 %

**Figure :** `docs/pitch_figures/fig2_meme_cote_r.png`

**Script.** Une institution québécoise accorde ses bourses avec un modèle exact à 88 %. Mais le comité qu'il imite
accorde 48,4 % aux grands centres, et 27,3 % au Bas-Saint-Laurent, sur la Côte-Nord et en Gaspésie. La cote R ne
l'explique pas : elle ne diffère que de 0,66 point. Regardez la courbe : à cote R égale, entre 28 et 30, 70,9 %
d'octrois au centre, 40,8 % en région. Même mérite, pas la même bourse.

## 2. Diagnostic : une pénalité plate, mesurée (0:35)

- Comité reconstitué : régression logistique linéaire, deux groupes (centres, régions éloignées)
- Terme régional : −1,90 logit, IC 95 % [−2,07 ; −1,73]
- Test des mêmes pentes : p = 0,82, une pénalité plate, pas des règles par région
- Modèles plus gros en validation croisée : aucun ne fait mieux

**Figure :** aucune (afficher « −1,90 logit » et « p = 0,82 » en grand)

**Script.** Nous avons reconstitué le comité. Une régression logistique sur les critères légitimes, plus un indicateur
« région éloignée », explique ses décisions : aucun modèle plus gros, forêt, boosting ou réseau, ne fait mieux en
validation croisée. Le terme régional vaut −1,90 logit, avec un intervalle de confiance de −2,07 à −1,73. Le test
d'égalité des pentes donne p = 0,82 : ce n'est pas une règle différente par région. C'est une pénalité plate, ajoutée
par-dessus le mérite, entre deux groupes : les centres et les régions éloignées.

## 3. Retirer la région ne suffit pas (0:30)

- Proxys de la région (AUC) : code postal 1,0, distance 0,998, heures 0,805, revenu 0,69
- Le revenu, plus bas en région, coûte −0,54 logit : 29 % de la pénalité explicite
- Les heures (+0,80) compensent presque la cote R plus basse (−0,90)

**Figure :** `docs/pitch_figures/fig3_proxys_auc.png`

**Script.** Supprimer la colonne région ? Inutile. Le code postal prédit la région parfaitement, la distance à 0,998,
les heures travaillées à 0,805, le revenu à 0,69. Pire : le comité récompense le revenu, plus bas en région. Cette
récompense coûte 0,54 logit aux candidats éloignés, 29 % de la pénalité explicite : une pénalité cachée. Les heures,
elles, les aident de 0,80 logit et compensent presque leur cote R plus basse.

## 4. Notre définition : l'égalité des chances (0:35)

- Pas la parité : l'écart de cote R justifie environ 7 points d'écart de taux
- Égalité des chances : à mérite égal, même probabilité d'obtenir la bourse
- Cinq examinateurs indépendants : mérite, besoin, juridique, données, équité régionale
- Signes fixés par des humains, amplitudes tirées des données

**Figure :** aucune (tableau des cinq examinateurs : R +, heures +, première génération 0, revenu −0,05 à +0,19)

**Script.** Quelle équité viser ? Pas la parité des taux : la cote R plus basse en région justifie légitimement
environ 7 points d'écart. Nous visons l'égalité des chances : à mérite égal, même probabilité d'obtenir la bourse.
Mais qui définit le mérite ? Cinq examinateurs indépendants : mérite, besoin, juridique, processus de données, équité
régionale. Leur consensus : la cote R d'abord, un petit crédit pour les heures travaillées, rien pour la région, la
distance ou la première génération. Les signes sont fixés par des humains ; les amplitudes viennent des données.

## 5. Un modèle appris sur l'historique seulement (0:40)

- Base : R + 0,185 × heures + 0,025 × revenu (réglages déclarés, retenus par essais)
- Résidu : 4 réseaux TabM appris sur les décisions du comité (R et heures seulement), amplifié 2,5
- Les k meilleurs (39,9 %) ; quatre couches de garde + garde de sortie
- Jury et raisonnement en mode audit : votes et traces, aucune décision déplacée

**Figure :** `docs/pitch_figures/fig4_pipeline.png`

**Script.** Notre modèle n'apprend que de l'historique. La base est la règle de consensus : la cote R, plus 0,185 fois
les heures, plus un petit 0,025 pour le revenu ; ce sont des réglages déclarés, retenus par essais. Par-dessus, un
résidu appris sur les décisions du comité par quatre réseaux TabM, entraînés sur GPU Kaggle, qui ne voient que la
cote R et les heures, amplifié par 2,5. Les 39,9 % meilleurs reçoivent la bourse. Ensuite, quatre couches de garde
automatiques, surveillance, jury, consensus et garde forte, peuvent décaler légèrement, retirer le jury ou bloquer
la publication.

## 6. Résultat : l'écart régional est refermé (0:30)

- 1 598 bourses sur 4 000 (39,95 %), dans le budget, décalage 0
- Centres 40,1 %, régions éloignées 39,7 % ; toutes les régions entre 38 % et 41 %
- Ratio d'impact (plus basse / plus haute région) : 0,924 ; écart EO signé vs mérite : −0,060
- Garde forte et garde du consensus : OK

**Figure :** `docs/pitch_figures/fig1_taux_par_region.png`

| | Comité (historique) | Production RF (historique, au budget) | Ancien pipeline | Final |
|---|---|---|---|---|
| Centres / éloignées | 48,4 % / 27,3 % | 47,4 % / 28,7 % | 42,3 % / 36,5 % | **40,1 % / 39,7 %** |
| Écart de parité | 0,211 | 0,187 | 0,057 | **0,005** |
| Ratio d'impact (régions) | 0,543 | n.d. | 0,819 | **0,924** |

**Script.** Sur les 4 000 candidats : 1 598 bourses, dans le budget. Grands centres 40,1 %, régions éloignées 39,7 %.
Du Bas-Saint-Laurent à la Gaspésie, toutes les régions sont entre 38 et 41 % : ratio d'impact 0,92. Face au mérite
pur, les régions méritantes sont même un peu mieux servies, moins 0,06, grâce au crédit d'heures, sous notre limite
de 0,09. Aucun décalage n'a été nécessaire ; garde forte et garde du consensus sont au vert.

## 7. Plus équitable sans perdre en utilité (0:25)

- 10 partitions de l'historique, même budget pour toutes les méthodes
- Écart EO moyen vs 5 références : 0,266 (production) → 0,018 (base déclarée)
- Accord moyen avec les 5 références : 0,911 → 0,983 ; seul point du front de Pareto

**Figure :** `docs/pitch_figures/fig5_pareto.png`

**Script.** Paie-t-on l'équité en utilité ? Non. Sur dix partitions de l'historique, notre base déclarée réduit
l'écart d'égalité des chances moyen à 0,018, contre 0,27 pour le modèle de production, et porte l'accord avec les
cinq références de 91 à 98 %. Elle domine ThresholdOptimizer, ExponentiatedGradient et le retrait des proxys.

## 8. Les garde-fous ont fait leurs preuves (0:30)

- Jury actif testé : 26 échanges proposés sur les cas limites
- Effet mesuré : +0,018 sur l'écart d'égalité des chances (seuil d'ALERTE 0,010)
- Le harnais a retiré le jury automatiquement (`REVERT_JURY`)

**Figure :** `docs/pitch_figures/fig6_jury_revert.png`

**Script.** Ces garde-fous ne sont pas décoratifs. Nous avons testé un jury actif : les cinq règles des examinateurs
votent sur les cas limites et échangent des décisions. Il proposait 26 échanges. La garde a mesuré qu'ils auraient
creusé l'écart d'égalité des chances de 0,018, au-delà du seuil de 0,01. Le harnais a retiré le jury tout seul. Le jury
reste donc en mode audit : il vote et trace chaque cas limite, sans rien changer.

## 9. Ce que nous avons rejeté (0:20)

- Modèles plus gros : le comité est linéaire, aucun gain en validation croisée
- « Plan équitable » par projection : efface surtout les heures, un critère légitime
- Jury actif : rejeté par nos propres garde-fous
- Tuner hors ligne : outil d'analyse, ne modifie jamais la configuration en production

**Figure :** aucune

**Script.** Nous avons aussi rejeté des pistes séduisantes. Des modèles plus gros : le comité est linéaire, ils
n'apprennent rien de plus. Un « plan équitable » par projection : il effaçait surtout les heures, un critère légitime.
Le jury actif, vous venez de le voir.

## 10. Gouvernance, Loi 25 et limites (0:25)

- Chaque lot : audit des quatre couches, dossier archivé, BLOCAGE si ALERTE
- Chaque trimestre : ré-estimation de la pénalité, réentraînement du résidu
- Chaque année : un comité humain revoit les signes de légitimité
- Loi 25 : trois facteurs principaux par personne, droit d'observation, réexamen humain

**Figure :** aucune

**Script.** En production, chaque lot est audité et archivé, et bloqué en cas d'alerte. Chaque trimestre, on
ré-estime la pénalité et on réentraîne le résidu. Chaque année, un comité humain revoit les signes de légitimité.
Chaque personne peut obtenir ses trois facteurs principaux et un réexamen humain, comme le demande la Loi 25. Notre
limite : la référence des juges reste inconnue ; nos mesures sont relatives à nos cinq examinateurs. Merci.

---

## Questions probables du jury

**1. Pourquoi l'égalité des chances plutôt que la parité démographique ?**
Les deux ne tiennent pas ensemble quand les profils diffèrent. La cote R moyenne des régions éloignées est plus basse
de 0,66 point : classer sur R seul donne 42,8 % au centre et 35,8 % en région. Forcer la parité accorderait des bourses
pour l'appartenance régionale ; l'égalité des chances ne retire que l'écart injustifié, à mérite égal.

**2. Pourquoi garder le revenu, même à 0,025 ?**
C'est le seul désaccord entre examinateurs : 0 pour trois d'entre eux, −0,05 pour le besoin, +0,19 pour le processus de
données. 0,025 est un réglage déclaré, retenu par essais, dans cette plage. Son coût est mesuré à chaque lot : 3,6 % de
la pénalité retirée, sous un plafond de 5 % appliqué par la garde forte. Un comité humain le revoit chaque année.

**3. TabM, n'est-ce pas une boîte noire ?**
Il est encadré de quatre façons. Il ne voit que la cote R et les heures, jamais la région, le code postal, la distance
ni le revenu. Son résidu est borné, sa monotonie en R et en heures est vérifiée sur une grille, et l'artefact est scellé
par SHA-256. Chaque explication donne sa contribution séparément de celle de la règle. Son gain hors échantillon est
minime (0,00015 à 0,00030 de perte logarithmique) : la décision reste portée par la règle de consensus lisible.

**4. Comment savez-vous que c'est juste sans les étiquettes cachées ?**
Par plusieurs preuves indépendantes, toutes tirées de l'historique. Le générateur est reconstitué : comité linéaire,
pénalité plate (pentes identiques, p = 0,82). Les chemins causaux sont mesurés : revenu −0,54 logit, heures +0,80,
cote R −0,90. Cinq examinateurs indépendants fixent les références ; la décision s'accorde à 97,6 % en moyenne avec
elles, et au moins à 94,5 % avec chacune. Sur dix partitions validées, l'écart d'égalité des chances moyen passe de
0,266 à 0,018. Enfin, quatre gardes automatiques bloquent tout lot qui dérive. Ce sont des substituts, pas une preuve :
nous le disons.

**5. Et si les données dérivent ?**
Chaque lot calcule la dérive (PSI par groupe, prédictibilité de la région par les variables du score). Au-delà de 0,25
de PSI ou de +0,05 d'AUC, c'est une ALERTE non corrigible : la publication est bloquée et un humain diagnostique. Sur ce
lot, PSI maximal 0,038 et variation d'AUC +0,008. La pénalité est ré-estimée chaque trimestre.

**6. Loi 25 : comment expliquez-vous une décision ?**
`explanations.csv` donne pour chaque candidat le score, ses trois facteurs principaux (par exemple « cote R +0,11 »),
les votes des cinq règles et la trace du raisonnement sur les cas limites. La lettre mentionne la décision automatisée
et le droit de présenter des observations ; un réexamen humain hors système est prévu. Les articles exacts restent à
faire valider par un juriste.

**7. Pourquoi ne pas simplement retirer la région ?**
Parce que les proxys la portent : code postal AUC 1,0, distance 0,998. Les consignes le montrent : retirer la région
fait passer l'écart de parité de 0,188 à 0,181 seulement. Notre score n'utilise ni région, ni code postal, ni
distance ; nous corrigeons la cible elle-même, la récompense du revenu comprise.

**8. Pourquoi faire confiance à vos cinq examinateurs ?**
Ils ont travaillé séparément, avec des mandats opposés (besoin contre processus de données, par exemple). Nous ne
gardons que leur consensus, signe par majorité et poids médian, et leurs désaccords restent publiés. Chaque règle
individuelle sert aussi de référence de contrôle : la décision ne peut pas plaire à une seule. Leurs signes sont revus
chaque année par un comité humain.

**9. Les régions éloignées sont maintenant légèrement avantagées (−0,060). Discrimination inverse ?**
Non : l'écart est mesuré contre la cote R seule. Le crédit d'heures, retenu par les cinq examinateurs, profite
davantage aux régions où l'on travaille plus. La règle de consensus elle-même donne environ −0,063. Une limite de dépassement à
0,09 empêche toute surcorrection.

**10. Les heures ne sont-elles pas elles-mêmes un proxy de la région ?**
Si : AUC 0,805. Mais travailler pendant ses études est un critère de mérite légitime pour les cinq examinateurs, et un
crédit linéaire, sans seuil. Nous ne le retirons pas parce qu'il corrèle avec la région ; nous retirons ce qui pénalise
la région sans justification.

**11. Quelles sont vos limites ?**
La référence des juges est inconnue : nos mesures sont relatives à nos cinq examinateurs. Le poids du revenu et le
mélange du résidu sont des réglages déclarés, retenus par essais. Le gain de TabM est minime. Les données sont
synthétiques et la monotonie n'est vérifiée que sur une grille. Deux avertissements restent ouverts : écart vs comité
corrigé 0,053, attendu puisque cette référence garde la récompense du revenu, et division des jurés sur 74 % des cas
limites.

---

## Sources des chiffres

| Chiffre | Source |
|---|---|
| 48,4 / 27,3 %, R 28–30 70,9 / 40,8 %, AUC des proxys, contributions revenu / heures / R | recalculés sur `data/donnees_demandes.csv` avec `src.policy` (même code que `audit_rapport.ipynb`) |
| −1,90, IC, p = 0,82 | `decision_record.json` → `label_correction` |
| Résultats finaux, gardes, jury en mode audit, avertissements | `decision_record.json` |
| Ancien pipeline 42,3 / 36,5 %, ratio d'impact 0,819 | `decision_record.json` publié au commit `5fe40d6` (jury validateur) |
| Production RF, front de Pareto | `resultats_pareto.csv` (10 partitions de l'historique) |
| Jury actif : 26 échanges, +0,018, `REVERT_JURY` | recalculé par `scripts/pitch_figures.py` (configuration déclarée, jury non audit) |
| Modèles plus gros, plan équitable | `docs/FINDINGS.md`, section 5 |
| Gains TabM, grille de monotonie | `models/tabm_residual_ensemble_rh/manifest.json` |
| 42,8 / 35,8 % (R seul), règles des examinateurs | `docs/reviews/CONSENSUS.md` |
