import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "audit_rapport.ipynb"

CELLS = [
    ("markdown", """# ÉquiAlgo : rapport d'audit et de correction du biais régional

Ce carnet raconte, dans l'ordre : (a) le diagnostic du biais, (b) la définition de l'équité retenue, (c) la chaîne de correction et ses chiffres finaux, (d) les garde-fous, (e) le front de Pareto, (f) les pistes rejetées, (g) les limites.

Les calculs appellent les fonctions du dépôt (`src.policy`, `src.monitoring`, `src.evaluation`) ; rien n'y est réimplémenté. Les chiffres finaux sont lus dans `decision_record.json`. Tout vient de l'historique des 10 000 demandes : aucun score externe n'entre dans la chaîne. Données synthétiques, graines fixes."""),
    ("code", """import warnings; warnings.filterwarnings('ignore')
import json, time
T0 = time.time()
import numpy as np, pandas as pd
from IPython.display import Image, display
from src.policy import (CommitteeModel, DECLARED_INCOME_WEIGHT, REVIEWER_INCOME_WEIGHTS, auc, budget_share,
                        correction_report, is_remote, reference_weights)
from src.monitoring.checks import proxy_auc

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
history = pd.read_csv('data/donnees_demandes.csv')
batch = pd.read_csv('data/candidats_evaluation.csv')
record = json.load(open('decision_record.json'))
y = history['decision_octroi'].to_numpy()
remote = is_remote(history)
share = budget_share(history)
print(f"{len(history):,} dossiers historiques, {remote.sum():,} en région éloignée ; part historique d'octrois {share:.2%}")"""),
    ("markdown", """## (a) Diagnostic : le comité pénalise la région

Le modèle du comité (régression logistique sur 8 critères légitimes + indicateur « région éloignée ») sert **au diagnostic**, et à estimer le rapport heures / cote R. `correction_report` donne la pénalité, son intervalle de confiance par rééchantillonnage et le test d'égalité des pentes entre régions."""),
    ("code", """report = correction_report(history, share)
rates = pd.Series(y).groupby(remote).mean()
print(f"Taux historique d'octroi : centre {rates[0]:.1%}, éloigné {rates[1]:.1%}")
print(f"Pénalité régionale : {report.penalty:.2f} logit, IC 95 % [{report.penalty_ci[0]:.2f} ; {report.penalty_ci[1]:.2f}] (cotes x {np.exp(report.penalty):.2f})")
print(f"Test des mêmes pentes : khi² = {report.slope_test_statistic:.2f}, ddl = {report.slope_test_df}, p = {report.slope_test_p:.2f}"
      "  -> p élevé : une pénalité plate, pas des règles différentes par région")"""),
    ("markdown", """### Variables proxys de la région

Pouvoir de chaque variable pour prédire « région éloignée » (AUC, 0,5 = aucun, 1 = parfait). Le code postal est encodé par son taux éloigné sur une moitié de l'échantillon et évalué sur l'autre."""),
    ("code", """half = np.random.default_rng(0).random(len(history)) < 0.5
postal_rate = pd.Series(remote[half]).groupby(history['code_postal_3'][half].to_numpy()).mean()
postal_score = history['code_postal_3'][~half].map(postal_rate).fillna(0.5).to_numpy()
columns = {'code postal (3 car.)': None, 'distance au campus': 'distance_domicile_campus_km',
           'heures de travail': 'heures_travail_semaine', 'revenu familial': 'revenu_familial_estime',
           'cote R': 'cote_r_equivalent'}
rows = {}
for label, column in columns.items():
    score = postal_score if column is None else history[column].to_numpy()
    truth = remote[~half] if column is None else remote
    a = auc(truth, score)
    rows[label] = max(a, 1 - a)
proxies = pd.Series(rows, name='AUC région').sort_values(ascending=False)
print(proxies.round(3).to_string())
print(f"\\nAUC (validation croisée) du modèle sur les trois variables de score (cote R, revenu, heures) : {proxy_auc(history):.3f}")"""),
    ("markdown", """Retirer la colonne région ne suffit donc pas : code postal et distance l'encodent presque parfaitement ; les heures et le revenu en portent une grande part.

### Même cote R, issue différente"""),
    ("code", """band = history[(history['cote_r_equivalent'] > 28) & (history['cote_r_equivalent'] <= 30)]
by_group = band.groupby(is_remote(band))['decision_octroi'].agg(['mean', 'size'])
by_group.index = ['Centre', 'Éloignée']
by_group['mean'] = (by_group['mean'] * 100).round(1)
print("Cote R dans ]28 ; 30] :")
print(by_group.rename(columns={'mean': "% d'octrois", 'size': 'dossiers'}).to_string())"""),
    ("markdown", """### Le revenu : une pénalité régionale cachée

Les candidats éloignés ont un revenu (log) plus bas. Comme le comité récompense le revenu, cette récompense désavantage mécaniquement la région. On mesure, **sur l'échelle du comité** (logit), la contribution moyenne de chaque critère à l'écart centre / éloigné, et on la compare à la pénalité explicite."""),
    ("code", """committee = CommitteeModel().fit(history, y)
from src.policy.schema import COMMITTEE_FEATURES, committee_features
Z = committee.scaler_.transform(committee_features(history))
coef = dict(zip(COMMITTEE_FEATURES, committee.lr_.coef_[0][:-1]))
shift = {name: Z[remote == 1, i].mean() - Z[remote == 0, i].mean() for i, name in enumerate(COMMITTEE_FEATURES)}
penalty = abs(committee.remote_penalty_)
table = pd.DataFrame({'décalage éloigné - centre (écart-types)': shift, 'contribution (logit)': {n: coef[n] * shift[n] for n in coef}})
table['part de la pénalité explicite'] = table['contribution (logit)'] / penalty
print(table.loc[['log_revenu', 'heures_travail', 'cote_r']].round(3).to_string())
print(f"\\nPénalité explicite : {committee.remote_penalty_:.2f} logit. Le revenu coûte {coef['log_revenu'] * shift['log_revenu']:.2f} logit aux candidats éloignés, "
      f"soit {abs(coef['log_revenu'] * shift['log_revenu']) / penalty:.0%} de la pénalité. Les heures les aident de {coef['heures_travail'] * shift['heures_travail']:+.2f} logit.")"""),
    ("markdown", """## (b) Définition de l'équité : cinq examinateurs, un consensus

Cinq examinateurs indépendants (mérite, besoin, juridique, processus de données, équité régionale ; `docs/reviews/`) ont proposé chacun une règle de référence. Le consensus (`docs/reviews/CONSENSUS.md`) : cote R d'abord, heures en petit crédit, première génération 0 ; **égalité des chances, pas parité des taux**.

Le revenu est le seul point de désaccord : mérite, juridique et régional lui donnent 0, besoin −0,05, processus de données +0,19. Le poids retenu, **+0,025 (en unités de cote R)**, est un **choix de modélisation déclaré, retenu par essais, dans la plage de désaccord des examinateurs** (−0,05 à +0,19). L'amplitude des heures n'est pas choisie : elle est dérivée de l'ajustement du comité à chaque exécution."""),
    ("code", """weights = pd.DataFrame(reference_weights(committee)).loc[['cote_r', 'heures_travail', 'log_revenu', 'premiere_generation']]
print("Poids des règles de référence (R = 1) :")
print(weights.round(4).to_string())
print(f"\\nRapport heures / R (dérivé de l'historique) : {weights.loc['heures_travail', 'consensus']:.4f}")
print(f"Poids déclaré du revenu : {DECLARED_INCOME_WEIGHT} ; plage des examinateurs : {min(REVIEWER_INCOME_WEIGHTS.values())} à {max(REVIEWER_INCOME_WEIGHTS.values())}")"""),
    ("markdown", """## (c) Chaîne de correction et décision finale

```
historique (10 000) -> validation -> modèle du comité (diagnostic : pénalité, IC, rapport heures / R)
   -> base = règle de consensus : R + (heures/R dérivé) x heures + 0,025 x log revenu  (unités de cote R)
   -> résidu TabM appris sur l'historique (Kaggle), artefact vérifié par SHA-256 ; mélange déclaré = 1
   -> score = base + résidu ; les k meilleurs candidats (k = part historique x n)
   -> jury (panel des cinq règles) et raisonnement en MODE AUDIT : votes, traces, contradictions enregistrés, aucun échange
   -> audit : surveillance + garde du jury + garde du consensus + garde forte
   -> au plus un décalage borné (|d| <= 0,10) ou BLOCAGE
   -> garde de sortie -> PUBLICATION ou BLOCAGE
```

Chiffres de la décision publiée (`decision_record.json`) :"""),
    ("code", """regions = ['Montreal', 'Capitale-Nationale', 'Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
monitoring = {c['name']: c for c in record['verdicts'][0]['checks']}
print(f"Statut : {record['status']} ; octrois {record['grants']:,} sur {record['applicants']:,} ({record['grants'] / record['applicants']:.2%}) ; décalage {record['offset']:+.3f}")
print(f"Configuration : {record['config']['name']} ; mélange du résidu {record['config']['residual_blend']}")
print(f"Taux par région : " + ', '.join(f"{r} {record['region_rates'][r]:.1%}" for r in regions))
print(f"Jury (mode audit) : {record['jury']['triggered']} cas examinés, {record['jury']['overturned_out']} échanges contestés, aucun appliqué")
print(f"Raisonnement : {record['deliberation']}")
for name in ['demographic parity gap (centre vs remote)', 'impact ratio, lowest/highest region', 'opportunity gap vs merit reference']:
    c = monitoring[name]; print(f"  {name}: {c['value']:+.3f}  [{c['status']}]  {c['detail']}")"""),
    ("markdown", """## (d) Garde-fous

1. **Surveillance** : budget, parité, ratio d'impact, écart d'opportunité **signé** contre le mérite (plage −0,09 à +0,05), écart contre le comité corrigé, intersections, dérive.
2. **Garde du jury** : qualité des jurés, fuite régionale (sans orientation), effet d'équité sur mérite / corrigé / consensus, volume d'échanges, accord.
3. **Garde du consensus** : bandes de taux et limites issues des cinq examinateurs ; une ALERTE bloque.
4. **Garde forte** : plancher absolu 0,90 et plancher relatif à la règle de consensus déclarée sur le ratio d'impact régional, écart des sous-groupes, écart de mérite sur cinq régions, écart maximal à une référence, coût du revenu."""),
    ("code", """def guard_table(checks):
    return pd.DataFrame([{'contrôle': c['name'], 'valeur': round(c['value'], 4), 'seuil': c['threshold'], 'statut': c['status']} for c in checks])

pd.set_option('display.max_colwidth', 70)
layers = [('1. Surveillance', record['verdicts'][0]), ('2. Garde du jury', record['jury_guard']),
          ('3. Garde du consensus', record['consensus_guard']), ('4. Garde forte', record['strong_guard'])]
for title, layer in layers:
    print(f"\\n{title} : statut global {layer['status']}")
    print(guard_table(layer['checks']).to_string(index=False))"""),
    ("markdown", """## (e) Front de Pareto

Axes principaux : **écart d'égalité des chances moyen** contre les cinq références des évaluateurs (à minimiser) et **accord moyen** avec elles (à maximiser). Les références sont construites sur l'ajustement du comité du jeu d'entraînement, appliquées au jeu de test au même budget (10 partitions). Le résidu TabM n'existe que pour les 4 000 candidats : le point « déclaré » du front est la base de consensus en mode audit, sans résidu."""),
    ("code", """pareto = pd.read_csv('resultats_pareto.csv')
cols = ['method', 'rate_centre', 'rate_remote', 'eo_gap_reviewers_mean', 'agree_reviewers_mean', 'eo_gap_consensus', 'g_merit',
        'eo_gap_corrected', 'eo_gap_merit', 'acc_historical', 'pareto']
key = [m for m in pareto.method if m.startswith('declared') or m in ('validator jury', 'income-blind validator jury', 'committee-corrected, no jury', 'production RF at budget')]
print(pareto[pareto.method.isin(key)][cols].round(3).to_string(index=False))
print(f"\\nPoints sur le front de Pareto principal : {', '.join(pareto[pareto.pareto].method)}")
display(Image('pareto_front.png'))"""),
    ("markdown", """**L'accord avec le consensus du point déclaré est en partie circulaire** : il optimise la cible qu'on utilise pour le mesurer. D'où l'importance des cinq références prises séparément (accord moyen) et des limites déclarées de la garde du consensus.

## (f) Pistes rejetées"""),
    ("code", """rejected = pd.DataFrame([
    ('Modèles plus gros, MLP', "La cible est un seuil linéaire : jurés entraînés AUC ≈ 1,0 sur l'étiquette, ils répètent le modèle principal (1 échange sur 293 révisions)."),
    ('Gros modèles contrefactuels sur les vraies décisions du comité', "Validation croisée 5 plis : aucun ne bat la régression logistique (perte log 0,2597 contre 0,2623 à 0,2734) ; le comité est linéaire."),
    ('Axes « mérite » / « effort »', "Reparamétrage d'une même régression (coefficients implicites 9,504 / 1,382 contre 9,506 / 1,382 ; Δp max 5e-4)."),
    ('« Plan équitable » (projection sur le noyau)', "Retire surtout les heures (u : R −0,144, revenu −0,465, heures +0,874) ; accord avec les étiquettes 0,997 → 0,974."),
    ('Harnais par étapes (stepwise) et veto à double jury', "Écartés : complexité supplémentaire sans gain démontré sur l'historique ; code retiré de la version publiée."),
    ('Ancien jury validateur', "35 échanges : écart EO de 0,002 à 0,016 (désormais bloqué par la garde du consensus)."),
    ("Seuil d'heures à 10 h", "Non identifié (retenu dans 48 % des rééchantillonnages) ; remplacé par un crédit linéaire."),
], columns=['Piste', 'Preuve du rejet'])
for _, r in rejected.iterrows():
    print(f"- {r['Piste']}\\n    {r['Preuve du rejet']}")"""),
    ("markdown", """## (g) Limites

- **Référence cachée inconnue.** Les « méritants » sont des hypothèses issues des examens ; l'accord avec les cinq règles est un substitut, non une mesure.
- **Désaccord sur le revenu.** Mérite, juridique, régional : 0 ; besoin : −0,05 ; processus de données : +0,19. Le poids déclaré (+0,025) et le mélange du résidu (1) sont des choix de modélisation déclarés, retenus par essais, à faire valider par un comité humain.
- **Le résidu TabM améliore la perte logarithmique historique de façon minime** (voir `models/tabm_residual/manifest.json`) : l'étiquette historique mesure les décisions du comité, pas le mérite.
- **Données synthétiques** ; pénalité estimée sur l'historique, à ré-estimer à chaque cycle ; le modèle du comité est logistique, pas le comité réel.
- Fondements juridiques (Charte québécoise art. 10, 86 ; Loi 25 art. 12.1) à vérifier avant citation."""),
    ("code", """print(f"Durée totale : {time.time() - T0:.0f} s")"""),
]


def build(target: Path, execute: bool) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}
    notebook.cells = [nbformat.v4.new_markdown_cell(text) if kind == "markdown" else nbformat.v4.new_code_cell(text)
                      for kind, text in CELLS]
    if execute:
        NotebookClient(notebook, timeout=900, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate audit_rapport.ipynb from the repository state.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-execute", action="store_true", help="write the notebook without running it")
    arguments = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    build(arguments.out, not arguments.no_execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
