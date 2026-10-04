# ÉquiAlgo — notre solution

Le biais du modèle de production n'est pas du bruit : c'est une pénalité régionale plate (−1,90 logit, IC95 [−2,07 ;
−1,73]) appliquée par-dessus le mérite, et le revenu familial agit en plus comme une pénalité régionale cachée (le
revenu des régions éloignées est plus bas de 0,68 écart-type). Tout ce qui suit est construit à partir de l'historique
des 10 000 demandes.

La décision est le top k (k = taux historique de 39,94 %, lu dans les données) du score `base + résidu` :

- **base** : règle de consensus des cinq examinateurs, en unités de cote R : cote R, un crédit linéaire pour les heures
  (rapport heures/R = 0,1835, dérivé de l'historique à chaque exécution) et un petit poids du revenu (+0,025, choix de
  modélisation déclaré, retenu par essais, dans la plage de désaccord des examinateurs de 0 à +0,19) ;
- **résidu** : un TabM borné appris sur l'historique (`kaggle/train.py`), livré comme artefact vérifié par SHA-256
  (`models/tabm_residual/`), mélangé à 1 (choix déclaré) ;
- **jury et raisonnement en mode audit** : le panel des cinq règles vote et les décisions sont tracées, sans échange ;
- un harnais automatisé audite le résultat (surveillance, garde du jury, garde du consensus, garde forte), applique au
  plus un décalage borné (|δ| ≤ 0,10) ou bloque la publication.

### Résultats (4 000 candidats)

| Mesure | Production (RF) | Déclaré |
|---|---|---|
| Taux d'octroi centres / régions éloignées | 48,4 % / 27,3 % (historique) | 40,2 % / 39,6 % |
| Écart EO signé vs mérite | 0,247 | −0,065 (OK) |
| Ratio d'impact (plus basse / plus haute région) | — | 0,916 |
| Octrois | — | 1 598 (décalage 0) |
| Jury (mode audit) | — | 200 examinés, 14 échanges contestés, 0 appliqué |
| Garde forte | — | OK |
| Contrôles d'acceptation | — | 79/79 |

Alertes résiduelles (WARN) : écart EO vs comité corrigé 0,046, attendu car cette référence garde la récompense du
revenu ; accord des jurés 0,78.

Les références « mérite », « comité corrigé » et « consensus » sont nos propres références approximatives, pas le
standard caché des juges. Voir `docs/FINDINGS.md` pour le diagnostic, les limites et les pistes rejetées.

### Livrables

- `predictions.csv`
- `audit_rapport.ipynb` (généré par `scripts/build_audit_report.py`)
- `model_corrige.py` + `pareto_front.png`
- `docs/MONITORING_PLAN.md`
- `models/tabm_residual/` et `kaggle/` (entraînement du résidu)

### Démarrage rapide

Avec [uv](https://docs.astral.sh/uv/) :

```bash
uv sync
OMP_NUM_THREADS=2 uv run python model_corrige.py
uv run python -m src.main decide --out-dir out
uv run python -m src.main monitor
OMP_NUM_THREADS=2 uv run python scripts/acceptance.py
```

Avec pip :

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

`data/` doit contenir les deux CSV fournis avant de lancer quoi que ce soit. La configuration déclarée lit
`models/tabm_residual/` et échoue clairement si l'artefact manque ou ne correspond pas aux données.

Entraînement du résidu (GPU Kaggle gratuit) : `uv run python kaggle/build_notebook.py` génère un carnet qui exécute
`kaggle/train.py` avec les deux CSV pour seules entrées ; copier `residuals.csv` et `manifest.json` dans
`models/tabm_residual/`.

### Documentation

- [docs/HARNESS_SPEC.md](docs/HARNESS_SPEC.md)
- [docs/JURY_SPEC.md](docs/JURY_SPEC.md)
- [docs/CONSENSUS_TARGET_SPEC.md](docs/CONSENSUS_TARGET_SPEC.md)
- [docs/reviews/CONSENSUS.md](docs/reviews/CONSENSUS.md)
- [docs/PREPROCESSING_SPEC.md](docs/PREPROCESSING_SPEC.md)
- [docs/FINDINGS.md](docs/FINDINGS.md)
- [docs/MONITORING_PLAN.md](docs/MONITORING_PLAN.md)

---

Original challenge brief:

# ÉquiAlgo: fair student financing

Engineering and Computer Science Hackathon 2026. 24-hour challenge.

A Quebec financial institution scores scholarship and student-loan applications
with a machine learning model. It is 88% accurate. An internal audit found it
grants awards to 48.4% of applicants from Montréal and the Capitale-Nationale,
against 27.3% from Bas-Saint-Laurent, Côte-Nord and
Gaspésie–Îles-de-la-Madeleine.

Average R score is 27.3 in the remote regions and 28.0 in the centres. That
accounts for part of the 21-point gap. The rest is unexplained.

Your task: diagnose the bias, correct it, and propose a monitoring plan for
production.

All data is synthetic. The institution is fictional.

## Setup

```bash
git clone <YOUR_REPO_URL>
cd defi-equialgo
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook baseline_model.ipynb
```

Python 3.10 or newer. Run the notebook once before changing anything. It trains
the production model, measures it, and audits it.

## Files

| File | Rows | Contents |
|---|---|---|
| `data/donnees_demandes.csv` | 10,000 | historical applications, with `decision_octroi` |
| `data/candidats_evaluation.csv` | 4,000 | applications to score, no label |
| `baseline_model.ipynb` | | production model, metrics, fairness audit |

### Columns

| Column | Meaning |
|---|---|
| `id_candidat` | identifier, `C000000` format |
| `cote_r_equivalent` | academic performance, R score equivalent, 15 to 40 |
| `programme_etudes` | program of study, 5 categories |
| `region_administrative` | sensitive attribute, Quebec administrative region |
| `code_postal_3` | first three characters of the postal code |
| `revenu_familial_estime` | gross annual household income |
| `heures_travail_semaine` | hours worked per week during studies |
| `distance_domicile_campus_km` | home to campus, km |
| `premiere_generation_universitaire` | 1 if first in family to attend university |
| `decision_octroi` | target, 1 granted, 0 refused |

Region values carry no accents and no spaces around the hyphen: `Montreal`,
`Capitale-Nationale`, `Bas-Saint-Laurent`, `Cote-Nord`,
`Gaspesie-Iles-de-la-Madeleine`.

## Constraints

**Fixed budget.** Your grant rate on the 4,000 evaluation applicants must fall
between 36% and 44%. Outside that range the technical section scores zero.

**`decision_octroi` is not the target.** Judges score against a reference
standard built independently of the historical committee. You do not have it.
The column records what the committee did, and the committee is under audit.

**Deleting `region_administrative` does not work.** Dropping it moves the parity
gap from 0.188 to 0.181. Dropping the postal code as well moves it to 0.173.
Distance, hours worked, household income and postal code all carry regional
information. Section 5 of the notebook measures this.

## Deliverables

A GitHub repository, public or shared with the judges, containing:

- `predictions.csv` at the root. Two columns, 4,000 rows plus a header, values
  0 or 1. The last notebook cell writes a valid example.
- `audit_rapport.ipynb`. Measurement of the bias, your fairness metrics with
  justification, and the proxy variables you found.
- `model_corrige.py` or `.ipynb`. Your mitigation, with a Pareto front plot
  across several settings of the fairness constraint.
- `presentation.pdf`. Support for a five-minute pitch.

```csv
id_candidat,decision_octroi
C000042,1
C000117,0
```

## Scoring

| Section | Points | Judged by |
|---|---|---|
| Diagnostic rigour | 25 | jury |
| Technical solution | 35 | automated scorer |
| Governance and ethics | 25 | jury |
| Pitch and code quality | 15 | jury |

The 35 automated points, both measured against the hidden reference standard:

- Equity, 20 points. Share of the baseline equal-opportunity gap closed. The
  baseline gap is 0.270.
- Utility, 15 points. Agreement with the reference standard, scaled between a
  random budget-respecting draw and a perfect allocation.

Both score zero if the budget constraint is broken.

## Notes

`fairlearn.postprocessing.ThresholdOptimizer` adjusts decision thresholds after
training and runs in seconds. `fairlearn.reductions.ExponentiatedGradient`
retrains under a constraint and takes minutes.

Demographic parity and equal opportunity cannot both hold when the two groups
have different profiles. Choose one and be ready to defend the choice.

A single model is not a Pareto front. Sweep the fairness constraint and plot the
results.

Mentors are available throughout.

## Running our solution

With [uv](https://docs.astral.sh/uv/) (Python version in `.python-version`, lockfile `uv.lock`):

```bash
uv sync
OMP_NUM_THREADS=2 uv run python model_corrige.py
uv run python -m src.main decide --out-dir out
uv run python -m src.main monitor
OMP_NUM_THREADS=2 uv run python scripts/acceptance.py
```

With pip, as in the original setup:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`data/` is not versioned: copy the two supplied CSVs into `data/` before running.

`model_corrige.py` writes `predictions.csv`, `resultats_pareto.csv` and `pareto_front.png`.
`decide` runs the declared consensus panel pipeline and writes the published decisions to `--out-dir`.
`monitor` runs the monitoring checks against the final decisions.
`scripts/acceptance.py` runs the 78 acceptance checks.
