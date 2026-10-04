# Chaîne de décision

Vue d'ensemble de `python -m src.main decide`. Les deux jurys tournent en mode audit : ils votent et tracent, sans
modifier les décisions.

```mermaid
flowchart LR
    H[("Historique")]
    B[("Lot<br/>4 000 candidats")]

    subgraph MODEL["Modèle principal"]
        direction TB
        BASE["Base déclarée<br/>R + heures + revenu"]
        RES["+ 2,5 × résidu TabM<br/>entraîné sur l'historique"]
        TOPK["Top 1 598"]
        BASE --> RES --> TOPK
    end

    subgraph JURIES["Deux jurys (audit)"]
        direction TB
        J1["Jury 1<br/>règles des 5 examinateurs"]
        J2["Jury 2<br/>raisonnement"]
        J1 --> J2
    end

    subgraph HARNESS["Harnais"]
        direction TB
        G1["Surveillance EO"]
        G2["Garde du jury"]
        G3["Garde du consensus"]
        G4["Garde forte"]
        G5["Contrôle de sortie"]
        G1 --> G2 --> G3 --> G4 --> G5
    end

    PUB[("predictions.csv")]
    BLOCK["BLOCK"]

    H --> MODEL
    B --> MODEL
    MODEL --> JURIES --> HARNESS
    HARNESS -- vert --> PUB
    HARNESS -- alerte --> BLOCK

    classDef box fill:#FFFFFF,stroke:#14213D,stroke-width:1px,color:#14213D
    classDef data fill:#2E6DB4,stroke:#14213D,color:#FFFFFF
    classDef guard fill:#FDF1E7,stroke:#C2611C,color:#14213D
    classDef stop fill:#C2611C,stroke:#7A3A0E,color:#FFFFFF
    class BASE,RES,TOPK,J1,J2 box
    class H,B,PUB data
    class G1,G2,G3,G4,G5 guard
    class BLOCK stop
    style MODEL fill:#EEF1F4,stroke:#2E6DB4,color:#14213D
    style JURIES fill:#EEF1F4,stroke:#2E6DB4,color:#14213D
    style HARNESS fill:#EEF1F4,stroke:#C2611C,color:#14213D
    linkStyle default stroke:#5A6B85,stroke-width:1.5px
```

| Étape | Rôle | Action possible |
|---|---|---|
| Base déclarée | z(cote R) + 0,185·z(heures) + 0,025·z(log revenu), pénalité régionale du comité (−1,90 logit) retirée | — |
| Résidu TabM | 4 réseaux entraînés sur l'historique (Kaggle), artefact `models/tabm_residual_ensemble_rh` | — |
| Top 1 598 | budget historique 39,94 % des 4 000 candidats | — |
| Jury 1 | 5 règles des examinateurs (`docs/reviews/consensus.json`) votent sur les cas limites, quorum 0,8 | vote seulement |
| Jury 2 | raisonnement sur cote R, heures, consensus, jusqu'à 3 rondes | trace seulement |
| Surveillance EO | écart signé vs mérite dans −0,09..+0,05 | `ADJUST_OFFSET` (\|δ\| ≤ 0,10) |
| Garde du jury | effet du jury sur l'équité ≤ 0,01 | `REVERT_JURY` |
| Garde du consensus | limites tirées des 5 examinateurs | `BLOCK` |
| Garde forte | ratio d'impact ≥ 0,90, sous-groupes, 5 régions, coût du revenu ≤ 5 % | `BLOCK` |
| Contrôle de sortie | identifiants, nombre d'octrois | `BLOCK` |

Sorties publiées : `predictions.csv`, `decision_record.json`, `explanations.csv` (3 facteurs par personne, Loi 25).
Sur le lot actuel : aucun décalage, aucun retrait de jury, toutes les gardes au vert, statut `published`.
