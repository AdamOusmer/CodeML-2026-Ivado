import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

REMOTE = {"Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"}
REGIONS = REMOTE | {"Montreal", "Capitale-Nationale"}
FEATURES = ["cote_r_equivalent", "heures_travail_semaine", "revenu_familial_estime"]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def find_data(explicit, filename):
    if explicit:
        return Path(explicit).resolve()
    matches = list(Path("/kaggle/input").rglob(filename))
    if len(matches) != 1:
        raise ValueError(f"Supply the CSV path explicitly: found {len(matches)} copies of {filename}")
    return matches[0]


def validate_data(history, candidates, smoke):
    required = {"id_candidat", "region_administrative", "programme_etudes", "code_postal_3",
                "distance_domicile_campus_km", "premiere_generation_universitaire", *FEATURES}
    for name, frame, rows in (("history", history, 10000), ("candidates", candidates, 4000)):
        if not required.issubset(frame.columns) or frame.columns.duplicated().any():
            raise ValueError(f"{name}: missing or duplicate columns")
        if not smoke and len(frame) != rows:
            raise ValueError(f"{name}: expected {rows} rows, found {len(frame)}")
        if frame.empty or frame.isna().any().any() or frame.id_candidat.duplicated().any():
            raise ValueError(f"{name}: empty data, missing values, or duplicate IDs")
        if not set(frame.region_administrative).issubset(REGIONS):
            raise ValueError(f"{name}: unknown administrative region")
        values = frame[FEATURES].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, 2] <= 0).any():
            raise ValueError(f"{name}: invalid finite numeric inputs or nonpositive income")
        if not frame.cote_r_equivalent.between(15, 40).all() or not frame.heures_travail_semaine.between(0, 168).all():
            raise ValueError(f"{name}: R or hours outside the declared domain")
    if "decision_octroi" not in history or not history.decision_octroi.isin([0, 1]).all():
        raise ValueError("Historical committee labels must be binary")
    if "decision_octroi" in candidates:
        raise ValueError("Evaluation candidates must not contain decision labels")
    if set(history.id_candidat) & set(candidates.id_candidat):
        raise ValueError("Historical and evaluation IDs overlap")
    if not 0.36 <= history.decision_octroi.mean() <= 0.44:
        raise ValueError("Historical budget is outside 36–44%")


def raw_features(frame):
    return np.column_stack([frame.cote_r_equivalent, frame.heures_travail_semaine,
                            np.log(frame.revenu_familial_estime)])


def normalization(frame):
    values = raw_features(frame)
    mean, scale = values.mean(0), values.std(0, ddof=1)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Normalization requires positive finite historical feature scales")
    return mean, scale


def matrix(frame, norm):
    return (raw_features(frame) - norm[0]) / norm[1]


def remote(frame):
    return frame.region_administrative.isin(REMOTE).to_numpy(dtype=float)


def strata(frame):
    return frame.region_administrative + "_" + frame.decision_octroi.astype(str)


def metrics(y, probabilities):
    return {"historical_logloss": float(log_loss(y, probabilities, labels=[0, 1])),
            "historical_brier": float(brier_score_loss(y, probabilities)),
            "historical_auc": float(roc_auc_score(y, probabilities))}


def predict(model, x, offset, cap, device, torch, batch_size):
    model.eval()
    residuals, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tensor = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            residual = cap * torch.tanh(model(tensor).squeeze(-1))
            base = torch.as_tensor(offset[start:start + batch_size], dtype=torch.float32, device=device)
            probabilities.append(torch.sigmoid(base[:, None].double() + residual.double()).mean(1).cpu().numpy())
            residuals.append(residual.mean(1).cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(residuals)


def gradient_grid(model, norm, beta_r, full_scale, cap, device, torch, batch_size):
    r, hours = np.meshgrid(np.linspace(15, 40, 41), np.linspace(0, 168, 41), indexing="ij")
    inputs = (np.column_stack([r.ravel(), hours.ravel()]) - norm[0][:2]) / norm[1][:2]
    model.eval()
    values = []
    for start in range(0, len(inputs), batch_size):
        x = torch.tensor(inputs[start:start + batch_size], dtype=torch.float32, device=device, requires_grad=True)
        residual = cap * torch.tanh(model(x).squeeze(-1)).mean(1)
        derivative = torch.autograd.grad(residual.sum(), x)[0]
        values.append(derivative.detach().cpu().numpy())
    derivative = np.concatenate(values)
    return {"R_values": r.ravel(), "hours_values": hours.ravel(),
            "R": derivative[:, 0] / beta_r,
            "hours": derivative[:, 1] * full_scale[1] / norm[1][1]
            * norm[1][0] / (beta_r * full_scale[0])}


def train_member(args, train, heldout, candidates, seed, fold, penalty, full_scale, deadline, torch, TabM, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    inner_train, inner_valid = train_test_split(np.arange(len(train)), test_size=0.2,
                                               stratify=strata(train), random_state=seed)
    norm = normalization(train.iloc[inner_train])
    z = matrix(train, norm)
    fitted = LogisticRegression(max_iter=3000).fit(
        np.column_stack([z[inner_train], remote(train)[inner_train]]),
        train.decision_octroi.to_numpy()[inner_train])
    beta_r = float(fitted.coef_[0][0])
    if beta_r <= 0:
        raise ValueError("Frozen committee model must have a positive R coefficient")
    offset = fitted.decision_function(np.column_stack([z, remote(train)]))
    model = TabM.make(n_num_features=2, d_out=1, k=args.members, n_blocks=args.blocks,
                      d_block=args.width, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x_tensor = torch.as_tensor(z[inner_train, :2], dtype=torch.float32, device=device)
    offset_tensor = torch.as_tensor(offset[inner_train], dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(train.decision_octroi.to_numpy()[inner_train], dtype=torch.float32, device=device)
    best_state = copy.deepcopy(model.state_dict())
    baseline_valid = metrics(train.decision_octroi.to_numpy()[inner_valid], expit(offset[inner_valid]))["historical_logloss"]
    best_loss, best_epoch, stale, epochs_run = float("inf"), 0, 0, 0
    for epoch in range(args.epochs):
        if time.monotonic() >= deadline:
            raise TimeoutError("Runtime budget reached before completing all folds and seeds")
        model.train()
        ordering = torch.randperm(len(inner_train), device=device)
        for start in range(0, len(ordering), args.batch_size):
            indices = ordering[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            residual = args.residual_cap * torch.tanh(model(x_tensor[indices]).squeeze(-1))
            logits = offset_tensor[indices, None] + residual
            target = y_tensor[indices, None].expand_as(logits)
            loss = loss_fn(logits, target) + penalty * residual.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_p, _ = predict(model, z[inner_valid, :2], offset[inner_valid], args.residual_cap,
                           device, torch, args.batch_size)
        val_loss = metrics(train.decision_octroi.to_numpy()[inner_valid], val_p)["historical_logloss"]
        epochs_run = epoch + 1
        if val_loss < best_loss - args.min_delta:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"seed": seed, "fold": fold, "penalty": penalty, "epoch": epoch + 1,
                              "inner_logloss": val_loss, "best_inner_logloss": best_loss}), flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    outputs = {}
    for name, frame in (("heldout", heldout), ("candidates", candidates)):
        transformed = matrix(frame, norm)
        base = fitted.decision_function(np.column_stack([transformed, remote(frame)]))
        p, residual = predict(model, transformed[:, :2], base, args.residual_cap, device, torch, args.batch_size)
        outputs[name] = {"p": p, "baseline_p": expit(base),
                         "residual_rsd": residual * norm[1][0] / (beta_r * full_scale[0])}
    outputs["gradient_grid"] = gradient_grid(model, norm, beta_r, full_scale, args.residual_cap,
                                              device, torch, args.batch_size)
    checkpoint = args.out_dir / "checkpoints" / f"lambda_{penalty:g}_seed_{seed}_fold_{fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {k: value.detach().cpu() for k, value in best_state.items()},
                "mean": norm[0], "scale": norm[1], "committee_coef": fitted.coef_,
                "committee_intercept": fitted.intercept_, "beta_r": beta_r,
                "full_history_scale": full_scale, "residual_cap": args.residual_cap,
                "tabm_settings": {"k": args.members, "n_blocks": args.blocks, "d_block": args.width,
                                  "dropout": args.dropout}, "seed": seed, "fold": fold, "penalty": penalty}, checkpoint)
    row = {"penalty": penalty, "seed": seed, "fold": fold, "best_epoch": best_epoch,
           "epochs_run": epochs_run, "inner_baseline_logloss": baseline_valid,
           "inner_best_logloss": best_loss, "beta_r": beta_r,
           **metrics(heldout.decision_octroi, outputs["heldout"]["p"]),
           **{f"baseline_{k}": value for k, value in metrics(heldout.decision_octroi,
                                                         outputs["heldout"]["baseline_p"]).items()}}
    np.savez_compressed(checkpoint.with_suffix(".npz"), heldout_ids=heldout.id_candidat.to_numpy(dtype=str),
                        heldout_p=outputs["heldout"]["p"], baseline_p=outputs["heldout"]["baseline_p"],
                        candidate_residual_rsd=outputs["candidates"]["residual_rsd"],
                        grid_R=outputs["gradient_grid"]["R_values"],
                        grid_hours=outputs["gradient_grid"]["hours_values"],
                        residual_derivative_R=outputs["gradient_grid"]["R"],
                        residual_derivative_hours=outputs["gradient_grid"]["hours"])
    pd.DataFrame(outputs["gradient_grid"]).to_csv(checkpoint.with_suffix(".gradient_grid.csv"), index=False)
    del model, optimizer, x_tensor, offset_tensor, y_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TabM residual on the historical committee labels and export it.")
    parser.add_argument("--train")
    parser.add_argument("--candidates")
    parser.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/tabm_residual"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--penalties", default="0.1,1")
    parser.add_argument("--max-minutes", type=float, default=60)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--residual-cap", type=float, default=0.25)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(",")]
    args.penalties = [float(penalty) for penalty in args.penalties.split(",")]
    if args.folds < 2 or min(args.max_minutes, args.epochs, args.patience, args.batch_size,
                            args.members, args.blocks, args.width, args.residual_cap) <= 0:
        parser.error("Fold count and training/runtime parameters must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.penalties)) != len(args.penalties):
        parser.error("Seeds and penalties must be unique")
    if not args.seeds or not args.penalties or min(args.penalties) < 0:
        parser.error("Supply seeds and nonnegative residual penalties")
    if not 0 <= args.dropout < 1 or min(args.learning_rate, args.weight_decay, args.min_delta) < 0:
        parser.error("Dropout must be in [0,1); optimizer parameters and min-delta must be nonnegative")
    return args


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committee_hours_ratio(history, norm):
    z = matrix(history, norm)
    fitted = LogisticRegression(max_iter=3000).fit(np.column_stack([z, remote(history)]), history.decision_octroi)
    return float(abs(fitted.coef_[0][1] / fitted.coef_[0][0]))


def penalty_bundle(history, mean_p, mean_base, residuals, gradients):
    return {"p": mean_p, "baseline_p": mean_base, "score": metrics(history.decision_octroi, mean_p),
            "candidate_residual": np.mean(residuals, axis=0),
            "gradient_R": np.mean([g["R"] for g in gradients], axis=0),
            "gradient_hours": np.mean([g["hours"] for g in gradients], axis=0),
            "grid_R": gradients[0]["R_values"], "grid_hours": gradients[0]["hours_values"]}


def monotonicity_report(bundle, hours_ratio, out_dir):
    derivative_r = 1 + bundle["gradient_R"]
    derivative_hours = hours_ratio + bundle["gradient_hours"]
    if not np.isfinite(derivative_r).all() or not np.isfinite(derivative_hours).all():
        raise ValueError("Nonfinite monotonicity diagnostic; export blocked")
    pd.DataFrame({"R": bundle["grid_R"], "hours": bundle["grid_hours"], "R_derivative": derivative_r,
                  "hours_derivative": derivative_hours}).to_csv(out_dir / "monotonicity_grid.csv", index=False)
    violations = int((derivative_r < -1e-8).sum() + (derivative_hours < -1e-8).sum())
    return {"residual_inputs": ["cote_r_equivalent", "heures_travail_semaine"],
            "committee_hours_ratio": hours_ratio,
            "min_R_derivative": float(derivative_r.min()), "min_hours_derivative": float(derivative_hours.min()),
            "grid_violations": violations, "grid_status": "GRIDFAIL" if violations else "GRIDPASS",
            "scope": "NUMERICAL_GRID_ONLY; not a global proof between grid points"}


def base_manifest(args, device, torch, train_path, candidate_path):
    return {"status": "running", "model": "tabm.TabM", "device": str(device),
            "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "tabm", "numpy", "pandas", "scikit-learn")},
            "input_hashes": {"history": sha256_file(train_path), "candidates": sha256_file(candidate_path)},
            "seeds": args.seeds, "folds": args.folds, "smoke_only": args.allow_cpu_smoke,
            "label_warning": "Historical decision_octroi measures committee decisions, not independent deservingness. "
                             "OOF improvement is diagnostic, not hidden accuracy.",
            "validation": f"Outer {args.folds}-fold historical OOF; outer and inner splits stratified by region and "
                          "committee label. Normalizers and frozen linear model fit inner training rows only; inner 20% "
                          "early stopping. Penalty selected by OOF historical log loss, with selection optimism.",
            "residual_scaling": "Each fold's residual * sdR_fold / (betaR_fold * sdR_full), then average folds and seeds."}


def train_penalty(args, history, candidates, penalty, full_scale, deadline, torch, TabM, device, rows):
    oof_p = np.full((len(args.seeds), len(history)), np.nan)
    oof_base = np.full_like(oof_p, np.nan)
    residuals, gradients = [], []
    for seed_index, seed in enumerate(args.seeds):
        splitter = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for fold, (train_ids, test_ids) in enumerate(splitter.split(history, strata(history))):
            row, predictions = train_member(args, history.iloc[train_ids], history.iloc[test_ids], candidates,
                                            seed * 100 + fold, fold, penalty, full_scale, deadline, torch, TabM, device)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out_dir / "training_metrics.csv", index=False)
            oof_p[seed_index, test_ids] = predictions["heldout"]["p"]
            oof_base[seed_index, test_ids] = predictions["heldout"]["baseline_p"]
            residuals.append(predictions["candidates"]["residual_rsd"])
            gradients.append(predictions["gradient_grid"])
            print(json.dumps({"completed_model": row}), flush=True)
    if not np.isfinite(oof_p).all() or not np.isfinite(oof_base).all():
        raise ValueError("Incomplete OOF predictions; export blocked")
    bundle = penalty_bundle(history, oof_p.mean(0), oof_base.mean(0), residuals, gradients)
    pd.DataFrame({"id_candidat": history.id_candidat, "committee_label": history.decision_octroi,
                  "oof_historical_probability": bundle["p"],
                  "oof_frozen_baseline_probability": bundle["baseline_p"]}).to_csv(
                      args.out_dir / f"oof_penalty_{penalty:g}.csv", index=False)
    return bundle


def main():
    args = parse_args()
    try:
        import torch
        from tabm import TabM
    except ImportError as error:
        raise RuntimeError("Install kaggle/requirements.txt in the notebook environment; no model fallback is used") from error
    torch.set_num_threads(2)
    if not torch.cuda.is_available() and not args.allow_cpu_smoke:
        raise RuntimeError("CUDA GPU required. Enable a free Kaggle GPU accelerator")
    if args.allow_cpu_smoke:
        args.folds, args.seeds, args.penalties = 2, args.seeds[:1], args.penalties[:1]
        args.epochs, args.patience, args.members, args.width = min(args.epochs, 2), 2, 2, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = find_data(args.train, "donnees_demandes.csv")
    candidate_path = find_data(args.candidates, "candidats_evaluation.csv")
    history, candidates = pd.read_csv(train_path), pd.read_csv(candidate_path)
    validate_data(history, candidates, args.allow_cpu_smoke)
    full_norm = normalization(history)
    manifest = base_manifest(args, device, torch, train_path, candidate_path)
    atomic_json(args.out_dir / "manifest.json", manifest)
    rows = []
    try:
        bundles = {penalty: train_penalty(args, history, candidates, penalty, full_norm[1], deadline, torch, TabM,
                                          device, rows)
                   for penalty in args.penalties}
        chosen = min(bundles, key=lambda penalty: bundles[penalty]["score"]["historical_logloss"])
        bundle = bundles[chosen]
        selection = pd.DataFrame({"penalty": penalty, **value["score"]} for penalty, value in bundles.items())
        selection.to_csv(args.out_dir / "penalty_selection.csv", index=False)
        residuals_path = args.out_dir / "residuals.csv"
        pd.DataFrame({"id_candidat": candidates.id_candidat, "residual_rsd": bundle["candidate_residual"]}).to_csv(
            residuals_path, index=False)
        baseline_score = metrics(history.decision_octroi, bundle["baseline_p"])
        manifest.update(status="complete", residuals_sha256=sha256_file(residuals_path), selected_penalty=chosen,
                        penalty_selection=selection.to_dict("records"),
                        selected_oof_metrics=bundle["score"], frozen_baseline_oof_metrics=baseline_score,
                        oof_logloss_delta=bundle["score"]["historical_logloss"] - baseline_score["historical_logloss"],
                        monotonicity=monotonicity_report(bundle, committee_hours_ratio(history, full_norm),
                                                         args.out_dir),
                        elapsed_minutes=(time.monotonic() - started) / 60)
        atomic_json(args.out_dir / "manifest.json", manifest)
        print(json.dumps(manifest), flush=True)
    except Exception as error:
        manifest.update(status="timeout" if isinstance(error, TimeoutError) else "failed", error=str(error),
                        completed_models=len(rows), elapsed_minutes=(time.monotonic() - started) / 60)
        atomic_json(args.out_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
