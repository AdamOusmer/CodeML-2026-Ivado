import os

os.environ["OMP_NUM_THREADS"] = "2"

import ast
import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HISTORY_PATH = ROOT / "data/donnees_demandes.csv"
BATCH_PATH = ROOT / "data/candidats_evaluation.csv"
OUTPUT_NAMES = ("predictions.csv", "explanations.csv", "decision_record.json")
SENTINEL = b"sentinel-do-not-touch\n"
CLI = ["uv", "run", "python", "-m", "src.main"]
CLI_FLAGS = ["--plain", "--no-log-file"]

IO_CALLS = {
    "read_csv", "to_csv", "savefig", "write_text", "write_bytes", "read_text", "read_bytes", "open",
    "dump", "to_json", "to_parquet", "to_pickle", "save", "savetxt", "mkdir", "unlink", "rmtree",
}
DYNAMIC_IMPORTS = {"import_module", "__import__"}
DYNAMIC_IMPORT_ALLOWED = ("src/policy/__init__.py",)
FLAGGED_REGIONS = ("Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine")
NO_OFFSET_REASON = "no offset in the allowed grid lowers the gap"
IO_ALLOWED = ("src/adapters/", "src/main.py", "src/common/logging/", "src/preprocessing/validation.py")
THIRD_PARTY_OK = {
    "policy": {"numpy", "pandas", "scipy", "sklearn"},
    "common": {"rich"},
}
MAY_IMPORT = {
    "main": {"adapters", "harness", "evaluation", "monitoring", "preprocessing", "pipelines", "policy", "common"},
    "model_corrige": {"adapters", "harness", "evaluation", "monitoring", "preprocessing", "pipelines", "policy", "common"},
    "adapters": {"harness", "common"},
    "harness": {"monitoring", "explain", "policy", "common"},
    "monitoring": {"policy", "common"},
    "pipelines": {"adapters", "harness", "evaluation", "preprocessing", "policy", "common"},
    "explain": {"policy"},
    "evaluation": {"policy"},
    "preprocessing": {"policy", "common"},
    "policy": set(),
    "common": set(),
}

RESULTS: list[tuple[str, str, str]] = []
SHARED: dict = {}


def report(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((status, name, detail))
    print(f"{status} {name} {detail}", flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_hashes() -> dict[str, str]:
    return {path.name: sha256_bytes(path.read_bytes()) for path in (HISTORY_PATH, BATCH_PATH)}


def run_cli(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    return subprocess.run([*CLI, *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def run_decide_cli(history: Path, batch: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return run_cli("decide", *CLI_FLAGS, *PANEL_CLI, "--history", str(history), "--batch", str(batch),
                   "--out-dir", str(out_dir))


def reject_constant(name: str):
    raise ValueError(f"non-strict JSON constant {name}")


def load_strict(path: Path) -> dict:
    return json.loads(path.read_text(), parse_constant=reject_constant)


def action_kinds(record) -> list[str]:
    return [getattr(action.kind, "value", str(action.kind)) for action in record.actions]


def output_hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_bytes((directory / name).read_bytes()) for name in OUTPUT_NAMES}


def make_sentinel_dir(base: Path, name: str) -> Path:
    directory = base / name
    directory.mkdir()
    (directory / "predictions.csv").write_bytes(SENTINEL)
    return directory


FORCED_ALERT = -0.1
PANEL_CLI = ["--config", "consensus panel"]


def validator_config():
    from src.policy import VALIDATOR_JURY_CONFIG

    return VALIDATOR_JURY_CONFIG


def forced_config():
    from src.policy import INCOME_BLIND_NO_JURY_CONFIG

    return INCOME_BLIND_NO_JURY_CONFIG


def panel_config():
    from src.policy import CONSENSUS_PANEL_CONFIG

    return CONSENSUS_PANEL_CONFIG


def decide_and_write(history, batch, directory: Path, **kwargs):
    from src.adapters import write_decision
    from src.harness import decide

    kwargs.setdefault("config", panel_config())
    record = decide(history, batch, input_hashes(), **kwargs)
    write_decision(record, directory)
    return record


def check_baseline(ctx) -> str:
    from src.adapters import write_decision
    from src.harness import decide

    history, batch = ctx["history"], ctx["batch"]
    share = float(history["decision_octroi"].mean())
    assert 0.36 <= share <= 0.44, f"share {share:.4f} outside 36-44%"
    assert abs(share - 0.3994) < 0.001, f"share {share:.4f} != ~0.3994"
    record = decide(history, batch, input_hashes(), config=panel_config())
    ctx["baseline"] = record
    grants = int(np.sum(record.decisions))
    assert record.status == "published", f"status {record.status}"
    assert len(record.decisions) == 4000, f"rows {len(record.decisions)}"
    assert grants == round(share * 4000), f"grants {grants} != round(share*n)"
    assert grants == 1598, f"grants {grants} != 1598"
    assert "ADJUST_OFFSET" not in action_kinds(record), f"actions {action_kinds(record)}"
    out = ctx["tmp"] / "baseline"
    write_decision(record, out)
    load_strict(out / "decision_record.json")
    predictions = pd.read_csv(out / "predictions.csv")
    assert list(predictions["id_candidat"]) == list(batch["id_candidat"]), "id order differs from batch"
    assert len(predictions) == 4000 and int(predictions["decision_octroi"].sum()) == grants, "predictions.csv mismatch"
    return f"rows=4000 grants={grants} share={share:.4f} actions={action_kinds(record)}"


def check_budget_guard(ctx) -> str:
    history = ctx["history"].copy()
    history["decision_octroi"] = np.arange(len(history)) % 2
    rate = history["decision_octroi"].mean()
    history_path = ctx["tmp"] / "history_rate50.csv"
    history.to_csv(history_path, index=False)
    out = ctx["tmp"] / "budget_guard"
    result = run_decide_cli(history_path, BATCH_PATH, out)
    lines = [line for line in result.stderr.splitlines() if line.strip().startswith("ERROR")]
    assert result.returncode == 1, f"exit {result.returncode}, stderr tail {result.stderr.splitlines()[-1:]}"
    assert "Traceback" not in result.stderr, "traceback in stderr"
    assert not out.exists() or not any(out.iterdir()), f"files written: {sorted(p.name for p in out.iterdir())}"
    assert len(lines) == 1, f"{len(lines)} ERROR lines: {lines[:3]}"
    return f"rate={rate:.2f} exit=1 {lines[0][:100]!r}"


def check_duplicate_id(ctx) -> str:
    batch = ctx["batch"].copy()
    batch.loc[1, "id_candidat"] = batch.loc[0, "id_candidat"]
    batch_path = ctx["tmp"] / "batch_duplicate.csv"
    batch.to_csv(batch_path, index=False)
    out = ctx["tmp"] / "duplicate"
    result = run_decide_cli(HISTORY_PATH, batch_path, out)
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert result.returncode == 1, f"exit {result.returncode}, stderr tail {lines[-1:]}"
    assert "Traceback" not in result.stderr, "traceback in stderr"
    assert re.search(r"duplicated? id_candidat", result.stderr), f"stderr tail {lines[-1:]}"
    assert not out.exists() or not any(out.iterdir()), f"files written: {sorted(p.name for p in out.iterdir())}"
    return f"exit=1 stderr={lines[-1][:90]!r}" if lines else "exit=1"


def check_forced_correctable_alert(ctx) -> str:
    from src.policy import is_remote

    record = ctx["baseline_decide"](config=forced_config(), eo_gap_alert=FORCED_ALERT)
    kinds = action_kinds(record)
    n = len(record.decisions)
    grants = int(np.sum(record.decisions))
    offset = float(record.offset)
    assert "ADJUST_OFFSET" in kinds, f"no ADJUST_OFFSET: {kinds}"
    assert offset != 0 and abs(offset) <= 0.10 + 1e-12, f"offset {offset}"
    assert len(record.verdicts) == 2, f"{len(record.verdicts)} verdicts"
    assert grants == round(record.share * n) == 1598, f"grants {grants}"
    assert record.offset_moved_ids, "offset_moved_ids empty"
    pipeline, batch = fitted_pipeline(ctx, forced_config()), ctx["batch"]
    ids = batch["id_candidat"].to_numpy()
    remote = is_remote(batch).astype(bool)
    flat = pipeline.decide(batch).proposed
    shifted = pipeline.decide(batch, offset)
    assert np.array_equal(shifted.decisions, record.decisions), "decisions differ from independent decide(offset)"
    entered = (flat == 0) & (shifted.proposed == 1)
    left = (flat == 1) & (shifted.proposed == 0)
    assert entered.sum() == left.sum() > 0, f"proposal entered {entered.sum()} != left {left.sum()}"
    sign_group = remote if offset > 0 else ~remote
    assert sign_group[entered].all(), f"offset {offset:+.3f} admitted entrants from the wrong group"
    assert set(record.offset_moved_ids) == set(ids[entered | left].tolist()), "offset_moved_ids != flipped proposals"
    out, into = record.jury["overturned_out"], record.jury["overturned_in"]
    assert out == into, f"jury swaps out {out} != in {into}"
    assert len(record.jury_moved_ids) == out + into, "jury_moved_ids count != swaps"
    assert set(record.jury_moved_ids) == set(ids[shifted.decisions != shifted.proposed].tolist()), "jury_moved_ids != independent jury"
    return (f"ADJUST_OFFSET offset={offset:+.3f} offset_moved={len(record.offset_moved_ids)} "
            f"jury_moved={len(record.jury_moved_ids)} status={record.status} grants={grants}")


def block_action(summary: dict) -> dict:
    blocks = [action for action in summary["actions"] if action["kind"] == "BLOCK"]
    assert len(blocks) == 1, f"{len(blocks)} BLOCK actions"
    return blocks[0]


def alert_checks(summary: dict) -> list[dict]:
    return [check for verdict in summary["verdicts"] for check in verdict["checks"] if check["status"] == "ALERT"]


def check_drift_block(ctx) -> str:
    batch = ctx["batch"].copy()
    batch["cote_r_equivalent"] = (batch["cote_r_equivalent"] + 3).clip(upper=40)
    batch_path = ctx["tmp"] / "batch_drift.csv"
    batch.to_csv(batch_path, index=False)
    out = make_sentinel_dir(ctx["tmp"], "drift")
    result = run_decide_cli(HISTORY_PATH, batch_path, out)
    assert result.returncode == 3, f"exit {result.returncode}, stderr tail {result.stderr.splitlines()[-1:]}"
    assert (out / "predictions.csv").read_bytes() == SENTINEL, "sentinel predictions.csv changed (I5)"
    summary = load_strict(out / "decision_record.json")
    kinds = [action["kind"] for action in summary["actions"]]
    assert summary["status"] == "blocked", f"status {summary['status']}"
    assert "ADJUST_OFFSET" not in kinds, f"actions {kinds}"
    checks = block_action(summary)["params"]["checks"]
    assert "feature drift, max PSI" in checks, f"BLOCK checks {checks}"
    return f"exit=3 status=blocked actions={kinds} sentinel intact"


def check_categorical_drift(ctx) -> str:
    batch = ctx["batch"].copy()
    batch["premiere_generation_universitaire"] = 0
    batch_path = ctx["tmp"] / "batch_categorical_drift.csv"
    batch.to_csv(batch_path, index=False)
    out = ctx["tmp"] / "categorical_drift"
    result = run_decide_cli(HISTORY_PATH, batch_path, out)
    assert result.returncode == 3, f"exit {result.returncode}, stderr tail {result.stderr.splitlines()[-1:]}"
    summary = load_strict(out / "decision_record.json")
    kinds = [action["kind"] for action in summary["actions"]]
    assert "ADJUST_OFFSET" not in kinds, f"actions {kinds}"
    drift = [check for check in alert_checks(summary) if check["name"] == "categorical drift, max PSI"]
    assert drift and not any(check["correctable"] for check in drift), f"drift alerts {drift}"
    assert "categorical drift, max PSI" in block_action(summary)["params"]["checks"], "BLOCK lacks drift check"
    return f"exit=3 PSI={drift[0]['value']:.2f} actions={kinds}"


def check_uncomputable_metric_block(ctx) -> str:
    batch = ctx["batch"]
    remote = batch[batch["region_administrative"].isin(FLAGGED_REGIONS)]
    assert 0 < len(remote) < len(batch), f"remote rows {len(remote)}"
    batch_path = ctx["tmp"] / "batch_remote_only.csv"
    remote.to_csv(batch_path, index=False)
    out = ctx["tmp"] / "uncomputable"
    result = run_decide_cli(HISTORY_PATH, batch_path, out)
    assert result.returncode == 3, f"exit {result.returncode}, stderr tail {result.stderr.splitlines()[-1:]}"
    summary = load_strict(out / "decision_record.json")
    kinds = [action["kind"] for action in summary["actions"]]
    assert "ADJUST_OFFSET" not in kinds, f"actions {kinds}"
    stuck = [check for check in alert_checks(summary) if not check["correctable"] and check["value"] is None]
    assert stuck, "no non-correctable ALERT check with null value"
    return f"exit=3 rows={len(remote)} actions={kinds} uncomputable={[check['name'] for check in stuck][:2]}"


def check_alert_after_correction(ctx) -> str:
    batch = ctx["batch"]
    out = make_sentinel_dir(ctx["tmp"], "alert_after_correction")
    record = decide_and_write(ctx["history"], batch, out, config=forced_config(), eo_gap_alert=-1.0)
    kinds = action_kinds(record)
    assert kinds[-1] == "BLOCK" and "ADJUST_OFFSET" in kinds, f"actions {kinds}"
    assert kinds.index("ADJUST_OFFSET") < kinds.index("BLOCK"), f"BLOCK before ADJUST_OFFSET: {kinds}"
    assert record.offset != 0 and len(record.verdicts) == 2, f"offset {record.offset} verdicts {len(record.verdicts)}"
    block = record.actions[-1]
    assert block.reason == "alert remains after allowed corrections", f"reason {block.reason!r}"
    assert record.status == "blocked", f"status {record.status}"
    assert (out / "predictions.csv").read_bytes() == SENTINEL, "sentinel predictions.csv changed (I5)"
    return f"actions={kinds} offset={record.offset:+.3f} reason={block.reason!r} sentinel intact"


def check_zero_offset_block(ctx) -> str:
    import src.harness.postprocessing as postprocessing

    original = postprocessing.fit_offset
    postprocessing.fit_offset = lambda *args, **kwargs: 0.0
    try:
        record = ctx["baseline_decide"](eo_gap_alert=FORCED_ALERT)
    finally:
        postprocessing.fit_offset = original
    kinds = action_kinds(record)
    assert kinds == ["SELECT_CONFIG", "BLOCK"], f"actions {kinds}"
    assert record.actions[-1].reason == NO_OFFSET_REASON, f"reason {record.actions[-1].reason!r}"
    assert len(record.verdicts) == 1, f"{len(record.verdicts)} verdicts"
    assert record.status == "blocked", f"status {record.status}"
    return f"actions={kinds} reason={NO_OFFSET_REASON!r}"


def check_replay(ctx) -> str:
    hashes = []
    for name in ("replay_a", "replay_b"):
        directory = ctx["tmp"] / name
        decide_and_write(ctx["history"], ctx["batch"], directory)
        hashes.append(output_hashes(directory))
    assert hashes[0] == hashes[1], f"differs: {[k for k in hashes[0] if hashes[0][k] != hashes[1][k]]}"
    ctx["replay_hashes"] = hashes[0]
    return "sha256 equal " + " ".join(f"{k}={v[:8]}" for k, v in hashes[0].items())


def check_region_blindness(ctx) -> str:
    batch = ctx["batch"]
    pipeline = declared_pipeline(ctx)
    rng = np.random.default_rng(0)
    shuffled = batch.copy()
    for column in ("region_administrative", "code_postal_3", "distance_domicile_campus_km"):
        shuffled[column] = rng.permutation(batch[column].to_numpy())
    changed = sum(not shuffled[c].equals(batch[c]) for c in ("region_administrative", "code_postal_3", "distance_domicile_campus_km"))
    assert changed == 3, "permutation left a column unchanged"
    base, other = np.asarray(pipeline.score(batch, 0.0)), np.asarray(pipeline.score(shuffled, 0.0))
    assert np.array_equal(base, other), f"max abs diff {np.max(np.abs(base - other)):.3g}"
    return "scores identical after permuting region, postal, distance"


def check_tuner_not_live(ctx) -> str:
    out = ctx["tmp"] / "tuner"
    out.mkdir()
    result = run_cli("tune", *CLI_FLAGS, "--splits", "1", "--workers", "2", "--out-dir", str(out))
    assert result.returncode == 0, f"tune exit {result.returncode}: {result.stderr.splitlines()[-1:]}"
    produced = {path.name for path in out.iterdir()}
    assert produced == {"resultats_tuner.csv"}, f"tune wrote {sorted(produced)}"
    directory = ctx["tmp"] / "after_tune"
    decide_and_write(ctx["history"], ctx["batch"], directory)
    reference = ctx.get("replay_hashes")
    if reference is None:
        decide_and_write(ctx["history"], ctx["batch"], ctx["tmp"] / "after_tune_ref")
        reference = output_hashes(ctx["tmp"] / "after_tune_ref")
    assert output_hashes(directory) == reference, "decide output changed after tune"
    return "tune exit 0; decide hashes equal to replay"


def package_of(parts: list[str]) -> str | None:
    if parts[0] == "model_corrige":
        return "model_corrige"
    if parts[0] != "src" or len(parts) < 2:
        return None
    return parts[1]


def module_parts(root: Path, path: Path) -> tuple[list[str], list[str]]:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    package = parts[:-1]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return parts, package


def is_submodule(root: Path, package: str, name: str) -> bool:
    base = root / "src" / package
    if not base.is_dir():
        return False
    return name in {path.name.removesuffix(".py") for path in base.iterdir() if path.suffix == ".py" or path.is_dir()}


def resolve_imports(root: Path, node, package_parts: list[str]) -> list[list[str]]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".") for alias in node.names]
    if node.level:
        keep = len(package_parts) - (node.level - 1)
        base = package_parts[:max(keep, 0)]
    else:
        base = []
    target = base + (node.module.split(".") if node.module else [])
    if target == ["src"]:
        return [target + [alias.name] for alias in node.names]
    if len(target) == 2 and target[0] == "src":
        return [target + [alias.name] if is_submodule(root, target[1], alias.name) else target for alias in node.names]
    return [target]


def attribute_chain(node) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return []


def is_stdlib(name: str) -> bool:
    return name in sys.stdlib_module_names or name == "__future__"


def scan_file(root: Path, path: Path) -> list[tuple[str, str, int, str]]:
    relative = path.relative_to(root).as_posix()
    module, package_parts = module_parts(root, path)
    own = package_of(module)
    if own is None:
        return []
    violations = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for target in resolve_imports(root, node, package_parts):
                check_import(relative, node.lineno, own, target, violations)
        if isinstance(node, ast.Attribute):
            chain = attribute_chain(node)
            if len(chain) >= 3 and chain[0] == "src" and is_submodule(root, chain[1], chain[2]):
                check_import(relative, node.lineno, own, chain[:3], violations)
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in IO_CALLS and not relative.startswith(IO_ALLOWED):
                violations.append(("io", relative, node.lineno, f"{name}() outside allowed files"))
            if name in DYNAMIC_IMPORTS and relative not in DYNAMIC_IMPORT_ALLOWED:
                violations.append(("deep", relative, node.lineno, f"{name}() dynamic import"))
    return violations


def check_import(relative: str, line: int, own: str, target: list[str], violations: list) -> None:
    if target[0] != "src":
        top = target[0]
        if own in THIRD_PARTY_OK and not is_stdlib(top) and top not in THIRD_PARTY_OK[own]:
            violations.append(("external", relative, line, f"{own} imports {top}"))
        return
    other = package_of(target)
    if other is None or other == own:
        return
    limit = 3 if other == "common" else 2
    if len(target) > limit:
        violations.append(("deep", relative, line, ".".join(target)))
    if other not in MAY_IMPORT.get(own, set()):
        violations.append(("direction", relative, line, f"{own} -> {other}"))


def scan_boundaries(root: Path) -> list[tuple[str, str, int, str]]:
    files = sorted((root / "src").rglob("*.py"))
    if (root / "model_corrige.py").exists():
        files.append(root / "model_corrige.py")
    violations = []
    for path in files:
        violations.extend(scan_file(root, path))
    return violations


def summarize_violations(violations: list, rule: str) -> str:
    selected = [f"{file}:{line} {message}" for kind, file, line, message in violations if kind == rule]
    return f"{len(selected)} violation(s)" + (": " + "; ".join(selected[:4]) if selected else "")


def check_policy_lazy() -> str:
    code = "import sys; from src.policy import REGIONS; bad=[m for m in sys.modules if m=='sklearn' or m.startswith('sklearn.')]; sys.exit(1 if bad else 0)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"
    return "REGIONS import loads no sklearn"


def copy_tree(destination: Path) -> Path:
    destination.mkdir()
    shutil.copytree(ROOT / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__"))
    if (ROOT / "model_corrige.py").exists():
        shutil.copy2(ROOT / "model_corrige.py", destination / "model_corrige.py")
    return destination


def first_module(root: Path, package: str) -> Path:
    candidates = sorted(path for path in (root / "src" / package).glob("*.py") if path.name != "__init__.py")
    assert candidates, f"no module in src/{package}"
    return candidates[0]


def injected_violations(tmp: Path, name: str, package: str, line: str) -> list:
    root = copy_tree(tmp / name)
    target = first_module(root, package)
    target.write_text(target.read_text() + "\n" + line + "\n")
    return scan_boundaries(root)


def new_violations(after: list, before: list) -> list:
    known = {(kind, file, message) for kind, file, _, message in before}
    return [item for item in after if (item[0], item[1], item[3]) not in known]


def check_boundary_proof(ctx) -> str:
    tmp = ctx["tmp"]
    baseline = scan_boundaries(copy_tree(tmp / "proof_base"))
    cases = [
        ("proof_deep", "monitoring", "from src.policy.core import eo_gap", {"deep"}),
        ("proof_direction", "monitoring", "from src.evaluation import tune", {"direction"}),
        ("proof_io", "harness", "pd_frame = None\npd_frame.to_csv('x.csv')", {"io"}),
        ("proof_io_dump", "harness", "import json\njson.dump({}, None)", {"io"}),
        ("proof_io_mkdir", "harness", "from pathlib import Path\nPath('x').mkdir()", {"io"}),
        ("proof_from_submodule", "monitoring", "from src.policy import core", {"deep"}),
        ("proof_import_module", "monitoring",
         "import importlib\nimportlib.import_module('src.evaluation.tuner')", {"deep"}),
        ("proof_dunder_import", "monitoring", "__import__('src.policy.core')", {"deep"}),
        ("proof_attribute_chain", "monitoring", "import src.policy\nsrc.policy.core.eo_gap", {"deep"}),
    ]
    notes = []
    for name, package, line, expected in cases:
        found = {item[0] for item in new_violations(injected_violations(tmp, name, package, line), baseline)}
        assert expected <= found, f"{name}: scanner missed {expected - found}"
        notes.append(f"{name}=caught")
    allowed = new_violations(injected_violations(tmp, "proof_allowed", "monitoring", "from src.policy import eo_gap"), baseline)
    assert not allowed, f"allowed edge flagged: {allowed}"
    notes.append("proof_allowed=clean")
    return " ".join(notes)


def check_explain_identity(ctx) -> str:
    from src.explain import explain
    from src.policy import VALIDATOR_JURY_CONFIG, FairPipeline, budget_share, scoring_features

    history, batch = ctx["history"], ctx["batch"]
    pipeline = FairPipeline(VALIDATOR_JURY_CONFIG, budget_share(history)).fit(history)
    contributions = pipeline.contributions(batch)
    model = pipeline.main_model_
    intercept = float(model[-1].intercept_[0])
    reconstructed = contributions.to_numpy().sum(axis=1) + intercept
    expected = model.decision_function(scoring_features(batch))
    error = float(np.max(np.abs(reconstructed - expected)))
    assert error < 1e-9, f"max abs err {error:.3g}"
    outcome = pipeline.decide(batch)
    table = explain(pipeline, batch, outcome, pipeline.score(batch, 0.0), 0.0)
    needed = {"id_candidat", "decision", "score", "merit_vote", "model_vote", "offset", "factor_1", "factor_2", "factor_3",
              "validated", "trigger_reasons", "juror_votes", "jury_outcome"}
    assert needed <= set(table.columns), f"missing columns {needed - set(table.columns)}"
    return f"max abs err {error:.2e}"


def fitted_pipeline(ctx, config=None):
    from src.policy import VALIDATOR_JURY_CONFIG, FairPipeline, budget_share

    config = config or VALIDATOR_JURY_CONFIG
    key = f"pipeline:{config.name}"
    if key not in ctx:
        ctx[key] = FairPipeline(config, budget_share(ctx["history"])).fit(ctx["history"])
    return ctx[key]


def jury_variants() -> dict:
    from dataclasses import replace

    from src.policy import JurySettings

    default = JurySettings()
    return {
        "default": default,
        "band0": replace(default, band=0.0),
        "zero_triggers": replace(default, band=0.0, conf=0.0, disagree=1.0),
        "quorum_half": replace(default, quorum=0.5),
        "merit_only": replace(default, jurors=("merit",)),
    }


def jury_outcomes(ctx) -> tuple[dict, int]:
    from src.policy import validate

    pipeline, batch = fitted_pipeline(ctx), ctx["batch"]
    probability = pipeline.score(batch, 0.0)
    jurors = pipeline.juror_scores(batch)
    k = int(round(pipeline.share * len(batch)))
    return {name: validate(probability, jurors, k, settings) for name, settings in jury_variants().items()}, k


def check_jury_grant_count(ctx) -> str:
    outcomes, k = jury_outcomes(ctx)
    grants = {name: int(outcome.decisions.sum()) for name, outcome in outcomes.items()}
    wrong = {name: count for name, count in grants.items() if count != k}
    assert not wrong, f"grants != {k}: {wrong}"
    zero = outcomes["zero_triggers"]
    assert not zero.triggered.any(), f"{int(zero.triggered.sum())} triggered with triggers off"
    assert np.array_equal(zero.decisions, zero.proposed), "decisions != proposed with zero triggers"
    return f"k={k} grants " + " ".join(f"{name}={count}" for name, count in grants.items())


def check_jury_swap_symmetry(ctx) -> str:
    outcomes, _ = jury_outcomes(ctx)
    for name, outcome in outcomes.items():
        out, into = outcome.overturned_out, outcome.overturned_in
        assert len(out) == len(into), f"{name}: out {len(out)} != in {len(into)}"
        assert np.all(outcome.proposed[out] == 1), f"{name}: overturned_out not proposed grants"
        assert np.all(outcome.proposed[into] == 0), f"{name}: overturned_in not proposed refusals"
        assert np.all(outcome.triggered[out]) and np.all(outcome.triggered[into]), f"{name}: untriggered overturn"
    return "swaps " + " ".join(f"{name}={len(outcome.overturned_out)}" for name, outcome in outcomes.items())


def check_jury_region_invariance(ctx) -> str:
    pipeline, batch = fitted_pipeline(ctx), ctx["batch"]
    columns = ["region_administrative", "code_postal_3", "distance_domicile_campus_km"]
    order = np.random.default_rng(11).permutation(len(batch))
    shuffled = batch.copy()
    shuffled[columns] = batch[columns].to_numpy()[order]
    assert not shuffled[columns].equals(batch[columns]), "permutation left columns unchanged"
    assert np.array_equal(pipeline.decide(batch).decisions, pipeline.decide(shuffled).decisions), "decisions differ (I2)"
    return "decisions identical after joint shuffle of region, postal, distance"


def check_jury_determinism(ctx) -> str:
    pipeline, batch = fitted_pipeline(ctx), ctx["batch"]
    first, second = pipeline.decide(batch), pipeline.decide(batch)
    for field in ("decisions", "proposed", "triggered", "overturned_out", "overturned_in"):
        assert np.array_equal(getattr(first, field), getattr(second, field)), f"{field} differs"
    assert first.reasons == second.reasons, "reasons differ"
    assert first.votes.keys() == second.votes.keys(), "juror names differ"
    assert all(np.array_equal(first.votes[name], second.votes[name]) for name in first.votes), "votes differ"
    return f"two decide calls identical over {len(batch)} rows"


def check_jury_record(ctx) -> str:
    summary = json.loads((ctx["tmp"] / "baseline" / "decision_record.json").read_text())
    jury = summary["jury"]
    assert 150 <= jury["triggered"] <= 250, f"triggered {jury['triggered']} outside 150-250"
    assert jury["overturned_out"] == jury["overturned_in"], f"out {jury['overturned_out']} != in {jury['overturned_in']}"
    assert {"applied", "reasons"} <= set(jury) and jury["applied"], f"jury keys {sorted(jury)}"
    declared = fitted_pipeline(ctx).decide(ctx["batch"])
    assert 20 <= len(declared.overturned_out) <= 40, f"validator jury swaps {len(declared.overturned_out)} outside 20-40"
    return (f"panel record triggered={jury['triggered']} swaps={jury['overturned_out']}; "
            f"validator jury (pipeline) swaps={len(declared.overturned_out)}")


def check_jury_offset_monotone(ctx) -> str:
    from src.harness import OFFSET_GRID
    from src.policy import is_remote

    pipeline, batch = fitted_pipeline(ctx), ctx["batch"]
    remote = is_remote(batch).astype(bool)
    k = int(round(pipeline.share * len(batch)))
    grants, remote_grants = {}, {}
    for offset in OFFSET_GRID:
        decisions = pipeline.predict(batch, float(offset))
        grants[round(float(offset), 3)] = int(decisions.sum())
        remote_grants[round(float(offset), 3)] = int(decisions[remote].sum())
    wrong = {offset: count for offset, count in grants.items() if count != k}
    assert not wrong, f"grants != {k}: {wrong}"
    series = list(remote_grants.values())
    drops = [(a, b) for a, b in zip(series, series[1:]) if b < a - 2]
    assert not drops, f"remote grants fall by more than 2: {drops}"
    assert series[-1] >= series[0], f"remote grants {series[0]} -> {series[-1]} decrease overall"
    reach = remote_grants[0.1] - remote_grants[-0.1]
    assert reach >= 10, f"offset reach {reach} < 10: remote grants {remote_grants[-0.1]} at -0.10, {remote_grants[0.1]} at +0.10"
    return f"remote grants {remote_grants[-0.1]}/{remote_grants[0.0]}/{remote_grants[0.1]} at -0.10/0/+0.10, grants={k} at {len(grants)} offsets"


def check_frame_rejects(ctx) -> str:
    from src.preprocessing import DataValidationError, validate_frames

    history, batch = ctx["history"], ctx["batch"]

    def zero_income(frame):
        frame.loc[frame.index[0], "revenu_familial_estime"] = 0

    def unknown_programme(frame):
        frame.loc[frame.index[0], "programme_etudes"] = "Droit"

    def unknown_region(frame):
        frame.loc[frame.index[0], "region_administrative"] = "Laval"

    def nan_cote(frame):
        frame.loc[frame.index[0], "cote_r_equivalent"] = np.nan

    def inf_hours(frame):
        frame["heures_travail_semaine"] = frame["heures_travail_semaine"].astype(float)
        frame.loc[frame.index[0], "heures_travail_semaine"] = np.inf

    def duplicate_id(frame):
        frame.loc[frame.index[1], "id_candidat"] = frame.loc[frame.index[0], "id_candidat"]

    def overlapping_id(frame):
        frame.loc[frame.index[0], "id_candidat"] = history["id_candidat"].iloc[0]

    def nan_id(frame):
        frame["id_candidat"] = frame["id_candidat"].astype(object)
        frame.loc[frame.index[0], "id_candidat"] = np.nan

    def nan_postal(frame):
        frame["code_postal_3"] = frame["code_postal_3"].astype(object)
        frame.loc[frame.index[0], "code_postal_3"] = np.nan

    def unicode_digit_id(frame):
        frame["id_candidat"] = frame["id_candidat"].astype(object)
        frame.loc[frame.index[0], "id_candidat"] = "C١٢٣٤٥٦"

    mutations = [
        zero_income,
        unknown_programme,
        unknown_region,
        nan_cote,
        inf_hours,
        duplicate_id,
        overlapping_id,
        nan_id,
        nan_postal,
        unicode_digit_id,
    ]
    for mutate in mutations:
        mutated = batch.copy()
        mutate(mutated)
        try:
            validate_frames(history, mutated)
        except DataValidationError:
            continue
        raise AssertionError(f"{mutate.__name__} not rejected")
    frame_report = validate_frames(history, batch)
    warnings = list(frame_report.warnings)
    assert len(warnings) >= 2, f"{len(warnings)} warnings: {warnings}"
    joined = " | ".join(str(warning) for warning in warnings)
    for column in ("cote_r_equivalent", "revenu_familial_estime"):
        assert column in joined, f"no warning mentions {column}: {joined[:200]}"
    return f"{len(mutations)} mutations rejected; real frames pass with {len(warnings)} warnings"


def check_csv_income_zero(ctx) -> str:
    from src.preprocessing.validation import DataValidationError, validate_csv

    batch = ctx["batch"].copy()
    batch.loc[batch.index[0], "revenu_familial_estime"] = 0
    path = ctx["tmp"] / "batch_income_zero.csv"
    batch.to_csv(path, index=False)
    try:
        validate_csv(path, False, None)
    except DataValidationError as error:
        assert "revenu_familial_estime" in str(error), f"message lacks column: {str(error)[:200]}"
        return "income 0 rejected naming revenu_familial_estime"
    raise AssertionError("income 0 accepted by validate_csv")


def check_region_invariance(ctx) -> str:
    batch = ctx["batch"]
    pipeline = declared_pipeline(ctx)
    columns = ["region_administrative", "code_postal_3", "distance_domicile_campus_km"]
    order = np.random.default_rng(7).permutation(len(batch))
    shuffled = batch.copy()
    shuffled[columns] = batch[columns].to_numpy()[order]
    assert not shuffled[columns].equals(batch[columns]), "permutation left columns unchanged"
    assert np.array_equal(pipeline.score(batch, offset=0), pipeline.score(shuffled, offset=0)), "scores differ (I2)"
    return "scores identical after joint shuffle of region, postal, distance"


def check_schemas(ctx) -> str:
    from src.policy import COMMITTEE_FEATURES, SCORING_FEATURES, committee_features, scoring_features

    history, batch = ctx["history"], ctx["batch"]
    assert list(scoring_features(batch).columns) == SCORING_FEATURES == ["cote_r", "log_revenu", "heures_travail"], "scoring order"
    for name, frame in (("history", history), ("batch", batch)):
        committee, scoring = committee_features(frame), scoring_features(frame)
        assert committee.shape[1] == 8 and list(committee.columns) == COMMITTEE_FEATURES, f"{name} committee columns"
        assert scoring.shape[1] == 3 and list(scoring.columns) == SCORING_FEATURES, f"{name} scoring columns"
    feature_history = [c for c in history.columns if c != "decision_octroi"]
    assert list(batch.columns) == feature_history, "history/batch columns differ"
    assert list(committee_features(history).columns) == list(committee_features(batch).columns), "committee misaligned"
    return f"committee={len(COMMITTEE_FEATURES)} scoring={SCORING_FEATURES}"


def check_categorical_drift_ok(ctx) -> str:
    from src.monitoring import run_checks

    record = ctx["baseline_decide"]()
    checks = run_checks(ctx["history"], ctx["batch"], record.decisions)
    found = [c for c in checks if c.name == "categorical drift, max PSI"]
    assert len(found) == 1, f"{len(found)} categorical drift checks"
    check = found[0]
    assert check.status == "OK", f"status {check.status}"
    assert abs(check.value - 0.0383) < 0.001, f"value {check.value:.4f}"
    assert "Gaspesie" in check.detail and "Genie +7.98" in check.detail, f"detail {check.detail!r}"
    return f"psi={check.value:.4f} status=OK detail={check.detail[:80]!r}"


def check_monitoring_concurrency(ctx) -> str:
    from src.monitoring import run_checks

    record = ctx["baseline_decide"]()
    serial = run_checks(ctx["history"], ctx["batch"], record.decisions, workers=1)
    parallel = run_checks(ctx["history"], ctx["batch"], record.decisions, workers=4)
    left = json.dumps([c.to_dict() for c in serial], sort_keys=True)
    right = json.dumps([c.to_dict() for c in parallel], sort_keys=True)
    assert left == right, "workers=1 and workers=4 results differ"
    return f"{len(serial)} checks identical for workers 1 and 4"


def check_graph_runtime(ctx) -> str:
    from src.common.graph import Graph, GraphError, Node

    def slow(value):
        time.sleep(0.3)
        return value + 1

    diamond = Graph([
        Node("top", lambda x: x * 2, ("x",)),
        Node("left", slow, ("top",)),
        Node("right", slow, ("top",)),
        Node("join", lambda a, b: a + b, ("left", "right")),
    ])
    serial = diamond.run({"x": 1}, workers=1)
    started = time.time()
    parallel = diamond.run({"x": 1}, workers=2)
    elapsed = time.time() - started
    assert elapsed < 0.5, f"workers=2 took {elapsed:.2f}s"
    assert serial == parallel, "workers=1 and workers=2 results differ"
    assert parallel["join"] == 6, f"join {parallel['join']}"
    try:
        Graph([Node("a", lambda b: b, ("b",)), Node("b", lambda a: a, ("a",))])
    except GraphError:
        pass
    else:
        raise AssertionError("cycle accepted")
    try:
        diamond.run({})
    except GraphError:
        pass
    else:
        raise AssertionError("missing seed accepted")
    assert diamond.order() == diamond.order() == list(diamond.order()), "order unstable"
    assert [n for n in diamond.order()] == [n for n in Graph(diamond.nodes).order()], "order differs across instances"

    class Boom(KeyError):
        pass

    failing = Graph([Node("bad", lambda x: (_ for _ in ()).throw(Boom("boom")), ("x",))])
    for workers in (1, 2):
        try:
            failing.run({"x": 1}, workers=workers)
        except Boom:
            pass
        except Exception as error:
            raise AssertionError(f"workers={workers} raised {type(error).__name__}")
        else:
            raise AssertionError("failing node did not raise")
    return f"diamond workers=2 {elapsed:.2f}s; cycle, missing seed, order, exception type ok"


def check_cli_invalid_batch(ctx) -> str:
    batch = ctx["batch"].copy()
    batch.loc[batch.index[0], "revenu_familial_estime"] = 0
    batch_path = ctx["tmp"] / "batch_income_zero_cli.csv"
    batch.to_csv(batch_path, index=False)
    out = make_sentinel_dir(ctx["tmp"], "invalid_batch")
    result = run_decide_cli(HISTORY_PATH, batch_path, out)
    lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert result.returncode == 1, f"exit {result.returncode}, stderr tail {lines[-1:]}"
    assert "Traceback" not in result.stderr, "traceback in stderr"
    assert (out / "predictions.csv").read_bytes() == SENTINEL, "sentinel predictions.csv changed"
    written = [name for name in OUTPUT_NAMES[1:] if (out / name).exists()]
    assert not written, f"outputs written: {written}"
    return f"exit=1 nothing written stderr={lines[-1][:80]!r}" if lines else "exit=1 nothing written"


def check_extra_fields(ctx) -> str:
    lines = ctx["batch"].to_csv(index=False).splitlines(keepends=True)
    lines[1] = "X," + lines[1]
    batch_path = ctx["tmp"] / "batch_extra_field.csv"
    batch_path.write_text("".join(lines))
    out = ctx["tmp"] / "extra_field"
    result = run_decide_cli(HISTORY_PATH, batch_path, out)
    stderr_lines = [line for line in result.stderr.splitlines() if line.strip()]
    assert result.returncode == 1, f"exit {result.returncode}, stderr tail {stderr_lines[-1:]}"
    assert "Traceback" not in result.stderr, "traceback in stderr"
    written = sorted(path.name for path in out.iterdir()) if out.exists() else []
    assert not written, f"outputs written: {written}"
    return f"exit=1 nothing written stderr={stderr_lines[-1][:80]!r}" if stderr_lines else "exit=1 nothing written"


def check_label_correction(ctx) -> str:
    from src.policy import CommitteeModel, allocate, budget_share, correct_labels, is_remote

    history = ctx["history"]
    share = budget_share(history)
    result = correct_labels(history, share)
    ctx["correction"] = result
    expected = allocate(CommitteeModel().fit(history, history["decision_octroi"]).corrected_logit(history, 1.0), share)
    assert np.array_equal(result.labels, expected), "corrected labels differ from allocate(corrected_logit)"
    assert result.k == 3994 == int(result.labels.sum()), f"k {result.k} labels {int(result.labels.sum())}"
    assert len(result.flipped_in) == len(result.flipped_out) == 651, f"flips {len(result.flipped_in)}/{len(result.flipped_out)}"
    remote = is_remote(history)
    flipped_remote = int(remote[result.flipped_in].sum())
    flipped_centre = int((remote[result.flipped_out] == 0).sum())
    assert flipped_remote == 432, f"remote flipped in {flipped_remote}"
    assert flipped_centre == 577, f"centre flipped out {flipped_centre}"
    centre_rate = float(result.labels[remote == 0].mean())
    remote_rate = float(result.labels[remote == 1].mean())
    assert abs(centre_rate - 0.424) < 0.002, f"centre rate {centre_rate:.4f}"
    assert abs(remote_rate - 0.362) < 0.002, f"remote rate {remote_rate:.4f}"
    return f"k=3994 flips=651/651 remote_in=432 centre_out=577 rates centre={centre_rate:.3f} remote={remote_rate:.3f}"


def check_correction_report(ctx) -> str:
    from src.policy import budget_share, correction_report, report_warnings

    share = budget_share(ctx["history"])
    first = correction_report(ctx["history"], share)
    low, high = first.penalty_ci
    assert abs(first.penalty + 1.90) < 0.02, f"penalty {first.penalty:.3f}"
    assert abs(low + 2.07) < 0.02 and abs(high + 1.73) < 0.02, f"ci [{low:.3f}, {high:.3f}]"
    assert high < 0, "ci includes 0"
    assert abs(first.slope_test_statistic - 4.38) < 0.05, f"slope stat {first.slope_test_statistic:.3f}"
    assert first.slope_test_df == 8, f"df {first.slope_test_df}"
    assert abs(first.slope_test_p - 0.82) < 0.01, f"p {first.slope_test_p:.3f}"
    assert (first.flipped_in, first.flipped_out) == (651, 651), "report flips"
    assert (first.flipped_in_remote, first.flipped_out_centre) == (432, 577), "report flip groups"
    assert correction_report(ctx["history"], share) == first, "same-seed replay differs"
    warnings = report_warnings(first)
    assert warnings == [], f"warnings {warnings}"
    ctx["report"] = first
    return (f"penalty={first.penalty:.3f} ci=[{low:.3f}, {high:.3f}] slope={first.slope_test_statistic:.2f} "
            f"df={first.slope_test_df} p={first.slope_test_p:.3f} replay equal warnings=[]")


def check_decide_skips_bootstrap(ctx) -> str:
    import src.policy.label_correction as module
    from src.harness import decide

    def boom(*args, **kwargs):
        raise AssertionError("bootstrap called on decision path")

    saved = module.correction_report, module._penalty_interval
    module.correction_report = module._penalty_interval = boom
    try:
        record = decide(ctx["history"], ctx["batch"], input_hashes(), config=panel_config())
    finally:
        module.correction_report, module._penalty_interval = saved
    assert record.status == "published", f"status {record.status}"
    assert record.label_correction is None, "harness decide computed label_correction"
    return "harness decide ran with correction_report and bootstrap patched to raise"


def check_tiny_history_report(ctx) -> str:
    import dataclasses

    from src.policy import correction_report, is_remote, report_warnings
    PENALTY_NOT_DISTINGUISHABLE_WARNING = "penalty not distinguishable from zero"

    history = ctx["history"]
    remote = is_remote(history).astype(bool)
    granted = history["decision_octroi"].to_numpy() == 1
    picks = []
    for is_remote_group, is_granted, count in ((True, True, 2), (False, True, 2), (True, False, 3), (False, False, 3)):
        picks.extend(np.flatnonzero((remote == is_remote_group) & (granted == is_granted))[:count].tolist())
    tiny = history.iloc[picks].reset_index(drop=True)
    assert len(tiny) == 10 and int(tiny["decision_octroi"].sum()) == 4, "tiny history shape"
    assert 0 < int(is_remote(tiny).sum()) < len(tiny), "tiny history lacks a region"
    result = correction_report(tiny, 0.4)
    assert result.penalty_ci is None or len(result.penalty_ci) == 2, f"ci {result.penalty_ci}"
    spans_zero = result.penalty_ci is None or result.penalty_ci[0] <= 0 <= result.penalty_ci[1]
    assert (PENALTY_NOT_DISTINGUISHABLE_WARNING in report_warnings(result)) == spans_zero, f"warnings {report_warnings(result)} ci {result.penalty_ci}"
    unbootstrapped = correction_report(tiny, 0.4, n_boot=0)
    assert unbootstrapped.penalty_ci is None, f"ci {unbootstrapped.penalty_ci}"
    assert PENALTY_NOT_DISTINGUISHABLE_WARNING in report_warnings(unbootstrapped), "no warning for ci None"
    assert json.loads(json.dumps(dataclasses.asdict(unbootstrapped)))["penalty_ci"] is None, "penalty_ci not null in json"
    return f"correction_report on 10 rows / 4 grants ok ci={result.penalty_ci}; n_boot=0 -> ci None, warning, json null"


def check_reference_consistency(ctx) -> str:
    from src.policy import budget_share, correct_labels, is_remote, reference_labels
    from src.policy import CommitteeModel

    history = ctx["history"]
    share = budget_share(history)
    swapped = history.copy()
    remote = is_remote(history).astype(bool)
    swapped["region_administrative"] = np.where(remote, "Montreal", "Cote-Nord")
    penalties = {}
    for name, frame in (("current", history), ("swapped", swapped)):
        penalties[name] = float(CommitteeModel().fit(frame, frame["decision_octroi"].to_numpy()).remote_penalty_)
        reference = reference_labels(frame, frame, share)["corrected"]
        labels = correct_labels(frame, share).labels
        assert np.array_equal(reference, labels), f"{name}: reference differs from training labels on {int((reference != labels).sum())} rows"
    assert penalties["current"] < 0, f"current penalty {penalties['current']:.3f}"
    assert penalties["swapped"] > 0, f"swapped penalty {penalties['swapped']:.3f} not positive"
    return f"reference == training labels; penalty current={penalties['current']:+.3f} swapped={penalties['swapped']:+.3f}"


def check_tune_requires_penalty(ctx) -> str:
    from src.evaluation import SEARCH_SPACE, tune
    from src.policy import budget_share, is_remote

    history = ctx["history"].copy()
    remote = is_remote(history).astype(bool)
    history["region_administrative"] = np.where(remote, "Montreal", "Cote-Nord")
    try:
        tune(history, budget_share(history), SEARCH_SPACE[:1], splits=1, workers=1)
    except ValueError as error:
        assert str(error) == "no regional penalty: equity undefined", f"message {error}"
    else:
        raise AssertionError("tune returned a table without a regional penalty")
    return "region-swapped history (penalty > 0) -> ValueError, no -inf table"


def check_fit_uses_history_rate(ctx) -> str:
    from sklearn.model_selection import train_test_split

    from src.policy import FairPipeline, budget_share

    history = ctx["history"]
    share = budget_share(history)
    for seed in range(50):
        train, _ = train_test_split(history, train_size=0.7, stratify=history["region_administrative"], random_state=seed)
        grants = int(train["decision_octroi"].sum())
        if grants != round(share * len(train)):
            break
    else:
        raise AssertionError("no split with differing grant count")
    assert grants != round(share * len(train)), "split not discriminating"
    pipeline = FairPipeline(panel_config(), share).fit(train)
    correction = pipeline.label_correction_
    assert correction.k == grants, f"k {correction.k} != train grants {grants} (share-based {round(share * len(train))})"
    assert int(correction.labels.sum()) == grants, f"labels {int(correction.labels.sum())} != train grants {grants}"
    return f"seed={seed} n_train={len(train)} train_grants={grants} share_based={round(share * len(train))} k={correction.k}"


def check_cli_record(ctx) -> str:
    out = ctx["tmp"] / "cli_record"
    result = run_cli("decide", *CLI_FLAGS, *PANEL_CLI, "--out-dir", str(out))
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr.splitlines()[-1:]}"
    ctx["cli_record_dir"] = out
    summary = json.loads((out / "decision_record.json").read_text())
    correction = summary["label_correction"]
    assert correction is not None, "label_correction missing"
    low, high = correction["penalty_ci"]
    assert abs(correction["penalty"] + 1.90) < 0.02, f"penalty {correction['penalty']:.3f}"
    assert abs(low + 2.07) < 0.02 and abs(high + 1.73) < 0.02, f"ci [{low:.3f}, {high:.3f}]"
    assert abs(correction["slope_test_statistic"] - 4.38) < 0.05, f"slope {correction['slope_test_statistic']:.3f}"
    assert correction["slope_test_df"] == 8, f"df {correction['slope_test_df']}"
    assert abs(correction["slope_test_p"] - 0.82) < 0.01, f"p {correction['slope_test_p']:.3f}"
    assert (correction["flipped_in"], correction["flipped_out"]) == (651, 651), "flips"
    assert summary["warnings"] == [], f"warnings {summary['warnings']}"
    assert summary["status"] == "published" and (out / "predictions.csv").exists(), "not published"
    return f"CLI record label_correction penalty={correction['penalty']:.3f} warnings=[] exit=0"


def check_cli_replay(ctx) -> str:
    first = ctx["cli_record_dir"]
    second = ctx["tmp"] / "cli_record_b"
    result = run_cli("decide", *CLI_FLAGS, *PANEL_CLI, "--out-dir", str(second))
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr.splitlines()[-1:]}"
    left, right = [(directory / "decision_record.json").read_bytes() for directory in (first, second)]
    assert left == right, "decision_record.json differs between two CLI runs"
    assert b"penalty_ci" in left and b"slope_test_p" in left, "record lacks bootstrap CI or slope test"
    return f"decision_record.json byte-identical sha256={sha256_bytes(left)[:8]} bytes={len(left)}"


def check_postprocessing_default(ctx) -> str:
    record = ctx["baseline"]
    assert record.status == "published", f"status {record.status}"
    assert int(record.decisions.sum()) == 1598, f"grants {int(record.decisions.sum())}"
    assert record.offset == 0, f"offset {record.offset}"
    assert record.output_issues == [], f"issues {record.output_issues}"
    assert record.offset_moved_ids == [], f"offset_moved_ids {len(record.offset_moved_ids)}"
    assert record.jury["triggered"] > 0 and "reasons" in record.jury, f"jury {record.jury}"
    return f"published grants=1598 offset=0 issues=[] jury_moved={len(record.jury_moved_ids)} triggered={record.jury['triggered']}"


def injected_nodes(corrupt, target="final_outcome"):
    from dataclasses import replace

    from src.common.graph import Node
    from src.harness import POSTPROCESSING_NODES

    def corrupted(*args):
        outcome = next(node for node in POSTPROCESSING_NODES if node.name == target).fn(*args)
        return replace(outcome, decisions=corrupt(outcome.decisions.copy()))

    inputs = next(node for node in POSTPROCESSING_NODES if node.name == target).inputs
    return tuple(Node(target, corrupted, inputs) if node.name == target else node for node in POSTPROCESSING_NODES)


def check_output_guard(ctx) -> str:
    from src.harness import output_issues

    batch = ctx["batch"]
    ids = batch["id_candidat"].to_numpy()
    k = 1598
    valid = np.zeros(len(batch), dtype=int)
    valid[:k] = 1
    assert output_issues(batch, ids, valid, k) == [], "valid output flagged"
    duplicate = ids.copy()
    duplicate[1] = duplicate[0]
    reordered = ids.copy()
    reordered[[0, 1]] = reordered[[1, 0]]
    direct = {
        "duplicate_id": output_issues(batch, duplicate, valid, k),
        "wrong_order": output_issues(batch, reordered, valid, k),
        "value_2": output_issues(batch, ids, np.where(np.arange(len(batch)) == 0, 2, valid), k),
        "count_k_plus_1": output_issues(batch, ids, np.where(np.arange(len(batch)) == k, 1, valid), k),
    }
    empty = [name for name, issues in direct.items() if not issues]
    assert not empty, f"not flagged: {empty}"

    def value_two(decisions):
        decisions[np.flatnonzero(decisions == 1)[0]] = 2
        return decisions

    def extra_grant(decisions):
        decisions[np.flatnonzero(decisions == 0)[0]] = 1
        return decisions

    notes = []
    for name, corrupt in (("value_2", value_two), ("count_k_plus_1", extra_grant)):
        out = make_sentinel_dir(ctx["tmp"], f"guard_{name}")
        record = decide_and_write(ctx["history"], batch, out, nodes=injected_nodes(corrupt))
        blocks = [action for action in record.actions if getattr(action.kind, "value", action.kind) == "BLOCK"]
        assert record.status == "blocked", f"{name}: status {record.status}"
        assert len(blocks) == 1, f"{name}: {len(blocks)} BLOCK actions"
        assert "output" in blocks[0].params["checks"], f"{name}: checks {blocks[0].params['checks']}"
        assert record.output_issues, f"{name}: no output_issues"
        assert (out / "predictions.csv").read_bytes() == SENTINEL, f"{name}: sentinel predictions.csv changed"
        notes.append(f"{name}=blocked")
    return "direct issues flagged for " + ",".join(direct) + "; injected final_outcome " + " ".join(notes) + "; sentinel intact (CLI not injected)"


def check_output_cases(ctx) -> str:
    from src.harness import output_issues

    batch = ctx["batch"]
    n, k = len(batch), 1598
    ids = batch["id_candidat"].to_numpy()
    valid = np.zeros(n, dtype=int)
    valid[:k] = 1
    missing_ids = pd.array(np.arange(n), dtype="Int64")
    missing_ids[0] = pd.NA
    duplicated = batch.copy()
    duplicated.loc[duplicated.index[1], "id_candidat"] = duplicated.loc[duplicated.index[0], "id_candidat"]
    cases = {
        "decisions_2d": (batch, ids, np.zeros((n, 2), dtype=int)),
        "decisions_bool": (batch, ids, valid.astype(bool)),
        "decisions_0d": (batch, ids, np.array(1)),
        "ids_pd_na": (batch, missing_ids, valid),
        "ids_2d": (batch, np.stack([ids, ids], axis=1), valid),
    }
    for name, (frame, case_ids, decisions) in cases.items():
        try:
            issues = output_issues(frame, case_ids, decisions, k)
        except Exception as error:
            raise AssertionError(f"{name} raised {type(error).__name__}: {error}")
        assert issues, f"{name} not flagged"
    isolated = output_issues(duplicated, duplicated["id_candidat"].to_numpy(), valid, k)
    assert isolated == ["ids are not unique"], f"uniqueness rule not isolated: {isolated}"
    return "2-D, bool, 0-d decisions and pd.NA, 2-D ids flagged without raising; duplicate ids give only 'not unique'"


def check_offset_bound(ctx) -> str:
    from src.harness import OFFSET_GRID
    from src.policy import OFFSET_BOUND

    pipeline, batch = fitted_pipeline(ctx), ctx["batch"]
    for name, call in (("score", pipeline.score), ("predict", pipeline.predict), ("decide", pipeline.decide)):
        for offset in (0.11, -0.11):
            try:
                call(batch, offset)
            except ValueError:
                continue
            raise AssertionError(f"{name}({offset}) accepted")
        for offset in (float("nan"), float("inf"), float("-inf")):
            try:
                call(batch, offset)
            except ValueError:
                continue
            raise AssertionError(f"{name}({offset}) accepted")
        call(batch, 0.10)
        call(batch, -0.10)
    assert OFFSET_BOUND == 0.10, f"OFFSET_BOUND {OFFSET_BOUND}"
    assert len(OFFSET_GRID) == 41, f"grid points {len(OFFSET_GRID)}"
    assert float(np.max(np.abs(OFFSET_GRID))) <= OFFSET_BOUND + 1e-12, "grid beyond bound"
    assert np.allclose(OFFSET_GRID, np.linspace(-0.10, 0.10, 41)), "grid values changed"
    return "score/predict/decide reject +-0.11, NaN, +-inf, accept +-0.10; grid 41 points within bound"


def check_explain_fields(ctx) -> str:
    table = ctx["baseline"].explanations
    assert {"proposed_decision", "final_rank"} <= set(table.columns), f"columns {list(table.columns)}"
    n = len(table)
    assert sorted(table["final_rank"]) == list(range(1, n + 1)), "final_rank not a permutation of 1..n"
    assert set(table["proposed_decision"].unique()) <= {0, 1}, "proposed_decision not binary"
    assert int(table["proposed_decision"].sum()) == int(ctx["baseline"].proposed.sum()), "proposed_decision mismatch"
    return f"proposed_decision and final_rank present; final_rank permutation of 1..{n}"


def check_config_option(ctx) -> str:
    from src.harness import decide
    from src.policy import (AUDIT_PANEL_CONFIG, CONFIGS, CONSENSUS_PANEL_CONFIG, DECLARED_CONFIG, INCOME_BLIND_CONFIG,
                            INCOME_BLIND_NO_JURY_CONFIG, MODEL_JURY_CONFIG, SCORING_FEATURES,
                            ACTIVE_BOTH_CONFIG, ACTIVE_JURY_CONFIG, ACTIVE_MODEL_JURY_CONFIG, ACTIVE_REASONING_CONFIG,
                            ACTIVE_RESIDUAL_JURY_CONFIG, ACTIVE_SAFE_JURY_CONFIG, SINGLE_RESIDUAL_CONFIG, VALIDATOR_JURY_CONFIG, FairPipeline, budget_share, scoring_features)

    history, batch = ctx["history"], ctx["batch"]
    assert DECLARED_CONFIG.residual_blend is not None and DECLARED_CONFIG.features == SCORING_FEATURES, "declared config"
    assert VALIDATOR_JURY_CONFIG.discounts == () and VALIDATOR_JURY_CONFIG.features == SCORING_FEATURES, "validator jury changed"
    assert INCOME_BLIND_CONFIG.features == ["cote_r", "heures_travail"], f"features {INCOME_BLIND_CONFIG.features}"
    assert {c.name: c for c in (VALIDATOR_JURY_CONFIG, INCOME_BLIND_CONFIG,
                              INCOME_BLIND_NO_JURY_CONFIG, MODEL_JURY_CONFIG, CONSENSUS_PANEL_CONFIG,
                              AUDIT_PANEL_CONFIG, SINGLE_RESIDUAL_CONFIG, DECLARED_CONFIG, ACTIVE_BOTH_CONFIG,
                              ACTIVE_JURY_CONFIG, ACTIVE_REASONING_CONFIG, ACTIVE_RESIDUAL_JURY_CONFIG,
                              ACTIVE_MODEL_JURY_CONFIG, ACTIVE_SAFE_JURY_CONFIG)} == CONFIGS, "CONFIGS registry"
    declared = ctx["baseline"]
    share = budget_share(history)
    for config in (INCOME_BLIND_CONFIG,):
        record = decide(history, batch, input_hashes(), config=config)
        assert record.status in ("published", "blocked") and int(record.decisions.sum()) == 1598, f"{config.name}: {record.status}"
        assert record.actions[0].params == {"config": config.name}, f"{config.name}: {record.actions[0].params}"
        assert "ADJUST_OFFSET" not in action_kinds(record), f"{config.name}: {action_kinds(record)}"
        assert not np.array_equal(record.decisions, declared.decisions), f"{config.name}: decisions equal panel"
        pipeline = FairPipeline(config, share).fit(history)
        assert not np.array_equal(pipeline.label_correction_.labels, ctx["correction"].labels), f"{config.name}: labels equal declared"
        model = pipeline.main_model_
        reconstructed = pipeline.contributions(batch).to_numpy().sum(axis=1) + float(model[-1].intercept_[0])
        error = float(np.max(np.abs(reconstructed - model.decision_function(pipeline.features(batch)))))
        assert error < 1e-9, f"{config.name}: explain identity error {error:.3g}"
        ctx[f"record:{config.name}"] = record
    blind = FairPipeline(INCOME_BLIND_CONFIG, share).fit(history)
    shuffled = batch.copy()
    shuffled["revenu_familial_estime"] = np.random.default_rng(3).permutation(batch["revenu_familial_estime"].to_numpy())
    assert np.array_equal(blind.score(batch, 0.0), blind.score(shuffled, 0.0)), "income-blind scores read income"
    factors = ctx[f"record:{INCOME_BLIND_CONFIG.name}"].explanations["factor_3"]
    assert (factors == "").all(), "income-blind explanations have a third factor"
    assert scoring_features(batch).shape[1] == 3, "scoring schema changed"
    return "income-blind: 2 features, income-invariant scores, factor_3 empty, decides 1598 grants; registry complete"


REGION_COLUMNS = ["region_administrative", "code_postal_3", "distance_domicile_campus_km"]
REGION_MARKERS = ("region", "postal", "distance")


def model_jury_pipeline(ctx):
    from src.policy import MODEL_JURY_CONFIG, FairPipeline, budget_share

    if "model_jury_pipeline" not in ctx:
        ctx["model_jury_pipeline"] = FairPipeline(MODEL_JURY_CONFIG, budget_share(ctx["history"])).fit(ctx["history"])
    return ctx["model_jury_pipeline"]


def check_model_jury_decide(ctx) -> str:
    from src.policy import MODEL_JURORS

    out = ctx["tmp"] / "model_jury"
    result = run_cli("decide", *CLI_FLAGS, "--config", "model jury", "--out-dir", str(out))
    assert result.returncode in (0, 3), f"exit {result.returncode}: {result.stderr.splitlines()[-1:]}"
    summary = load_strict(out / "decision_record.json")
    assert summary["config"]["name"] == "model jury", f"config {summary['config']}"
    assert summary["status"] == ("published" if result.returncode == 0 else "blocked"), f"status {summary['status']}"
    grants = summary["grants"]
    assert grants == 1598, f"grants {grants}"
    jury = summary["jury"]
    assert jury["triggered"] > 0, "jury triggered nothing"
    assert jury["overturned_out"] == jury["overturned_in"], f"out {jury['overturned_out']} != in {jury['overturned_in']}"
    outcome = model_jury_pipeline(ctx).decide(ctx["batch"])
    missing = set(MODEL_JURORS) - set(outcome.votes)
    assert not missing, f"jurors without votes: {sorted(missing)}"
    return f"published grants={grants} triggered={jury['triggered']} swaps={jury['overturned_out']} jurors={len(outcome.votes)}"


def check_model_jury_determinism(ctx) -> str:
    from src.policy import MODEL_JURY_CONFIG, FairPipeline, budget_share

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    first = FairPipeline(MODEL_JURY_CONFIG, share).fit(history).decide(batch)
    second = FairPipeline(MODEL_JURY_CONFIG, share).fit(history).decide(batch)
    assert np.array_equal(first.decisions, second.decisions), "decisions differ across fits"
    return f"two fits give identical decisions over {len(batch)} rows"


def check_model_jury_region_invariance(ctx) -> str:
    pipeline, batch = model_jury_pipeline(ctx), ctx["batch"]
    order = np.random.default_rng(13).permutation(len(batch))
    shuffled = batch.copy()
    shuffled[REGION_COLUMNS] = batch[REGION_COLUMNS].to_numpy()[order]
    assert not shuffled[REGION_COLUMNS].equals(batch[REGION_COLUMNS]), "permutation left columns unchanged"
    assert np.array_equal(pipeline.decide(batch, 0.0).decisions, pipeline.decide(shuffled, 0.0).decisions), "decisions differ (I2)"
    return "decisions identical at offset 0 after joint shuffle of region, postal, distance"


def check_model_jurors_region_blind(ctx) -> str:
    from src.policy import MODEL_JURORS

    leaks = {name: [feature for feature in schema if any(marker in feature for marker in REGION_MARKERS)]
             for name, (schema, _) in MODEL_JURORS.items()}
    leaks = {name: features for name, features in leaks.items() if features}
    assert not leaks, f"region features in schemas: {leaks}"
    return f"{len(MODEL_JURORS)} model juror schemas free of region, postal, distance"


def check_blend_config(ctx) -> str:
    from src.policy import Config, FairPipeline, budget_share

    config = Config("blend check", blend=("boosting", "neural_net"))
    pipeline = FairPipeline(config, budget_share(ctx["history"])).fit(ctx["history"])
    grants = int(pipeline.decide(ctx["batch"]).decisions.sum())
    assert grants == 1598, f"grants {grants}"
    return f"blend {'+'.join(config.blend)} decides with {grants} grants"


def guard_checks(record, prefix: str) -> list:
    return [check for check in record.jury_guard.checks if check.name.startswith(prefix)]


def check_jury_guard_declared(ctx) -> str:
    record = ctx["baseline_decide"](config=validator_config())
    ctx["declared_record"] = record
    summary = record.summary()["jury_guard"]
    names = [check["name"] for check in summary["checks"]]
    assert ("REVERT_JURY" in action_kinds(record)) == (summary["status"] == "ALERT"), f"actions {action_kinds(record)}"
    assert not guard_checks(record, "juror quality") and not guard_checks(record, "juror region"), f"model checks {names}"
    for prefix in ("jury fairness effect", "jury swap volume", "juror agreement"):
        assert any(name.startswith(prefix) for name in names), f"missing {prefix}: {names}"
    return f"guard {summary['status']} " + " ".join(f"{check['value']:.4f}" for check in summary["checks"])


def check_jury_guard_model(ctx) -> str:
    from src.policy import MODEL_JURORS, MODEL_JURY_CONFIG

    record = ctx["baseline_decide"](config=MODEL_JURY_CONFIG)
    ctx["model_jury_record"] = record
    quality, leakage = guard_checks(record, "juror quality"), guard_checks(record, "juror region leakage")
    assert len(quality) == len(leakage) == len(MODEL_JURORS), f"{len(quality)} quality, {len(leakage)} leakage checks"
    assert all(0.5 < check.value <= 1.0 for check in quality), [check.value for check in quality]
    assert record.jury_guard.status != "ALERT" and "REVERT_JURY" not in action_kinds(record), action_kinds(record)
    assert int(record.decisions.sum()) == 1598, f"{record.status} {int(record.decisions.sum())}"
    return (f"guard {record.jury_guard.status} status {record.status} AUC " + " ".join(f"{check.name.split(': ')[1]}={check.value:.4f}" for check in quality)
            + f"; max leakage {max(check.value for check in leakage):+.4f}")


def revert_consistency(record) -> str:
    kinds = action_kinds(record)
    assert kinds.count("REVERT_JURY") == 1, kinds
    assert record.status != "blocked" or kinds[-1] == "BLOCK", kinds
    assert int(record.decisions.sum()) == 1598, f"grants {int(record.decisions.sum())}"
    assert np.array_equal(record.decisions, record.proposed) and not record.jury_moved_ids, "jury still moved decisions"
    assert record.jury["overturned_out"] == record.jury["overturned_in"] == 0, record.jury
    assert (record.status == "blocked") == ("BLOCK" in kinds), f"status {record.status} with {kinds}"
    assert record.verdicts[-1].status != "ALERT" or record.status == "blocked", "published with ALERT audit"
    return f"{kinds} status={record.status}"


def check_jury_guard_revert(ctx) -> str:
    import src.harness.jury_guard as guard_module

    original = guard_module.SWAP_SHARE_ALERT
    guard_module.SWAP_SHARE_ALERT = 0.001
    try:
        record = ctx["baseline_decide"](config=validator_config())
    finally:
        guard_module.SWAP_SHARE_ALERT = original
    revert = next(action for action in record.actions if action.kind.value == "REVERT_JURY")
    assert "jury swap volume" in revert.params["checks"], revert.params
    return f"swap threshold 0.001: {revert_consistency(record)}"


@contextlib.contextmanager
def leaky_juror(orientation):
    from src.policy import MODEL_JURORS, Config, FairPipeline, JurySettings, is_remote

    original = FairPipeline.juror_scores

    def leaky_scores(self, df):
        scores = original(self, df)
        scores["leaky"] = orientation * is_remote(df) + 0.01 * scores["leaky"]
        return scores

    MODEL_JURORS["leaky"] = MODEL_JURORS["lr_all"]
    FairPipeline.juror_scores = leaky_scores
    try:
        yield Config("leaky jury", jury=JurySettings(quorum=0.8, jurors=("merit", "leaky")))
    finally:
        FairPipeline.juror_scores = original
        del MODEL_JURORS["leaky"]


def check_jury_guard_leaky_juror(ctx) -> str:
    with leaky_juror(1) as config:
        record = ctx["baseline_decide"](config=config)
    revert = next(action for action in record.actions if action.kind.value == "REVERT_JURY")
    assert "juror region leakage: leaky" in revert.params["checks"], revert.params
    return f"leaky juror reverted ({revert.params['checks']}): {revert_consistency(record)}"


def check_jury_guard_orientation_free(ctx) -> str:
    with leaky_juror(-1) as config:
        record = ctx["baseline_decide"](config=config)
    leak = next(check for check in record.jury_guard.checks if check.name == "juror region leakage: leaky")
    revert = next(action for action in record.actions if action.kind.value == "REVERT_JURY")
    assert leak.status == "ALERT" and leak.value > 0.05, f"inverted leak {leak.status} {leak.value:+.3f}"
    assert "absolute leakage" in leak.detail, leak.detail
    assert "juror region leakage: leaky" in revert.params["checks"], revert.params
    return f"inverted region juror: leakage {leak.value:+.3f} ALERT, reverted"


def check_revert_refits_offset(ctx) -> str:
    import src.harness.postprocessing as postprocessing

    original = postprocessing.fit_offset
    calls = []

    def recording(*args, **kwargs):
        offset = original(*args, **kwargs)
        calls.append((kwargs.get("jury", True), offset))
        return offset

    postprocessing.fit_offset = recording
    try:
        with leaky_juror(1) as config:
            record = ctx["baseline_decide"](config=config, eo_gap_alert=FORCED_ALERT)
    finally:
        postprocessing.fit_offset = original
    kinds = action_kinds(record)
    assert "REVERT_JURY" in kinds and calls[-1][0] is False, f"calls {calls} actions {kinds}"
    assert record.offset == calls[-1][1], f"offset {record.offset} != re-fitted {calls[-1][1]}"
    revert = next(action for action in record.actions if action.kind.value == "REVERT_JURY")
    assert revert.params["offset"] == record.offset, revert.params
    assert np.array_equal(record.decisions, record.proposed), "jury still active after revert"
    return f"fit_offset calls (jury, offset) {calls}; final offset {record.offset:+.3f} chosen with jury disabled"


def check_consensus_target_labels(ctx) -> str:
    from src.policy import (COMMITTEE_FEATURES, CONSENSUS_PANEL_CONFIG, DECLARED_HOURS_WEIGHT, DECLARED_INCOME_WEIGHT,
                            FairPipeline, budget_share)

    history = ctx["history"]
    share = budget_share(history)
    pipeline = FairPipeline(CONSENSUS_PANEL_CONFIG, share).fit(history)
    committee = pipeline.committee_
    score = committee.rule_score(history, DECLARED_INCOME_WEIGHT, DECLARED_HOURS_WEIGHT)
    k = int(round(share * len(history)))
    expected = np.zeros(len(history), dtype=int)
    expected[np.argsort(-score, kind="stable")[:k]] = 1
    labels = pipeline.training_labels_
    assert np.array_equal(labels, expected) and int(labels.sum()) == k, "labels are not the top-k consensus scores"
    weights = committee.rule_weights(DECLARED_INCOME_WEIGHT, DECLARED_HOURS_WEIGHT)
    names = [name for name, weight in zip(COMMITTEE_FEATURES, weights) if weight != 0]
    assert names == ["cote_r", "log_revenu", "heures_travail"] and (weights >= 0).all(), f"weighted criteria {names}"
    assert weights[COMMITTEE_FEATURES.index("cote_r")] == 1.0, "rule not expressed in R units"
    assert pipeline.config.features == ["cote_r", "log_revenu", "heures_travail"], pipeline.config.features
    report = ctx["baseline"].training_labels
    historical = history["decision_octroi"].to_numpy()
    assert report["differs_from_committee"] == int((labels != historical).sum()), report
    assert report["differs_from_corrected"] == int((labels != pipeline.label_correction_.labels).sum()), report
    assert pipeline.label_correction_.k == 3994 and report["target"] == "consensus", report
    return (f"labels = top-{k} of the consensus score; main features {pipeline.config.features}; differs from committee "
            f"{report['differs_from_committee']}, from corrected {report['differs_from_corrected']}")


def check_consensus_panel_decide(ctx) -> str:
    from src.policy import REFERENCE_JURORS, FairPipeline, budget_share

    history, batch = ctx["history"], ctx["batch"]
    record = ctx["baseline"]
    assert record.status == "published" and int(record.decisions.sum()) == 1598, f"{record.status}"
    assert record.verdicts[-1].status != "ALERT", f"monitoring {record.verdicts[-1].status}"
    assert record.jury_guard.status != "ALERT" and record.consensus_guard.status != "ALERT", \
        f"guards {record.jury_guard.status} {record.consensus_guard.status}"
    pipeline = fitted_pipeline(ctx, panel_config())
    assert set(REFERENCE_JURORS) <= set(pipeline.decide(batch).votes), "reference jurors did not vote"
    again = FairPipeline(panel_config(), budget_share(history)).fit(history).decide(batch).decisions
    assert np.array_equal(pipeline.decide(batch).decisions, again), "decisions differ across fits"
    order = np.random.default_rng(23).permutation(len(batch))
    shuffled = batch.copy()
    shuffled[REGION_COLUMNS] = batch[REGION_COLUMNS].to_numpy()[order]
    assert not shuffled[REGION_COLUMNS].equals(batch[REGION_COLUMNS]), "permutation left columns unchanged"
    assert np.array_equal(pipeline.decide(batch).decisions, pipeline.decide(shuffled).decisions), "decisions differ (I2)"
    return (f"published 1598 grants, deterministic, region-invariant; monitoring {record.verdicts[-1].status}, "
            f"jury guard {record.jury_guard.status}, consensus guard {record.consensus_guard.status}")




def check_monitoring_signed_gap(ctx) -> str:
    from src.monitoring import EO_GAP_ALERT, opportunity_checks
    from src.policy import allocate, budget_share, is_remote, reference_labels

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    references, remote = reference_labels(history, batch, share), is_remote(batch)
    z = (batch["cote_r_equivalent"] - history["cote_r_equivalent"].mean()) / history["cote_r_equivalent"].std()
    cases = {
        "consensus rule": references["consensus"],
        "remote favoured": allocate((z + 0.8 * remote).to_numpy(), share),
        "centre favoured": allocate((z - 0.8 * remote).to_numpy(), share),
    }
    merit = {name: opportunity_checks(decisions, references, remote, EO_GAP_ALERT)[0] for name, decisions in cases.items()}
    assert merit["consensus rule"].status == "OK" and -0.09 <= merit["consensus rule"].value < 0, merit["consensus rule"]
    assert merit["remote favoured"].status == "ALERT" and merit["remote favoured"].value < -0.09, merit["remote favoured"]
    assert merit["centre favoured"].status == "ALERT" and merit["centre favoured"].value > EO_GAP_ALERT, merit["centre favoured"]
    assert all(check.correctable for check in merit.values()), "merit check not correctable"
    assert "absolute" in merit["consensus rule"].detail, merit["consensus rule"].detail
    return " ".join(f"{name}: g={check.value:+.4f} {check.status}" for name, check in merit.items())


def check_correctable_consensus_alert(ctx) -> str:
    from src.harness.postprocessing import correctable_alerts
    from src.monitoring import Check, Verdict

    def verdict(*checks):
        return Verdict("ALERT" if any(check.status == "ALERT" for check in checks) else "OK", list(checks))

    rate = Check("consensus: grant rate remote", 0.3, "", "ALERT", correctable=True)
    agreement = Check("consensus: mean agreement", 0.8, "", "ALERT", correctable=False)
    quiet = verdict(Check("budget", 0.4, "", "OK"))
    assert correctable_alerts(quiet, verdict(rate)), "correctable consensus rate alert ignored"
    assert not correctable_alerts(quiet, verdict(rate, agreement)), "non-correctable alert allowed correction"
    assert not correctable_alerts(quiet, quiet), "no alert still corrected"
    from src.harness.consensus import consensus_guard
    from src.policy import allocate, budget_share, is_remote

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    committee = committee_of(ctx)
    tilted = allocate(committee.rule_score(batch, 0.0) - 1.2 * is_remote(batch), share)
    guard = consensus_guard(committee, history, batch, share, tilted)
    alerts = [check for check in guard.checks if check.status == "ALERT"]
    assert alerts and all(check.correctable for check in alerts if "rate" in check.name or "gap" in check.name), alerts
    return f"{len(alerts)} consensus alerts on a centre-tilted ranking, rate and gap checks correctable"


def check_offset_feasibility_first(ctx) -> str:
    from src.harness import OFFSET_GRID
    from src.harness.consensus import consensus_guard
    from src.harness.postprocessing import fit_offset
    from src.monitoring import opportunity_checks
    from src.policy import allocate, budget_share, is_remote, reference_labels

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    committee, remote = committee_of(ctx), is_remote(batch)
    references = reference_labels(history, batch, share)
    base = committee.rule_score(batch, 0.0) - 0.15 * remote

    class Stub:
        class config:
            target = "consensus"

        committee_ = committee

        def decide(self, df, offset, jury=True):
            from types import SimpleNamespace

            return SimpleNamespace(decisions=allocate(base + 5 * offset * remote, share))

    def feasible(offset, alert):
        decisions = Stub().decide(batch, offset).decisions
        checks = [*opportunity_checks(decisions, references, remote, alert, False),
                  *consensus_guard(committee, history, batch, share, decisions).checks]
        return not any(check.status == "ALERT" and check.correctable for check in checks)

    gaps = {float(offset): float(opportunity_checks(Stub().decide(batch, offset).decisions, references, remote, 1.0, False)[0].value)
            for offset in OFFSET_GRID}
    alert = gaps[0.0] - 0.0004
    chosen = fit_offset(Stub(), history, batch, share, alert)
    reachable = [float(offset) for offset in OFFSET_GRID if feasible(float(offset), alert)]
    assert reachable and chosen != 0, f"no feasible offset or none chosen: {chosen}"
    assert feasible(chosen, alert), f"offset {chosen} infeasible"
    assert abs(chosen) == min(abs(offset) for offset in reachable), f"{chosen} is not the smallest of {reachable[:6]}"
    return f"alert {alert:.4f} (unrounded): chosen offset {chosen:+.3f} is the smallest feasible"


def check_cli_config(ctx) -> str:
    out = ctx["tmp"] / "cli_config"
    result = run_cli("decide", *CLI_FLAGS, "--config", "income-blind validator jury", "--out-dir", str(out))
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr.splitlines()[-1:]}"
    summary = load_strict(out / "decision_record.json")
    assert summary["config"]["name"] == "income-blind validator jury", f"config {summary['config']}"
    predictions = pd.read_csv(out / "predictions.csv")
    declared = pd.read_csv(ctx["cli_record_dir"] / "predictions.csv")
    changed = int((predictions["decision_octroi"] != declared["decision_octroi"]).sum())
    assert 0 < changed < 400 and int(predictions["decision_octroi"].sum()) == 1598, f"changed {changed}"
    bogus = ctx["tmp"] / "cli_config_bogus"
    result = run_cli("decide", *CLI_FLAGS, "--config", "bogus", "--out-dir", str(bogus))
    assert result.returncode == 1 and "Traceback" not in result.stderr, f"exit {result.returncode}"
    assert not bogus.exists() or not any(bogus.iterdir()), "bogus config wrote files"
    return f"--config run published, {changed} decisions differ from declared; unknown name exit 1, nothing written"


CONSENSUS_JSON = ROOT / "docs/reviews/consensus.json"


def committee_of(ctx):
    from src.policy import CommitteeModel

    if "committee" not in ctx:
        ctx["committee"] = CommitteeModel().fit(ctx["history"], ctx["history"]["decision_octroi"].to_numpy())
    return ctx["committee"]


def check_consensus_constants(ctx) -> str:
    from src.monitoring import EO_GAP_ALERT
    from src.policy import (DECLARED_HOURS_WEIGHT, GRANT_RATE_BANDS, GRANT_RATE_STRICT_BANDS, LIMITS,
                            REVIEWER_INCOME_WEIGHTS, STRICT_LIMITS, reference_weights)

    expected = json.loads(CONSENSUS_JSON.read_text())

    def as_tuples(bands):
        return {group: tuple(band) for group, band in bands.items()}

    reviewer_income = {name: expected["reference_rules"][name]["log_revenu"] for name in REVIEWER_INCOME_WEIGHTS}
    pairs = {
        "reviewer_income_weights": (dict(REVIEWER_INCOME_WEIGHTS), reviewer_income),
        "grant_rate": (GRANT_RATE_BANDS, as_tuples(expected["grant_rate"])),
        "grant_rate_strict": (GRANT_RATE_STRICT_BANDS, as_tuples(expected["grant_rate_strict"])),
        "limits": (LIMITS, expected["limits"]),
        "strict_limits": (STRICT_LIMITS, expected["strict_limits"]),
        "monitoring_eo_alert": (EO_GAP_ALERT, expected["limits"]["max_eo_gap_vs_merit"]),
    }
    mismatched = [name for name, (actual, wanted) in pairs.items() if actual != wanted]
    assert not mismatched, f"constants differ from consensus.json: {mismatched}"
    weights = reference_weights(committee_of(ctx))
    ratios = {name: {criterion: round(float(value), 4) for criterion, value in rule.items() if value != 0}
              for name, rule in weights.items()}
    declared_hours = weights["consensus"]["heures_travail"]
    derived_hours = committee_of(ctx).hours_over_r()
    assert declared_hours == DECLARED_HOURS_WEIGHT, f"declared hours weight not used: {declared_hours}"
    assert abs(declared_hours - derived_hours) <= 0.1 * derived_hours, \
        f"declared hours {declared_hours} outside +-10% of committee hours/R {derived_hours:.4f}"
    ctx["derived_weights"] = ratios
    return (f"reviewer income weights and limits equal consensus.json; declared hours {declared_hours} within 10% of "
            f"committee hours/R {derived_hours:.4f}; weights (R=1) {ratios}")


def check_consensus_guard_declared(ctx) -> str:
    record = ctx["baseline"]
    summary = record.summary()["consensus_guard"]
    names = [check["name"] for check in summary["checks"]]
    expected = ("grant rate overall", "grant rate centre", "grant rate remote", "opportunity gap vs merit",
                "opportunity overshoot vs merit", "demographic parity gap", "impact ratio", "intersectional gap",
                "mean agreement", "min agreement", "opportunity gap vs consensus rule", "hours shape, largest jump",
                "hours shape, largest drop")
    missing = [prefix for prefix in expected if not any(name.startswith(f"consensus: {prefix}") for name in names)]
    assert not missing, f"missing checks {missing}"
    alerting = [check["name"] for check in summary["checks"] if check["status"] == "ALERT"]
    assert summary["status"] != "ALERT" or record.status == "blocked", f"{summary['status']} {record.status}"
    strict_reported = sum("strict" in check["detail"] for check in summary["checks"])
    assert strict_reported == 10, f"{strict_reported} checks report strict limits"
    return f"consensus guard {summary['status']} {alerting}, {len(names)} checks recorded, status {record.status}"


def consensus_status(ctx, decisions):
    from src.harness.consensus import consensus_guard
    from src.policy import budget_share

    return consensus_guard(committee_of(ctx), ctx["history"], ctx["batch"], budget_share(ctx["history"]), decisions)


def check_consensus_rules_pass(ctx) -> str:
    from src.harness.consensus import reference_decisions
    from src.policy import REVIEWER_NAMES, budget_share

    references = reference_decisions(committee_of(ctx), ctx["batch"], budget_share(ctx["history"]))
    results = {name: consensus_status(ctx, references[name]) for name in ("consensus", *REVIEWER_NAMES)}
    statuses = {name: result.status for name, result in results.items()}
    assert statuses["consensus"] == "OK", statuses
    for name in ("merit", "legal", "regional"):
        assert statuses[name] != "ALERT", statuses
    gap = next(check for check in results["consensus"].checks
               if check.name == "consensus: opportunity gap vs merit (remote disadvantage)")
    assert gap.value < 0 and gap.status == "OK", f"signed g {gap.value}"
    alerts = {name: [check.name.removeprefix("consensus: ") for check in result.checks if check.status == "ALERT"]
              for name, result in results.items() if result.status == "ALERT"}
    return " ".join(f"{name}={status}" for name, status in statuses.items()) + f"; consensus signed g={gap.value:+.4f}; alerts {alerts}"


def check_consensus_guard_blocks(ctx) -> str:
    from src.harness.consensus import consensus_guard
    from src.policy import allocate, budget_share

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    committee = committee_of(ctx)
    penalty_kept = allocate(committee.corrected_logit(batch, 0.0), share)
    result = consensus_guard(committee, history, batch, share, penalty_kept)
    alerts = [check.name for check in result.checks if check.status == "ALERT"]
    assert result.status == "ALERT", f"direct guard {result.status}"
    assert "consensus: grant rate remote" in alerts and "consensus: demographic parity gap" in alerts, alerts
    out = make_sentinel_dir(ctx["tmp"], "consensus_block")
    record = decide_and_write(history, batch, out, nodes=injected_nodes(lambda decisions: penalty_kept.copy(), "released"))
    blocks = [action for action in record.actions if action.kind.value == "BLOCK"]
    assert record.status == "blocked" and len(blocks) == 1, f"{record.status} {action_kinds(record)}"
    named = [name for name in blocks[0].params["checks"] if name.startswith("consensus: ")]
    assert "consensus: grant rate remote" in named, blocks[0].params
    assert record.summary()["consensus_guard"]["status"] == "ALERT", record.summary()["consensus_guard"]["status"]
    assert (out / "predictions.csv").read_bytes() == SENTINEL, "predictions.csv written for blocked output"
    return f"penalty-kept committee top-k ALERT ({len(alerts)} checks), blocked with {len(named)} consensus checks named"


def check_consensus_hours_cliff(ctx) -> str:
    from src.harness.consensus import consensus_guard
    from src.policy import allocate, budget_share

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    merit = (batch["cote_r_equivalent"] - history["cote_r_equivalent"].mean()) / history["cote_r_equivalent"].std()
    cliff = allocate((merit + 0.5 * (batch["heures_travail_semaine"] > 10)).to_numpy(), share)
    result = consensus_guard(committee_of(ctx), history, batch, share, cliff)
    shape = {check.name: check.status for check in result.checks if "hours shape" in check.name}
    assert set(shape.values()) == {"ALERT"}, shape
    others = [check.name for check in result.checks if check.status == "ALERT" and "hours shape" not in check.name]
    return f"hours-above-10h rule ALERT on {len(shape)} shape checks; other alerts {others}"












def declared_record(ctx):
    if "residual_record" not in ctx:
        from src.policy import DECLARED_CONFIG

        ctx["residual_record"] = ctx["baseline_decide"](config=DECLARED_CONFIG, residual=ctx["residual"])
    return ctx["residual_record"]


def declared_pipeline(ctx):
    from src.policy import DECLARED_CONFIG, FairPipeline, budget_share

    if "declared_pipeline" not in ctx:
        ctx["declared_pipeline"] = FairPipeline(DECLARED_CONFIG, budget_share(ctx["history"]), ctx["residual"]).fit(
            ctx["history"])
    return ctx["declared_pipeline"]


def check_strong_guard(ctx) -> str:
    from src.harness import strong_guard
    from src.harness.postprocessing import publication_status
    from src.monitoring import Verdict
    from src.policy import FairPipeline, allocate, budget_share, is_remote

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    record = declared_record(ctx)
    summary = record.summary()
    assert record.status == "published" and summary["strong_guard"]["status"] == "OK", summary["strong_guard"]["status"]
    assert len(summary["strong_guard"]["checks"]) == 6 and "strong_guard" not in ctx["baseline"].summary(), "layer shape"
    pipeline = declared_pipeline(ctx)
    skewed = allocate(pipeline.model_probability(batch) - 0.5 * is_remote(batch), share)
    result = strong_guard(pipeline, history, batch, share, skewed)
    alerts = [check.name for check in result.checks if check.status == "ALERT"]
    assert result.status == "ALERT" and any("region impact" in name for name in alerts), alerts
    assert publication_status(Verdict("OK", []), [], Verdict("OK", []), result) == "blocked", "ALERT does not block"
    assert strong_guard(FairPipeline(panel_config(), share), history, batch, share, skewed).checks == [], "opt-in leaked"
    return f"declared config published with strong guard OK; remote-penalised ranking ALERT on {alerts} and blocks"


def check_reasoning_port(ctx) -> str:
    from src.policy import ReasoningSettings, deliberate, reason

    rng = np.random.default_rng(3)
    n, k = 600, 240
    base = rng.normal(size=n)
    evidence = {"cote_r": base + rng.normal(size=n), "heures_travail": base + rng.normal(size=n),
                "consensus": base + rng.normal(size=n)}
    decisions = np.zeros(n, dtype=int)
    decisions[np.argsort(-base)[:k]] = 1
    settings = ReasoningSettings(evidence=0.0, margin=0.0)
    outcome = reason(decisions, evidence, k, settings)
    assert len(outcome.moved_out) == len(outcome.moved_in) > 0 and outcome.decisions.sum() == k, "moves not paired"
    traced = deliberate(decisions, evidence, k, settings)
    assert np.array_equal(traced.decisions, decisions) and traced.counts["proposed_moves"] == len(outcome.moved_out), "traces only moved"
    assert all("not applied: traces only" in traced.traces[i] for i in outcome.moved_out), "withheld trace"
    applied = deliberate(decisions, evidence, k, ReasoningSettings(evidence=0.0, margin=0.0, apply_moves=True))
    assert applied.decisions.sum() == k and (applied.decisions != decisions).sum() == 2 * len(outcome.moved_out), "applied swap"
    class StubGate:
        def __init__(self, admit):
            self.admit = admit

        def admits(self, candidate, current):
            return self.admit(candidate)

        def settled(self, current):
            return False

    refused = deliberate(decisions, evidence, k, ReasoningSettings(evidence=0.0, margin=0.0, apply_moves=True),
                         gate=StubGate(lambda candidate: False))
    assert np.array_equal(refused.decisions, decisions) and refused.counts["moved_out"] == 0, "guard ignored"
    limit = 3
    capped = deliberate(decisions, evidence, k, ReasoningSettings(evidence=0.0, margin=0.0, apply_moves=True),
                        gate=StubGate(lambda candidate: (candidate != decisions).sum() <= 2 * limit))
    assert capped.counts["moved_out"] == limit and capped.decisions.sum() == k, f"{capped.counts}"
    for bad in ({"decisions": decisions[:-1]}, {"k": k + 1}, {"settings": ReasoningSettings(scope="x")}):
        arguments = {"decisions": decisions, "k": k, "settings": settings, **bad}
        try:
            reason(arguments["decisions"], evidence, arguments["k"], arguments["settings"])
        except ValueError:
            continue
        raise AssertionError(f"accepted {sorted(bad)}")
    return (f"{len(outcome.moved_out)} proposed swaps: traces only keeps decisions, applied moves keep k={k}, "
            f"guard refusal moves none, partial guard keeps exactly {limit}; bad inputs rejected")


def check_reasoning_record(ctx) -> str:
    record = declared_record(ctx)
    explanations = record.explanations
    counts = record.summary()["deliberation"]
    examined = explanations["reasoning_trace"] != ""
    assert int(examined.sum()) == counts["examined"] == int(explanations["validated"].sum()), "examined mismatch"
    outcomes = explanations["reasoning_outcome"].value_counts().to_dict()
    assert outcomes.get("contradicted_kept", 0) == counts["contradicted_kept"], f"{outcomes} {counts}"
    assert counts["moved_out"] == 0, "traces-only default moved applicants"
    assert "reasoning_trace" not in ctx["baseline"].explanations and "deliberation" not in ctx["baseline"].summary(), "leak"
    sample = explanations.loc[examined, "reasoning_trace"].iloc[0]
    assert "cote_r" in sample and "consensus" in sample, sample
    return f"traces for the {counts['examined']} reviewed applicants only; outcomes {outcomes}; counts {counts}"


def check_reasoning_gate(ctx) -> str:
    from src.harness.reasoning_gate import ReasoningGate, guard_checks, reasoning_gate
    from src.monitoring import EO_GAP_ALERT, Check, verdict
    from src.policy import allocate, budget_share, is_remote

    history, batch = ctx["history"], ctx["batch"]
    share = budget_share(history)
    pipeline = declared_pipeline(ctx)
    before = pipeline.jury_outcome(batch).decisions
    statuses = {check.status for check in guard_checks(pipeline, history, batch, share, EO_GAP_ALERT)(before)}
    assert "WARN" in statuses, f"no pre-existing WARN to test against: {statuses}"
    gate = reasoning_gate(pipeline, history, batch, share, EO_GAP_ALERT)(before)
    assert gate.admits(before, before), "a pre-existing WARN blocks an unchanged decision vector"
    skewed = allocate(pipeline.model_probability(batch) - 0.5 * is_remote(batch), share)
    assert not gate.admits(skewed, before), "a clearly worse vector was accepted"

    n = 40
    decisions = np.zeros(n, dtype=int)
    decisions[:18] = 1
    decisions[20:22] = 1
    favoured, harmed = np.arange(20, 30), np.arange(30, 40)

    def strong(candidate):
        return verdict([Check("strong: favoured grants", 10 - candidate[favoured].sum(), "", "ALERT"
                              if candidate[favoured].sum() < 5 else "OK", excess=5.0 - candidate[favoured].sum()),
                        Check("strong: harmed grants", float(candidate[harmed].sum()), "",
                              "ALERT" if candidate[harmed].sum() > 0 else "OK", excess=float(candidate[harmed].sum()))])

    def stub_gate():
        return ReasoningGate(lambda candidate: [], strong, decisions)

    improving = decisions.copy()
    improving[0], improving[22] = 0, 1
    new_failure = decisions.copy()
    new_failure[0], new_failure[30] = 0, 1
    worsening = decisions.copy()
    worsening[20], worsening[18] = 0, 1
    assert strong(decisions).status == "ALERT", "crafted batch has no strong-guard ALERT"
    assert stub_gate().admits(improving, decisions), "excess-reducing move rejected"
    assert not stub_gate().admits(new_failure, decisions), "move creating a new strong failure accepted"
    assert not stub_gate().admits(worsening, decisions), "excess-increasing move accepted"
    return (f"pre-existing statuses {sorted(statuses)}: unchanged vector accepted, region-penalised vector rejected; "
            f"crafted strong ALERT: excess-reducing move accepted, excess-increasing and new-failure moves rejected")




RESIDUAL_DIR = ROOT / "models/tabm_residual_ensemble_rh"
FORBIDDEN_WORDS = ("leader" + "board", "hx" + "buddy", "sub" + "mission", "batch" + "1", "pro" + "be", "son" + "de",
                   "reward" + " model", "SCO" + "RED")
FORBIDDEN_NUMBERS = tuple(re.escape(value) for value in ("94" + ".7", "95" + ".0", "95" + ".1", "95" + ".2"))


def residual_copy(ctx, name: str) -> Path:
    directory = ctx["tmp"] / name
    shutil.copytree(RESIDUAL_DIR, directory)
    return directory


def rejected(call, expected: str) -> str:
    from src.harness import InputError

    try:
        call()
    except InputError as error:
        assert expected in str(error), f"message {error}"
        return str(error)
    raise AssertionError("tampered artefact accepted")


def check_residual_artefact(ctx) -> str:
    from src.adapters import read_residual

    batch_ids = ctx["batch"]["id_candidat"]
    manifest = json.loads((RESIDUAL_DIR / "manifest.json").read_text())
    residual = read_residual(RESIDUAL_DIR, HISTORY_PATH, BATCH_PATH, batch_ids)
    assert len(residual) == 4000 and np.isfinite(residual).all(), "residual shape"
    members = manifest["members"]
    assert len(members) == 4 and all(m["oof_logloss_improvement"] > 0 for m in members), "member OOF improvements"
    assert all(m["monotonicity"]["grid_status"] == "GRIDPASS" for m in members), "member monotonicity"
    assert manifest["monotonicity"]["grid_status"] == "GRIDPASS", manifest["monotonicity"]
    assert set(manifest["source_hashes"]) == {m["cfg_id"] for m in members}, "source hashes"
    assert all(manifest["source_hashes"][m["cfg_id"]] == m["source_sha256"] for m in members), "member source hash"
    sources = list(manifest["source_hashes"].values())
    assert len(set(sources)) == 4 and all(re.fullmatch("[0-9a-f]{64}", value) for value in sources), "source hash format"
    assert manifest["aggregation"].startswith("id-aligned arithmetic mean"), manifest["aggregation"]
    tampered = residual_copy(ctx, "residual_values")
    path = tampered / "residuals.csv"
    path.write_text(path.read_text().replace(",", ",9", 1))
    rejected(lambda: read_residual(tampered, HISTORY_PATH, BATCH_PATH, batch_ids), "residuals.csv")
    wrong_data = residual_copy(ctx, "residual_manifest")
    document = json.loads((wrong_data / "manifest.json").read_text())
    document["input_hashes"]["history"] = "0" * 64
    (wrong_data / "manifest.json").write_text(json.dumps(document))
    rejected(lambda: read_residual(wrong_data, HISTORY_PATH, BATCH_PATH, batch_ids), HISTORY_PATH.name)
    other_batch = ctx["tmp"] / "other_batch.csv"
    ctx["batch"].iloc[:-1].to_csv(other_batch, index=False)
    rejected(lambda: read_residual(RESIDUAL_DIR, HISTORY_PATH, other_batch, batch_ids), "other_batch.csv")
    rejected(lambda: read_residual(ctx["tmp"] / "absent", HISTORY_PATH, BATCH_PATH, batch_ids), "incomplete")
    return (f"hashes match the manifest and both data files; tampered values, history hash, batch file and missing "
            f"directory rejected; four members, each with OOF log loss gain and GRIDPASS; mean gain "
            f"{manifest['oof_logloss_improvement_mean_of_members']:+.5f}")


def check_residual_required(ctx) -> str:
    empty = ctx["tmp"] / "no_residual"
    empty.mkdir()
    out = ctx["tmp"] / "no_residual_out"
    declared = run_cli("decide", *CLI_FLAGS, "--out-dir", str(out), "--residual-dir", str(empty))
    assert declared.returncode == 1 and "TabM residual artefact incomplete" in declared.stderr, declared.stderr[-300:]
    assert not (out / "predictions.csv").exists(), "predictions written without the artefact"
    panel = run_cli("decide", *CLI_FLAGS, *PANEL_CLI, "--out-dir", str(out), "--residual-dir", str(empty))
    assert panel.returncode == 0 and (out / "predictions.csv").exists(), panel.stderr[-300:]
    return "declared configuration fails clearly without the artefact (exit 1, no predictions); consensus panel still runs"


def check_declared_pipeline(ctx) -> str:
    from src.policy import DECLARED_CONFIG, DECLARED_INCOME_WEIGHT, DECLARED_RESIDUAL_BLEND, REVIEWER_INCOME_WEIGHTS

    low, high = min(REVIEWER_INCOME_WEIGHTS.values()), max(REVIEWER_INCOME_WEIGHTS.values())
    assert low <= DECLARED_INCOME_WEIGHT <= high, f"declared income weight {DECLARED_INCOME_WEIGHT} outside [{low}, {high}]"
    assert DECLARED_CONFIG.residual_blend == DECLARED_RESIDUAL_BLEND == 2.5, "blend knob"
    assert DECLARED_CONFIG.residual_artefact == "tabm_residual_ensemble_rh", "declared artefact"
    assert DECLARED_CONFIG.jury.audit_only and DECLARED_CONFIG.reasoning and not DECLARED_CONFIG.reasoning.apply_moves
    record = declared_record(ctx)
    batch = ctx["batch"]
    assert record.status == "published" and int(record.decisions.sum()) == 1598, f"{record.status}"
    assert np.array_equal(record.decisions, record.proposed) and not record.jury_moved_ids, "audit mode moved decisions"
    assert record.jury["applied"] is False and record.jury["triggered"] > 0, record.jury
    assert record.deliberation["moved_out"] == 0 == record.deliberation["moved_in"], record.deliberation
    pipeline = declared_pipeline(ctx)
    expected = pipeline.base_score(batch) + DECLARED_RESIDUAL_BLEND * ctx["residual"].reindex(batch["id_candidat"]).to_numpy()
    k = 1598
    top = np.zeros(len(batch), dtype=int)
    top[np.argsort(-expected, kind="stable")[:k]] = 1
    assert np.array_equal(top, record.decisions), "decisions are not the top-k of base + residual"
    reversed_batch = batch.iloc[::-1].reset_index(drop=True)
    shuffled = pipeline.decide(reversed_batch).decisions[::-1]
    assert np.array_equal(shuffled, record.decisions), "residual is not aligned by id"
    from src.pipelines import run_decision

    produced = run_decision(HISTORY_PATH, BATCH_PATH, ctx["tmp"] / "declared_out", residual_dir=RESIDUAL_DIR)["record"]
    assert any("residuals.csv" in name for name in produced.input_hashes), produced.input_hashes
    assert np.array_equal(produced.decisions, record.decisions), "pipeline run differs from the direct decide"
    return f"top-{k} of base + residual; audit mode: {record.jury['triggered']} reviewed, no swap, no move; income weight within [{low}, {high}]"


JUROR_DIR = ROOT / "models/tabm_residual_jurors"
SWEEP_DIR = Path.home() / "Downloads/equialgo_tabm_sweep_history_base"
ENSEMBLE_MEMBERS = ("m16x2x128_cap0.25_Rh", "m16x2x128_cap0.5_Rh", "m32x3x256_cap0.25_Rh", "m32x3x256_cap0.5_Rh")


def juror_copy(ctx, name: str) -> Path:
    directory = ctx["tmp"] / name
    shutil.copytree(JUROR_DIR, directory)
    return directory


def check_residual_jurors_artefact(ctx) -> str:
    from src.adapters import read_juror_residuals
    from src.policy import ACTIVE_MODEL_JURY_CONFIG, TABM_JURORS

    batch_ids = ctx["batch"]["id_candidat"]
    manifest = json.loads((JUROR_DIR / "manifest.json").read_text())
    table = read_juror_residuals(JUROR_DIR, HISTORY_PATH, BATCH_PATH, batch_ids, TABM_JURORS)
    assert table.shape == (4000, 8) and np.isfinite(table.to_numpy()).all(), table.shape
    members = manifest["members"]
    assert [m["cfg_id"] for m in members] == list(TABM_JURORS), "member order"
    assert all(m["oof_logloss_improvement"] > 0 for m in members), "member OOF improvements"
    assert all(m["monotonicity"]["grid_status"] == "GRIDPASS" for m in members), "member monotonicity"
    assert all(m["monotonicity"]["min_R_derivative_at_declared_blend"] > 0 and
               m["monotonicity"]["min_hours_derivative_at_declared_blend"] > 0 for m in members), "derivatives"
    sources = [m["source_sha256"] for m in members]
    assert len(set(sources)) == 8 and all(re.fullmatch("[0-9a-f]{64}", value) for value in sources), "source hashes"
    assert manifest["source_hashes"] == dict(zip(TABM_JURORS, sources)), "source hash map"
    ensemble = json.loads((RESIDUAL_DIR / "manifest.json").read_text())["source_hashes"]
    assert all(ensemble[name] == manifest["source_hashes"][name] for name in ENSEMBLE_MEMBERS), "shared sources differ"
    mean = table[list(ENSEMBLE_MEMBERS)].mean(axis=1)
    assert np.allclose(mean.reindex(ctx["residual"].index), ctx["residual"], atol=1e-12), "ensemble is not the mean of its jurors"
    verified = "sweep arrays absent, source hashes not re-read"
    if SWEEP_DIR.is_dir():
        for member in members:
            source = SWEEP_DIR / member["source_file"]
            assert sha256_bytes(source.read_bytes()) == member["source_sha256"], f"{member['cfg_id']}: source hash"
            arrays = np.load(source, allow_pickle=True)
            stored = pd.Series(arrays["mean_residual"], index=arrays["id_candidat"]).reindex(table.index)
            assert np.allclose(stored, table[member["cfg_id"]], atol=1e-12), f"{member['cfg_id']}: residual values"
        verified = "sweep arrays re-read: source hashes and residual values match"
    tampered = juror_copy(ctx, "juror_values")
    path = tampered / "residuals.csv"
    path.write_text(path.read_text().replace(",", ",9", 1))
    rejected(lambda: read_juror_residuals(tampered, HISTORY_PATH, BATCH_PATH, batch_ids, TABM_JURORS), "residuals.csv")
    wrong_data = juror_copy(ctx, "juror_manifest")
    document = json.loads((wrong_data / "manifest.json").read_text())
    document["input_hashes"]["history"] = "0" * 64
    (wrong_data / "manifest.json").write_text(json.dumps(document))
    rejected(lambda: read_juror_residuals(wrong_data, HISTORY_PATH, BATCH_PATH, batch_ids, TABM_JURORS), HISTORY_PATH.name)
    renamed = juror_copy(ctx, "juror_members")
    document = json.loads((renamed / "manifest.json").read_text())
    document["members"].reverse()
    (renamed / "manifest.json").write_text(json.dumps(document))
    rejected(lambda: read_juror_residuals(renamed, HISTORY_PATH, BATCH_PATH, batch_ids, TABM_JURORS), "members")
    other_batch = ctx["tmp"] / "other_juror_batch.csv"
    ctx["batch"].iloc[:-1].to_csv(other_batch, index=False)
    rejected(lambda: read_juror_residuals(JUROR_DIR, HISTORY_PATH, other_batch, batch_ids, TABM_JURORS), "other_juror_batch.csv")
    rejected(lambda: read_juror_residuals(ctx["tmp"] / "absent", HISTORY_PATH, BATCH_PATH, batch_ids, TABM_JURORS), "incomplete")
    config = ACTIVE_MODEL_JURY_CONFIG
    assert config.jury.jurors == TABM_JURORS and config.jury.quorum == 0.75 and not config.jury.audit_only, config.jury
    return (f"eight jurors with OOF gain and GRIDPASS, ensemble residual is the mean of its four jurors, {verified}; "
            f"tampered values, history hash, member order, batch file and missing directory rejected; quorum 0.75")


def check_residual_aware_jurors(ctx) -> str:
    from src.policy import (ACTIVE_MODEL_JURY_CONFIG, ACTIVE_RESIDUAL_JURY_CONFIG, DECLARED_CONFIG, DECLARED_RESIDUAL_BLEND,
                            REFERENCE_JURORS, TABM_JURORS, FairPipeline, budget_share)
    from src.adapters import read_juror_residuals

    history, batch, residual = ctx["history"], ctx["batch"], ctx["residual"]
    share = budget_share(history)
    plain = declared_pipeline(ctx).juror_scores(batch)
    aware = FairPipeline(ACTIVE_RESIDUAL_JURY_CONFIG, share, residual).fit(history).juror_scores(batch)
    shift = DECLARED_RESIDUAL_BLEND * residual.reindex(batch["id_candidat"]).to_numpy()
    for name in REFERENCE_JURORS:
        assert np.allclose(aware[name] - plain[name], shift, atol=1e-12), f"{name}: not rule + blend x residual"
    assert np.array_equal(aware["merit"], plain["merit"]), "non-reference juror changed"
    jurors = read_juror_residuals(JUROR_DIR, HISTORY_PATH, BATCH_PATH, batch["id_candidat"], TABM_JURORS)
    model = FairPipeline(ACTIVE_MODEL_JURY_CONFIG, share, residual, jurors).fit(history)
    scores = model.juror_scores(batch)
    base = model.base_score(batch)
    for name in TABM_JURORS:
        expected = base + DECLARED_RESIDUAL_BLEND * jurors[name].reindex(batch["id_candidat"]).to_numpy()
        assert np.allclose(scores[name], expected, atol=1e-12), f"{name}: not base + blend x juror residual"
    shuffled = batch.iloc[::-1].reset_index(drop=True)
    assert np.allclose(model.juror_scores(shuffled)[TABM_JURORS[0]][::-1], scores[TABM_JURORS[0]]), "juror residual not aligned by id"
    try:
        FairPipeline(ACTIVE_MODEL_JURY_CONFIG, share, residual)
    except ValueError as error:
        assert "juror residual artefact" in str(error), error
    else:
        raise AssertionError("model jury built without its artefact")
    assert DECLARED_CONFIG.reference_residual is False and DECLARED_CONFIG.juror_artefact is None, "declared config changed"
    return (f"reference jurors = rule + {DECLARED_RESIDUAL_BLEND} x residual (same alignment); "
            f"TabM jurors = base + {DECLARED_RESIDUAL_BLEND} x own residual, id-aligned; artefact required")


def check_vetted_swaps(ctx) -> str:
    from dataclasses import replace

    from src.harness.reasoning_gate import guard_checks, worse
    from src.harness.strong_guard import strong_measurer
    from src.harness.swap_gate import swap_gate
    from src.monitoring import EO_GAP_ALERT
    from src.policy import ACTIVE_SAFE_JURY_CONFIG, FairPipeline, budget_share

    history, batch, residual = ctx["history"], ctx["batch"], ctx["residual"]
    share = budget_share(history)
    open_pipeline = FairPipeline(replace(ACTIVE_SAFE_JURY_CONFIG, vetted_swaps=False), share, residual).fit(history)
    proposed = open_pipeline.jury_outcome(batch)
    pairs = list(zip(proposed.overturned_out, proposed.overturned_in))
    assert len(pairs) >= 3, f"too few swap pairs to test the gate: {len(pairs)}"

    from src.policy import vet_swaps

    seen = []
    rejected_pair = pairs[1]

    def stub(candidate):
        changed = np.flatnonzero(candidate != proposed.proposed)
        seen.append(changed)
        return not (candidate[rejected_pair[0]] == 0 and candidate[rejected_pair[1]] == 1)

    vetted = vet_swaps(proposed, stub)
    assert vetted.held_back == 1 and len(vetted.overturned_out) == len(pairs) - 1, "one pair should be held back"
    assert len(seen) == len(pairs), "every pair is tried once, in order, until none remain"
    first = seen[0]
    assert set(first) == {pairs[0][0], pairs[0][1]}, "first candidate is not the strongest pair alone"
    assert set(seen[1]) == {pairs[0][0], pairs[0][1], *rejected_pair}, "second candidate adds exactly the next pair"
    assert set(seen[2]) == {pairs[0][0], pairs[0][1], *pairs[2]}, "a rejected pair stays out of later candidates"
    assert vetted.decisions.sum() == proposed.proposed.sum(), "k not exact"
    assert vetted.decisions[rejected_pair[0]] == 1 and vetted.decisions[rejected_pair[1]] == 0, "rejected pair applied"
    nothing = vet_swaps(proposed, lambda candidate: False)
    assert np.array_equal(nothing.decisions, proposed.proposed) and nothing.held_back == len(pairs), "stop when none qualify"

    try:
        FairPipeline(ACTIVE_SAFE_JURY_CONFIG, share, residual).fit(history).jury_outcome(batch)
    except ValueError as error:
        assert "swap gate" in str(error), error
    else:
        raise AssertionError("vetted swaps ran without a gate")

    record = ctx["baseline_decide"](config=ACTIVE_SAFE_JURY_CONFIG, residual=residual)
    assert record.status == "published" and "REVERT_JURY" not in action_kinds(record), action_kinds(record)
    applied = int((record.decisions != record.proposed).sum()) // 2
    assert applied + record.jury["held_back_pairs"] == len(pairs), (applied, record.jury, len(pairs))
    assert int(record.decisions.sum()) == int(record.proposed.sum()) == 1598, "k not exact"
    checks = guard_checks(open_pipeline, history, batch, share, EO_GAP_ALERT)
    strong = strong_measurer(open_pipeline, history, batch, share)
    measure = lambda decisions: [*checks(decisions), *strong(decisions).checks]
    before = {check.name: check for check in measure(record.proposed)}
    worsened = [check.name for check in measure(record.decisions) if worse(check, before[check.name])]
    assert not worsened, f"checks worse after vetted swaps: {worsened}"
    effect = next(check for check in record.jury_guard.checks if "fairness effect" in check.name)
    assert effect.value <= 1e-9 and effect.status == "OK", (effect.value, effect.status)
    gate = swap_gate(open_pipeline, history, batch, share)(record.proposed)
    assert gate.admits(record.decisions) and gate.admits(record.proposed), "gate rejects its own accepted outcome"
    return (f"{len(pairs)} proposed pairs: stub gate holds back exactly the rejected pair, tries one pair at a time in strength order, "
            f"stops with none left; real gate applies {applied}, holds back {record.jury['held_back_pairs']}, k exact, "
            f"no monitoring/consensus/strong check worse, jury fairness effect {effect.value:+.4f}")


def check_no_derived_constants(ctx) -> str:
    pattern = re.compile("|".join([*(re.escape(word) for word in FORBIDDEN_WORDS), *FORBIDDEN_NUMBERS]), re.IGNORECASE)
    hits = []
    for folder in ("src", "kaggle", "models"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix in {".py", ".json", ".txt", ".ipynb"} and path.is_file():
                hits += [f"{path.relative_to(ROOT)}:{number}" for number, line in enumerate(path.read_text().splitlines(), 1)
                         if pattern.search(line)]
    assert not hits, f"externally derived constants or references: {hits[:5]}"
    return "no externally derived constant or reference in src, kaggle, models"


def run_check(name: str, function, *args) -> None:
    started = time.time()
    try:
        detail = function(*args)
        report(name, True, f"{detail} ({time.time() - started:.1f}s)")
    except AssertionError as error:
        report(name, False, f"{error} ({time.time() - started:.1f}s)")
    except Exception as error:
        report(name, False, f"{type(error).__name__}: {str(error)[:200]} ({time.time() - started:.1f}s)")


def run_boundary_checks(ctx) -> None:
    started = time.time()
    try:
        violations = scan_boundaries(ROOT)
    except Exception as error:
        report("boundaries", False, f"scan error {type(error).__name__}: {error}")
        return
    for label, rule in (("boundaries_deep_imports", "deep"), ("boundaries_direction", "direction"),
                        ("boundaries_third_party", "external"), ("boundaries_file_io", "io")):
        report(label, not any(item[0] == rule for item in violations), summarize_violations(violations, rule))
    run_check("boundaries_policy_lazy", check_policy_lazy)
    run_check("boundaries_proof", check_boundary_proof, ctx)
    print(f"boundaries total {time.time() - started:.1f}s", flush=True)


def build_context(tmp: Path) -> dict:
    history, batch = pd.read_csv(HISTORY_PATH), pd.read_csv(BATCH_PATH)
    hashes = input_hashes()

    def baseline_decide(**kwargs):
        from src.harness import decide

        kwargs.setdefault("config", panel_config())
        return decide(history, batch, hashes, **kwargs)

    from src.adapters import read_residual

    residual = read_residual(RESIDUAL_DIR, HISTORY_PATH, BATCH_PATH, batch["id_candidat"])
    return {"tmp": tmp, "history": history, "batch": batch, "residual": residual, "baseline_decide": baseline_decide}


def main() -> int:
    started = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="equialgo_acceptance_"))
    print(f"workdir {tmp}", flush=True)
    try:
        ctx = build_context(tmp)
    except Exception as error:
        report("setup", False, f"{type(error).__name__}: {error}")
        return 1
    run_check("baseline", check_baseline, ctx)
    run_check("budget_guard", check_budget_guard, ctx)
    run_check("duplicate_id", check_duplicate_id, ctx)
    run_check("forced_correctable_alert", check_forced_correctable_alert, ctx)
    run_check("zero_offset_block", check_zero_offset_block, ctx)
    run_check("drift_block", check_drift_block, ctx)
    run_check("categorical_drift", check_categorical_drift, ctx)
    run_check("uncomputable_metric_block", check_uncomputable_metric_block, ctx)
    run_check("alert_after_correction", check_alert_after_correction, ctx)
    run_check("replay", check_replay, ctx)
    run_check("region_blindness", check_region_blindness, ctx)
    run_check("tuner_not_live", check_tuner_not_live, ctx)
    run_check("frame_rejects", check_frame_rejects, ctx)
    run_check("csv_income_zero", check_csv_income_zero, ctx)
    run_check("region_invariance", check_region_invariance, ctx)
    run_check("schemas", check_schemas, ctx)
    run_check("categorical_drift_ok", check_categorical_drift_ok, ctx)
    run_check("monitoring_concurrency", check_monitoring_concurrency, ctx)
    run_check("graph_runtime", check_graph_runtime, ctx)
    run_check("cli_invalid_batch", check_cli_invalid_batch, ctx)
    run_check("extra_fields", check_extra_fields, ctx)
    run_boundary_checks(ctx)
    run_check("explain_identity", check_explain_identity, ctx)
    run_check("jury_grant_count", check_jury_grant_count, ctx)
    run_check("jury_swap_symmetry", check_jury_swap_symmetry, ctx)
    run_check("jury_region_invariance", check_jury_region_invariance, ctx)
    run_check("jury_determinism", check_jury_determinism, ctx)
    run_check("jury_record", check_jury_record, ctx)
    run_check("jury_offset_monotone", check_jury_offset_monotone, ctx)
    run_check("label_correction", check_label_correction, ctx)
    run_check("correction_report", check_correction_report, ctx)
    run_check("decide_skips_bootstrap", check_decide_skips_bootstrap, ctx)
    run_check("tiny_history_report", check_tiny_history_report, ctx)
    run_check("reference_consistency", check_reference_consistency, ctx)
    run_check("tune_requires_penalty", check_tune_requires_penalty, ctx)
    run_check("fit_uses_history_rate", check_fit_uses_history_rate, ctx)
    run_check("cli_record", check_cli_record, ctx)
    run_check("cli_replay", check_cli_replay, ctx)
    run_check("postprocessing_default", check_postprocessing_default, ctx)
    run_check("output_guard", check_output_guard, ctx)
    run_check("output_cases", check_output_cases, ctx)
    run_check("offset_bound", check_offset_bound, ctx)
    run_check("explain_fields", check_explain_fields, ctx)
    run_check("config_option", check_config_option, ctx)
    run_check("cli_config", check_cli_config, ctx)
    run_check("model_jury_decide", check_model_jury_decide, ctx)
    run_check("model_jury_determinism", check_model_jury_determinism, ctx)
    run_check("model_jury_region_invariance", check_model_jury_region_invariance, ctx)
    run_check("model_jurors_region_blind", check_model_jurors_region_blind, ctx)
    run_check("blend_config", check_blend_config, ctx)
    run_check("jury_guard_declared", check_jury_guard_declared, ctx)
    run_check("jury_guard_model", check_jury_guard_model, ctx)
    run_check("jury_guard_revert", check_jury_guard_revert, ctx)
    run_check("jury_guard_leaky_juror", check_jury_guard_leaky_juror, ctx)
    run_check("consensus_constants", check_consensus_constants, ctx)
    run_check("consensus_guard_declared", check_consensus_guard_declared, ctx)
    run_check("consensus_rules_pass", check_consensus_rules_pass, ctx)
    run_check("consensus_guard_blocks", check_consensus_guard_blocks, ctx)
    run_check("consensus_hours_cliff", check_consensus_hours_cliff, ctx)
    run_check("consensus_target_labels", check_consensus_target_labels, ctx)
    run_check("consensus_panel_decide", check_consensus_panel_decide, ctx)
    run_check("monitoring_signed_gap", check_monitoring_signed_gap, ctx)
    run_check("jury_guard_orientation_free", check_jury_guard_orientation_free, ctx)
    run_check("revert_refits_offset", check_revert_refits_offset, ctx)
    run_check("correctable_consensus_alert", check_correctable_consensus_alert, ctx)
    run_check("offset_feasibility_first", check_offset_feasibility_first, ctx)
    run_check("strong_guard", check_strong_guard, ctx)
    run_check("reasoning_port", check_reasoning_port, ctx)
    run_check("reasoning_record", check_reasoning_record, ctx)
    run_check("reasoning_gate", check_reasoning_gate, ctx)
    run_check("residual_artefact", check_residual_artefact, ctx)
    run_check("residual_required", check_residual_required, ctx)
    run_check("declared_pipeline", check_declared_pipeline, ctx)
    run_check("residual_jurors_artefact", check_residual_jurors_artefact, ctx)
    run_check("residual_aware_jurors", check_residual_aware_jurors, ctx)
    run_check("vetted_swaps", check_vetted_swaps, ctx)
    run_check("no_derived_constants", check_no_derived_constants, ctx)
    failures = [name for status, name, _ in RESULTS if status == "FAIL"]
    print(f"{len(RESULTS) - len(failures)}/{len(RESULTS)} passed in {time.time() - started:.0f}s; failed: {failures}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
