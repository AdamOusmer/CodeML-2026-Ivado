import os

os.environ["OMP_NUM_THREADS"] = "2"

import ast
import hashlib
import json
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
    "main": {"adapters", "harness", "evaluation", "monitoring", "preprocessing", "policy", "common"},
    "model_corrige": {"adapters", "harness", "evaluation", "monitoring", "preprocessing", "policy", "common"},
    "adapters": {"harness", "common"},
    "harness": {"monitoring", "explain", "policy", "common"},
    "monitoring": {"policy"},
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
    return run_cli("decide", *CLI_FLAGS, "--history", str(history), "--batch", str(batch), "--out-dir", str(out_dir))


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


def decide_and_write(history, batch, directory: Path, **kwargs):
    from src.adapters import write_decision
    from src.harness import decide

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
    record = decide(history, batch, input_hashes())
    ctx["baseline"] = record
    grants = int(np.sum(record.decisions))
    assert record.status == "published", f"status {record.status}"
    assert len(record.decisions) == 4000, f"rows {len(record.decisions)}"
    assert grants == round(share * 4000), f"grants {grants} != round(share*n)"
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
    assert "duplicated id_candidat" in result.stderr, f"stderr tail {lines[-1:]}"
    assert not out.exists() or not any(out.iterdir()), f"files written: {sorted(p.name for p in out.iterdir())}"
    return f"exit=1 stderr={lines[-1][:90]!r}" if lines else "exit=1"


def check_forced_correctable_alert(ctx) -> str:
    record = ctx["baseline_decide"](eo_gap_alert=0.015)
    kinds = action_kinds(record)
    n = len(record.decisions)
    grants = int(np.sum(record.decisions))
    offset = float(record.offset)
    assert "ADJUST_OFFSET" in kinds, f"no ADJUST_OFFSET: {kinds}"
    assert 0 < abs(offset) <= 0.10 + 1e-12, f"offset {offset}"
    assert len(record.moved_ids) > 0, "moved_ids empty"
    assert grants == round(record.share * n), f"grants {grants} != round(share*n)"
    return f"ADJUST_OFFSET offset={offset:+.3f} moved={len(record.moved_ids)} status={record.status} grants={grants}"


def block_action(summary: dict) -> dict:
    blocks = [action for action in summary["actions"] if action["kind"] == "BLOCK"]
    assert len(blocks) == 1, f"{len(blocks)} BLOCK actions"
    return blocks[0]


def alert_checks(summary: dict) -> list[dict]:
    return [check for verdict in summary["verdicts"] for check in verdict["checks"] if check["status"] == "ALERT"]


def check_drift_block(ctx) -> str:
    batch = ctx["batch"].copy()
    batch["cote_r_equivalent"] = batch["cote_r_equivalent"] + 3
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
    drift = [check for check in alert_checks(summary) if check["name"] == "feature drift, max PSI"]
    assert drift and not any(check["correctable"] for check in drift), f"drift alerts {drift}"
    assert "feature drift, max PSI" in block_action(summary)["params"]["checks"], "BLOCK lacks drift check"
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
    record = decide_and_write(ctx["history"], batch, out, eo_gap_alert=-1.0)
    kinds = action_kinds(record)
    assert "ADJUST_OFFSET" in kinds and "BLOCK" in kinds, f"actions {kinds}"
    assert kinds.index("ADJUST_OFFSET") < kinds.index("BLOCK"), f"order {kinds}"
    assert kinds[-1] == "BLOCK", f"actions {kinds}"
    assert record.status == "blocked", f"status {record.status}"
    assert (out / "predictions.csv").read_bytes() == SENTINEL, "sentinel predictions.csv changed (I5)"
    return f"actions={kinds} sentinel intact"


def check_zero_offset_block(ctx) -> str:
    import src.harness.controller as controller

    original = controller.fit_offset
    controller.fit_offset = lambda *args, **kwargs: 0.0
    try:
        record = ctx["baseline_decide"](eo_gap_alert=0.015)
    finally:
        controller.fit_offset = original
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
    from src.policy import DECLARED_CONFIG, FairPipeline, budget_share

    history, batch = ctx["history"], ctx["batch"]
    pipeline = FairPipeline(DECLARED_CONFIG, budget_share(history)).fit(history)
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
    probes = [
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
    for name, package, line, expected in probes:
        found = {item[0] for item in new_violations(injected_violations(tmp, name, package, line), baseline)}
        assert expected <= found, f"{name}: scanner missed {expected - found}"
        notes.append(f"{name}=caught")
    allowed = new_violations(injected_violations(tmp, "proof_allowed", "monitoring", "from src.policy import eo_gap"), baseline)
    assert not allowed, f"allowed edge flagged: {allowed}"
    notes.append("proof_allowed=clean")
    return " ".join(notes)


def check_explain_identity(ctx) -> str:
    from src.explain import explain
    from src.policy import DECLARED_CONFIG, FairPipeline, budget_share, legitimate_features

    history, batch = ctx["history"], ctx["batch"]
    pipeline = FairPipeline(DECLARED_CONFIG, budget_share(history)).fit(history)
    contributions = pipeline.contributions(batch)
    model = pipeline.main_model_
    intercept = float(model[-1].intercept_[0])
    reconstructed = contributions.to_numpy().sum(axis=1) + intercept
    expected = model.decision_function(legitimate_features(batch))
    error = float(np.max(np.abs(reconstructed - expected)))
    assert error < 1e-9, f"max abs err {error:.3g}"
    decisions = pipeline.predict(batch)
    table = explain(pipeline, batch, decisions, pipeline.score(batch, 0.0), 0.0)
    needed = {"id_candidat", "decision", "score", "merit_vote", "model_vote", "offset", "factor_1", "factor_2", "factor_3"}
    assert needed <= set(table.columns), f"missing columns {needed - set(table.columns)}"
    return f"max abs err {error:.2e}"


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

        return decide(history, batch, hashes, **kwargs)

    return {"tmp": tmp, "history": history, "batch": batch, "baseline_decide": baseline_decide}


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
    run_boundary_checks(ctx)
    run_check("explain_identity", check_explain_identity, ctx)
    failures = [name for status, name, _ in RESULTS if status == "FAIL"]
    print(f"{len(RESULTS) - len(failures)}/{len(RESULTS)} passed in {time.time() - started:.0f}s; failed: {failures}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
