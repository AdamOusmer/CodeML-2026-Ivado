from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.table import Table

from src.common.logging import RunContext, configure_logging
from src.harness import InputError
from src.policy import REGIONS
from src.preprocessing import DatasetReport, DataValidationError, validate_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = PROJECT_ROOT / "data/donnees_demandes.csv"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data/candidats_evaluation.csv"
EXIT_ALERT = 3
STATUS_STYLE = {"OK": "green", "WARN": "yellow", "ALERT": "bold red"}


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    verbosity = common.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                           help="show debug messages and exception details")
    verbosity.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                           help="show only errors (JSON output remains available)")
    common.add_argument("--plain", action="store_true", default=argparse.SUPPRESS,
                        help="use plain text with no color or animation")
    common.add_argument("--no-progress", action="store_true", default=argparse.SUPPRESS,
                        help="hide progress bars and spinners")
    common.add_argument("--log-dir", type=Path, default=argparse.SUPPRESS, metavar="DIR",
                        help="write a timestamped DEBUG log here (default: project/logs)")
    common.add_argument("--no-log-file", action="store_true", default=argparse.SUPPRESS,
                        help="disable file logging")

    parser = argparse.ArgumentParser(description="ÉquiAlgo data and experiment tools.", parents=[common])
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    check = commands.add_parser("check-data", parents=[common],
                                help="validate both CSVs and report dataset counts",
                                description="Check schema, values, row counts, IDs, and train/evaluation separation.")
    check.add_argument("--train-file", type=Path, default=DEFAULT_HISTORY,
                       metavar="CSV", help="historical CSV (default: supplied project data)")
    check.add_argument("--evaluation-file", type=Path, default=DEFAULT_CANDIDATES,
                       metavar="CSV", help="evaluation CSV (default: supplied project data)")
    check.add_argument("--expected-train-rows", type=positive_int, default=10_000, metavar="N",
                       help="expected historical row count (default: 10000)")
    check.add_argument("--expected-evaluation-rows", type=positive_int, default=4_000, metavar="N",
                       help="expected evaluation row count (default: 4000)")
    check.add_argument("--json", action="store_true", help="write the validation report as JSON to stdout")

    monitor = commands.add_parser("monitor", parents=[common],
                                  help="audit a batch of decisions for fairness and drift",
                                  description=f"Grade budget, parity, opportunity, intersectional, proxy and feature-drift "
                                              f"checks as OK, WARN or ALERT. Exits {EXIT_ALERT} when any check alerts.")
    monitor.add_argument("--history", type=Path, default=DEFAULT_HISTORY, metavar="CSV",
                         help="labeled historical CSV used as the baseline (default: supplied project data)")
    monitor.add_argument("--batch", type=Path, default=DEFAULT_CANDIDATES, metavar="CSV",
                         help="applications that were scored (default: supplied evaluation data)")
    monitor.add_argument("--decisions", type=Path, default=PROJECT_ROOT / "predictions.csv", metavar="CSV",
                         help="id_candidat,decision_octroi for the batch (default: predictions.csv)")
    monitor.add_argument("--reviewed", type=Path, metavar="CSV",
                         help="optional id_candidat,merite labels from a blind human review")
    monitor.add_argument("--json", action="store_true", help="write the monitoring report as JSON to stdout")

    decide = commands.add_parser("decide", parents=[common],
                                 help="take automated decisions for a batch through the harness",
                                 description=f"Fit the declared policy, decide, audit, correct once if allowed, or block. "
                                             f"Writes predictions.csv only when published. Exits {EXIT_ALERT} when blocked.")
    decide.add_argument("--history", type=Path, default=DEFAULT_HISTORY, metavar="CSV",
                        help="labeled historical CSV (default: supplied project data)")
    decide.add_argument("--batch", type=Path, default=DEFAULT_CANDIDATES, metavar="CSV",
                        help="applications to decide (default: supplied evaluation data)")
    decide.add_argument("--out-dir", type=Path, default=PROJECT_ROOT, metavar="DIR",
                        help="where to write predictions.csv, decision_record.json, explanations.csv")
    decide.add_argument("--json", action="store_true", help="write the decision record as JSON to stdout")

    for name, splits, help_text in [("pareto", 10, "evaluate every candidate and plot the Pareto front"),
                                    ("tune", 5, "score the jury weight search space against the fairness references")]:
        experiment = commands.add_parser(name, parents=[common], help=help_text, description=help_text)
        experiment.add_argument("--history", type=Path, default=DEFAULT_HISTORY, metavar="CSV",
                                help="labeled historical CSV (default: supplied project data)")
        experiment.add_argument("--splits", type=positive_int, default=splits, metavar="N",
                                help=f"stratified 70/30 splits (default: {splits})")
        experiment.add_argument("--workers", type=positive_int, default=4, metavar="N", help="threads (default: 4)")
        experiment.add_argument("--out-dir", type=Path, default=PROJECT_ROOT, metavar="DIR", help="output directory")
    return parser


def show_summary(reports: tuple[DatasetReport, DatasetReport], run: RunContext) -> None:
    historical, evaluation = reports
    rate = historical.grants / historical.rows
    if run.console is None:
        print(f"Historical: {historical.rows:,} rows | {len(historical.ids):,} unique IDs | grant rate {rate:.1%}", file=sys.stderr)
        print(f"Evaluation: {evaluation.rows:,} rows | {len(evaluation.ids):,} unique IDs", file=sys.stderr)
        for region in REGIONS:
            print(f"  {region}: historical={historical.regions[region]:,}, evaluation={evaluation.regions[region]:,}", file=sys.stderr)
        return

    datasets = Table(title="Application data validated")
    for column, justify in [("Dataset", "left"), ("Rows", "right"), ("Unique IDs", "right"), ("Historical grant rate", "right")]:
        datasets.add_column(column, justify=justify)
    datasets.add_row("Historical", f"{historical.rows:,}", f"{len(historical.ids):,}", f"{rate:.1%}")
    datasets.add_row("Evaluation", f"{evaluation.rows:,}", f"{len(evaluation.ids):,}", "—")
    run.console.print(datasets)

    regions = Table(title="Applicants by region")
    for column, justify in [("Region", "left"), ("Historical", "right"), ("Evaluation", "right")]:
        regions.add_column(column, justify=justify)
    for region in REGIONS:
        regions.add_row(region, f"{historical.regions[region]:,}", f"{evaluation.regions[region]:,}")
    run.console.print(regions)


def run_check_data(args: argparse.Namespace, run: RunContext, quiet: bool) -> int:
    reports = validate_datasets(args.train_file.resolve(), args.evaluation_file.resolve(), run=run,
                                train_rows=args.expected_train_rows,
                                evaluation_rows=args.expected_evaluation_rows)
    if args.json:
        from src.adapters import json_text

        print(json_text({"valid": True, "historical": reports[0].to_dict(), "evaluation": reports[1].to_dict()}))
    elif not quiet:
        show_summary(reports, run)
    return 0


def load_batch(args: argparse.Namespace):
    from src.adapters import read_decisions, read_table

    history, batch = read_table(args.history), read_table(args.batch)
    decisions = read_decisions(args.decisions)
    missing = set(batch["id_candidat"]) - set(decisions.index)
    if missing:
        raise DataValidationError(f"{args.decisions.name}: no decision for {len(missing):,} batch applicants")
    reviewed = None
    if args.reviewed is not None:
        reviewed = read_table(args.reviewed).set_index("id_candidat")["merite"].reindex(batch["id_candidat"])
    return history, batch, decisions.reindex(batch["id_candidat"]).to_numpy(), reviewed


def show_checks(checks, status: str, run: RunContext) -> None:
    if run.console is None:
        for check in checks:
            print(f"{check.status:<5}  {check.name}: {check.value:.3f} ({check.threshold}) {check.detail}", file=sys.stderr)
        print(f"Overall: {status}", file=sys.stderr)
        return
    table = Table(title=f"Fairness monitoring: [{STATUS_STYLE[status]}]{status}[/]")
    for column, justify in [("Check", "left"), ("Value", "right"), ("Threshold", "left"), ("Status", "center"), ("Detail", "left")]:
        table.add_column(column, justify=justify)
    for check in checks:
        table.add_row(check.name, f"{check.value:.3f}", check.threshold,
                      f"[{STATUS_STYLE[check.status]}]{check.status}[/]", check.detail)
    run.console.print(table)


def run_monitor(args: argparse.Namespace, run: RunContext, quiet: bool) -> int:
    from src.adapters import json_text
    from src.monitoring import overall_status, run_checks

    with run.stage("Monitoring decisions"):
        history, batch, decisions, reviewed = load_batch(args)
        checks = run_checks(history, batch, decisions, reviewed)
    status = overall_status(checks)
    if args.json:
        print(json_text({"status": status, "checks": [check.to_dict() for check in checks]}))
    elif not quiet:
        show_checks(checks, status, run)
    return EXIT_ALERT if status == "ALERT" else 0


def run_decide(args: argparse.Namespace, run: RunContext, quiet: bool) -> int:
    from src.adapters import json_text, read_inputs, write_decision
    from src.harness import decide

    with run.stage("Deciding"):
        inputs = read_inputs(args.history, args.batch)
        record = decide(inputs.history, inputs.batch, inputs.hashes)
        written = write_decision(record, args.out_dir)
    if args.json:
        print(json_text(record.summary()))
    elif not quiet:
        show_checks(record.verdicts[-1].checks, record.verdicts[-1].status, run)
        for action in record.actions:
            run.logger.info("%s: %s %s", action.kind.value, action.reason, action.params)
        run.logger.info("Status: %s; %s of %s granted; wrote %s", record.status, int(record.decisions.sum()),
                        len(record.decisions), ", ".join(path.name for path in written))
    return 0 if record.published else EXIT_ALERT


def run_pareto(args: argparse.Namespace, run: RunContext, quiet: bool) -> int:
    from src.adapters import read_table, save_figure, write_table
    from src.evaluation import pareto_report
    from src.policy import budget_share

    with run.stage(f"Evaluating candidates over {args.splits} splits"):
        history = read_table(args.history)
        report = pareto_report(history, budget_share(history), args.splits, args.workers)
        write_table(report.summary, args.out_dir / "resultats_pareto.csv")
        save_figure(report.figure, args.out_dir / "pareto_front.png")
    if not quiet:
        print(report.table.round(3).to_string(index=False))
    return 0


def run_tune(args: argparse.Namespace, run: RunContext, quiet: bool) -> int:
    from src.adapters import read_table, write_table
    from src.evaluation import SEARCH_SPACE, tune
    from src.policy import budget_share

    with run.stage(f"Tuning {len(SEARCH_SPACE)} configurations over {args.splits} splits"):
        history = read_table(args.history)
        table = tune(history, budget_share(history), SEARCH_SPACE, args.splits, args.workers)
        write_table(table, args.out_dir / "resultats_tuner.csv")
    if not quiet:
        print(table.round(3).to_string(index=False))
    return 0


COMMANDS = {"check-data": run_check_data, "monitor": run_monitor, "decide": run_decide,
            "pareto": run_pareto, "tune": run_tune}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    options = vars(args)
    verbose, quiet = options.get("verbose", False), options.get("quiet", False)
    if verbose and quiet:
        parser.error("--verbose and --quiet cannot be combined")
    log_dir = None if options.get("no_log_file", False) else options.get("log_dir", PROJECT_ROOT / "logs")
    try:
        run = configure_logging(log_dir=log_dir, verbose=verbose, quiet=quiet,
                                plain=options.get("plain", False),
                                no_progress=options.get("no_progress", False))
    except OSError as exc:
        print(f"ERROR  Cannot initialize logging: {exc}. Try --log-dir DIR or --no-log-file.", file=sys.stderr)
        return 1

    try:
        run.logger.debug("Command: %s; project: %s", args.command, PROJECT_ROOT)
        exit_code = COMMANDS[args.command](args, run, quiet)
        if run.log_path is not None:
            run.logger.info("Log: %s", run.log_path)
        return exit_code
    except (DataValidationError, FileNotFoundError, KeyError, InputError) as exc:
        run.logger.error("%s", exc)
        run.logger.debug("Validation error details", exc_info=True)
        return 1
    except KeyboardInterrupt:
        run.logger.error("Interrupted by user")
        run.logger.debug("Interrupt details", exc_info=True)
        return 130
    except Exception as exc:
        run.logger.error("Unexpected error: %s", exc, exc_info=True)
        if run.log_path is not None:
            run.logger.error("Details: %s", run.log_path)
        return 1
    finally:
        run.close()


if __name__ == "__main__":
    raise SystemExit(main())
