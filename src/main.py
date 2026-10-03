from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.table import Table

from src.common.logging import RunContext, configure_logging
from src.preprocessing.validation import REGIONS, DatasetReport, DataValidationError, validate_datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    check.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data/donnees_demandes.csv",
                       metavar="CSV", help="historical CSV (default: supplied project data)")
    check.add_argument("--evaluation-file", type=Path, default=PROJECT_ROOT / "data/candidats_evaluation.csv",
                       metavar="CSV", help="evaluation CSV (default: supplied project data)")
    check.add_argument("--expected-train-rows", type=positive_int, default=10_000, metavar="N",
                       help="expected historical row count (default: 10000)")
    check.add_argument("--expected-evaluation-rows", type=positive_int, default=4_000, metavar="N",
                       help="expected evaluation row count (default: 4000)")
    check.add_argument("--json", action="store_true", help="write the validation report as JSON to stdout")
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
        reports = validate_datasets(args.train_file.resolve(), args.evaluation_file.resolve(), run=run,
                                    train_rows=args.expected_train_rows,
                                    evaluation_rows=args.expected_evaluation_rows)
        if args.json:
            print(json.dumps({"valid": True, "historical": reports[0].to_dict(),
                              "evaluation": reports[1].to_dict()}, indent=2))
        elif not quiet:
            show_summary(reports, run)
        if run.log_path is not None:
            run.logger.info("Log: %s", run.log_path)
        return 0
    except DataValidationError as exc:
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
