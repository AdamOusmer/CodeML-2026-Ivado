from __future__ import annotations

import copy
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")


class _ConsoleHandler(logging.Handler):
    def __init__(self, delegate: logging.Handler, verbose: bool) -> None:
        super().__init__()
        self.delegate = delegate
        self.verbose = verbose

    def emit(self, record: logging.LogRecord) -> None:
        screen_record = copy.copy(record)
        if not self.verbose:
            screen_record.exc_info = None
            screen_record.exc_text = None
        self.delegate.emit(screen_record)

    def close(self) -> None:
        self.delegate.close()
        super().close()


@dataclass
class ProgressTask:
    total: int | None
    completed: int = 0
    _display: Progress | None = None
    _task_id: int | None = None

    def advance(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Progress cannot advance by a negative amount")
        self.completed += amount
        if self._display is not None:
            self._display.update(self._task_id, completed=self.completed)


@dataclass
class RunContext:
    logger: logging.Logger
    console: Console | None
    log_path: Path | None
    progress_enabled: bool

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        self.logger.info("%s...", name)
        try:
            yield
        except BaseException:
            self.logger.debug("%s stopped after %.2fs", name, perf_counter() - started)
            raise
        self.logger.info("%s completed in %.2fs", name, perf_counter() - started)

    @contextmanager
    def progress(self, description: str, total: int | None = None) -> Iterator[ProgressTask]:
        if total is not None and total < 0:
            raise ValueError("Progress total cannot be negative")
        task = ProgressTask(total=total)
        if not self.progress_enabled:
            yield task
            return
        columns = [SpinnerColumn(), TextColumn("{task.description}", markup=False)]
        if total is not None:
            columns += [BarColumn(), TaskProgressColumn(), MofNCompleteColumn()]
        columns.append(TimeElapsedColumn())
        with Progress(*columns, console=self.console, refresh_per_second=4, transient=True) as display:
            task._display = display
            task._task_id = display.add_task(description, total=total)
            yield task

    def track(self, items: Iterable[T], description: str, total: int | None = None) -> Iterator[T]:
        if total is None and hasattr(items, "__len__"):
            total = len(items)
        with self.progress(description, total=total) as task:
            for item in items:
                yield item
                task.advance()

    def close(self) -> None:
        _remove_handlers(self.logger)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"equialgo.{name}")


def _remove_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_logging(*, log_dir: Path | None, verbose: bool = False, quiet: bool = False,
                      plain: bool = False, no_progress: bool = False) -> RunContext:
    logger = logging.getLogger("equialgo")
    _remove_handlers(logger)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = None if plain else Console(file=sys.stderr)
    if console is None:
        delegate = logging.StreamHandler(sys.stderr)
        delegate.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    else:
        delegate = RichHandler(console=console, show_time=True, log_time_format="%H:%M:%S",
                               show_path=False, markup=False, rich_tracebacks=True,
                               tracebacks_show_locals=False)
    handler = _ConsoleHandler(delegate, verbose=verbose)
    handler.setLevel(logging.ERROR if quiet else logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(handler)

    log_path = None
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"equialgo-{datetime.now():%Y%m%d-%H%M%S-%f}.log"
            file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
        except OSError:
            _remove_handlers(logger)
            raise
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s"))
        logger.addHandler(file_handler)

    progress_enabled = console is not None and console.is_terminal and not (quiet or no_progress)
    return RunContext(logger, console, log_path, progress_enabled)
