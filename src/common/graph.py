from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
from dataclasses import dataclass
from typing import Any

_SKIPPED = object()


def _guarded(stop: threading.Event, fn: Callable[..., Any], args: list[Any]) -> Any:
    if stop.is_set():
        return _SKIPPED
    try:
        return fn(*args)
    except BaseException:
        stop.set()
        raise


@dataclass(frozen=True)
class Node:
    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...] = ()
    after: tuple[str, ...] = ()


class GraphError(ValueError):
    pass


class Graph:
    def __init__(self, nodes: Iterable[Node]) -> None:
        self._nodes = tuple(nodes)
        names = [node.name for node in self._nodes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise GraphError(f"duplicate node names: {duplicates}")
        self._by_name = {node.name: node for node in self._nodes}
        for node in self._nodes:
            unknown = [name for name in node.after if name not in self._by_name]
            if unknown:
                raise GraphError(f"node {node.name!r} runs after unknown nodes: {unknown}")
        self._deps = {
            node.name: tuple(
                dict.fromkeys(
                    name for name in (*node.inputs, *node.after) if name in self._by_name
                )
            )
            for node in self._nodes
        }
        self._order = self._sort()
        self._seeds = frozenset(
            name for node in self._nodes for name in node.inputs if name not in self._by_name
        )

    @property
    def nodes(self) -> tuple[Node, ...]:
        return self._nodes

    @property
    def seeds(self) -> frozenset[str]:
        return self._seeds

    def order(self) -> list[str]:
        return list(self._order)

    def add(self, *nodes: Node) -> Graph:
        return Graph((*self._nodes, *nodes))

    def replace(self, node: Node) -> Graph:
        if node.name not in self._by_name:
            raise GraphError(f"cannot replace unknown node {node.name!r}")
        return Graph(node if old.name == node.name else old for old in self._nodes)

    def without(self, *names: str) -> Graph:
        unknown = sorted(set(names) - set(self._by_name))
        if unknown:
            raise GraphError(f"cannot remove unknown nodes: {unknown}")
        return Graph(node for node in self._nodes if node.name not in names)

    def run(
        self,
        seeds: Mapping[str, Any],
        *,
        targets: Iterable[str] | None = None,
        workers: int = 1,
    ) -> dict[str, Any]:
        if workers < 1:
            raise GraphError(f"workers must be >= 1, got {workers}")
        wanted = list(self._by_name) if targets is None else list(targets)
        unknown = sorted(set(wanted) - set(self._by_name))
        if unknown:
            raise GraphError(f"unknown targets: {unknown}")
        clashes = sorted(set(seeds) & set(self._by_name))
        if clashes:
            raise GraphError(f"seeds share names with nodes: {clashes}")
        needed = self._closure(wanted)
        missing = sorted(
            {
                name
                for node in self._nodes
                if node.name in needed
                for name in node.inputs
                if name not in self._by_name and name not in seeds
            }
        )
        if missing:
            raise GraphError(f"missing seeds: {missing}")
        artifacts = dict(seeds)
        if workers == 1:
            for name in self._order:
                if name in needed:
                    artifacts[name] = self._call(name, artifacts)
        else:
            self._run_parallel(artifacts, needed, workers)
        return artifacts

    def _call(self, name: str, artifacts: Mapping[str, Any]) -> Any:
        node = self._by_name[name]
        return node.fn(*[artifacts[key] for key in node.inputs])

    def _closure(self, targets: Iterable[str]) -> set[str]:
        needed: set[str] = set()
        stack = list(targets)
        while stack:
            name = stack.pop()
            if name not in needed:
                needed.add(name)
                stack.extend(self._deps[name])
        return needed

    def _sort(self) -> list[str]:
        done: set[str] = set()
        remaining = [node.name for node in self._nodes]
        order: list[str] = []
        while remaining:
            ready = next(
                (name for name in remaining if all(dep in done for dep in self._deps[name])),
                None,
            )
            if ready is None:
                raise GraphError(f"cycle: {' -> '.join(self._find_cycle(done, remaining[0]))}")
            order.append(ready)
            done.add(ready)
            remaining.remove(ready)
        return order

    def _find_cycle(self, done: set[str], start: str) -> list[str]:
        path = [start]
        current = next(dep for dep in self._deps[start] if dep not in done)
        while current not in path:
            path.append(current)
            current = next(dep for dep in self._deps[current] if dep not in done)
        return [*path[path.index(current) :], current]

    def _run_parallel(self, artifacts: dict[str, Any], needed: set[str], workers: int) -> None:
        pending = [node.name for node in self._nodes if node.name in needed]
        finished: set[str] = set()
        running: dict[Future[Any], str] = {}
        failure: BaseException | None = None
        stop = threading.Event()
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            while running or (failure is None and pending):
                if failure is None:
                    ready = [
                        name for name in pending if all(dep in finished for dep in self._deps[name])
                    ]
                    for name in ready:
                        pending.remove(name)
                        node = self._by_name[name]
                        args = [artifacts[key] for key in node.inputs]
                        running[pool.submit(_guarded, stop, node.fn, args)] = name
                if not running:
                    break
                completed, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in completed:
                    name = running.pop(future)
                    if future.cancelled():
                        continue
                    error = future.exception()
                    if error is None:
                        result = future.result()
                        if result is not _SKIPPED:
                            artifacts[name] = result
                            finished.add(name)
                    elif failure is None:
                        failure = error
                        stop.set()
                if failure is not None:
                    stop.set()
                    for future in running:
                        future.cancel()
            pool.shutdown(wait=True)
        except BaseException:
            stop.set()
            for future in running:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        if failure is not None:
            raise failure
