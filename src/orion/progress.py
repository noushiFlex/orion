"""Progress reporting without UI coupling: core code opens tasks, the caller decides how (or whether) to show them.

    with progress.task("Backtest", total=8) as bar:
        bar.advance(0, label="2223 · gbm")  # update the label only
        bar.advance()                       # one step done

The default reporter does nothing, so library code and tests stay silent.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class Task:
    def advance(self, n: int = 1, label: str | None = None) -> None:
        pass


class Reporter:
    def start(self, label: str, total: int | None) -> Task:
        return Task()

    def finish(self, task: Task) -> None:
        pass


_current: ContextVar[Reporter] = ContextVar("orion_reporter", default=Reporter())


@contextmanager
def using(reporter: Reporter) -> Iterator[None]:
    token = _current.set(reporter)
    try:
        yield
    finally:
        _current.reset(token)


@contextmanager
def task(label: str, total: int | None = None) -> Iterator[Task]:
    reporter = _current.get()
    t = reporter.start(label, total)
    try:
        yield t
    finally:
        reporter.finish(t)
