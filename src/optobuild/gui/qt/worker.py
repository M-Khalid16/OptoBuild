"""Background execution of a project with progress and cooperative cancellation."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from optobuild.core.errors import OptoBuildError, SimulationCancelledError
from optobuild.engine import CancellationToken, FeedForwardExecutor, ProgressEvent, ResultCache
from optobuild.persistence.project import Project, run_project


class _RunThread(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    error = Signal(str)
    was_cancelled = Signal()

    def __init__(
        self, project: Project, executor: FeedForwardExecutor, token: CancellationToken
    ) -> None:
        super().__init__()
        self._project = project
        self._executor = executor
        self._token = token

    def run(self) -> None:  # executed in the worker thread
        def on_progress(event: ProgressEvent) -> None:
            self.progress.emit(event.fraction, event.node)

        try:
            result = run_project(
                self._project, executor=self._executor, progress=on_progress, cancel=self._token
            )
        except SimulationCancelledError:
            self.was_cancelled.emit()
        except OptoBuildError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # unexpected: report, never crash the GUI
            self.error.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done.emit(result)


class SimulationRunner(QObject):
    """Runs one project at a time in a QThread; results are cached between runs."""

    started = Signal()
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._executor = FeedForwardExecutor(ResultCache(max_entries=512))
        self._thread: _RunThread | None = None
        self._token: CancellationToken | None = None

    @property
    def running(self) -> bool:
        """True while a simulation is executing."""
        return self._thread is not None and self._thread.isRunning()

    def start(self, project: Project) -> None:
        """Run ``project`` (an independent snapshot) in the background."""
        if self.running:
            raise RuntimeError("A simulation is already running.")
        self._token = CancellationToken()
        thread = _RunThread(project, self._executor, self._token)
        thread.progress.connect(self.progress)
        thread.done.connect(self._on_done)
        thread.error.connect(self._on_error)
        thread.was_cancelled.connect(self._on_cancelled)
        self._thread = thread
        thread.start()
        self.started.emit()

    def cancel(self) -> None:
        """Request cancellation; takes effect at the next node boundary."""
        if self._token is not None:
            self._token.cancel()

    def wait(self, msecs: int = 60_000) -> bool:
        """Block until the thread finishes (used by tests and on shutdown)."""
        return True if self._thread is None else self._thread.wait(msecs)

    def _finish(self) -> None:
        if self._thread is not None:
            self._thread.wait()
        self._thread = None

    def _on_done(self, result: Any) -> None:
        self._finish()
        self.finished.emit(result)

    def _on_error(self, message: str) -> None:
        self._finish()
        self.failed.emit(message)

    def _on_cancelled(self) -> None:
        self._finish()
        self.cancelled.emit()
