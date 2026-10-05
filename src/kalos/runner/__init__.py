"""The M2 Singleton runner: pulls READY experiments, runs the BO engine, and
pushes results back through a `BackendAdapter` (`docs/M2_INTEGRATION.md`)."""
from .adapter import BackendAdapter, HttpBackendAdapter, LocalStoreAdapter, get_adapter
from .singleton import RunResult, SingletonLock, run_one, run_ready

__all__ = [
    "BackendAdapter", "LocalStoreAdapter", "HttpBackendAdapter", "get_adapter",
    "RunResult", "SingletonLock", "run_one", "run_ready",
]
