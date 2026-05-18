"""Storage layer — thin wrapper around LanceDB tuned per the perf guide."""

from lookback.store.lance_store import LanceStore

__all__ = ["LanceStore"]
