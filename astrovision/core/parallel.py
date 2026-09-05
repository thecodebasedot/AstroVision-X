"""Process pools for the per-source stages.

The morphology and lens stages do the same work on every source, and that
work is a few thousand independent fits on a survey field: exactly what
the cores of a desktop PC are for. This module keeps one pool of worker
processes for the life of the interpreter, hands each stage's payloads
to it, and returns the results in order. The stages call their per-source
function the same way with or without the pool, so a parallel run is
identical to a serial one, source for source; a test checks that.

Workers are started with the "spawn" method everywhere: it is the only one
that is safe from a process that has threads (the desktop application),
and the only one Windows has. It means each worker imports the package
once, which costs a second or two on the first stage of a run and nothing
after.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
from typing import Any, Callable, Iterable, List, Optional

from .logging import get_logger

log = get_logger("core.parallel")

#: Never more than this many workers by default; beyond it memory, not
#: cores, is what runs out on a laptop.
MAX_AUTO_WORKERS = 8
#: Below this many items the pool costs more than it saves.
MIN_ITEMS_FOR_POOL = 8

#: Seconds a fresh worker gets to answer its first message before the
#: pool is given up on.
POOL_START_TIMEOUT = 120.0

_pool: Optional[Any] = None
_pool_size = 0
_disabled = False


def worker_count(requested: Optional[int]) -> int:
    """How many processes ``requested`` means: 0 is all cores but one."""
    if requested is None or int(requested) == 0:
        return max(1, min(MAX_AUTO_WORKERS, (os.cpu_count() or 2) - 1))
    return max(1, int(requested))


def _ping() -> int:
    return os.getpid()


def _get_pool(size: int):
    """The pool, started on first use and checked to be alive.

    A spawned worker re-imports the main module; where that is impossible
    (a script fed on standard input, some notebooks) the workers die on
    start and a plain ``Pool.map`` would wait forever while new ones are
    spawned and die in turn.  So the first thing asked of a new pool is a
    ping with a deadline, and a pool that cannot answer is given up on for
    the rest of the process.
    """
    global _pool, _pool_size, _disabled
    if _disabled:
        raise RuntimeError("worker processes could not be started in this session")
    if _pool is not None and _pool_size != size:
        close_pool()
    if _pool is None:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(processes=size)
        try:
            pool.apply_async(_ping).get(timeout=POOL_START_TIMEOUT)
        except Exception as exc:                             # noqa: BLE001 - the pool is unusable
            pool.terminate()
            pool.join()
            _disabled = True
            raise RuntimeError(f"worker processes did not start ({type(exc).__name__}); "
                               "running in one process") from exc
        _pool, _pool_size = pool, size
        log.info("started %d worker processes", size)
    return _pool


def close_pool() -> None:
    """Stop the workers; they are started again on the next parallel call."""
    global _pool, _pool_size
    if _pool is not None:
        try:
            _pool.terminate()
            _pool.join()
        except Exception:                                    # pragma: no cover
            pass
        _pool, _pool_size = None, 0


atexit.register(close_pool)


def map_work(function: Callable[[Any], Any], items: Iterable[Any],
             n_workers: Optional[int] = 1, min_items: int = MIN_ITEMS_FOR_POOL) -> List[Any]:
    """``[function(item) for item in items]``, over the pool when it pays.

    ``function`` must be importable by name (a module-level function) and
    the items and results picklable. Any failure of the pool itself -- not
    of the function -- falls back to running in this process, so a run
    never fails because of how it was parallelised.
    """
    payloads = list(items)
    size = worker_count(n_workers) if n_workers != 1 else 1
    if size <= 1 or len(payloads) < min_items:
        return [function(item) for item in payloads]
    chunksize = max(1, len(payloads) // (size * 4))
    try:
        return _get_pool(size).map(function, payloads, chunksize=chunksize)
    except Exception as exc:                                 # noqa: BLE001 - reported
        log.warning("parallel stage fell back to one process: %s", exc)
        close_pool()
        return [function(item) for item in payloads]


__all__ = ["map_work", "worker_count", "close_pool", "MAX_AUTO_WORKERS"]
