"""
Worker processes for the per-tone analyses (photon detection, HEE
tails, resonator pages): one pool per process, started fresh (spawn, so
it works alike on every platform) and kept until the process ends.
"""
import atexit
import concurrent.futures
import multiprocessing
import os
from tqdm.auto import tqdm

_POOL = None
_POOL_N = 0

def resolve_workers(n_workers):
    """
    Return the number of worker processes to use.

    Parameters:
    n_workers (int or None): the setting (``N_WORKERS``). None: up to 4,
        by the cores.

    Returns:
    n (int): at least 1 (1: no pool, this process).
    """
    if n_workers is None:
        return max(1, min(os.cpu_count() or 1, 4))
    return max(1, int(n_workers))

def worker_pool(n_workers):
    """
    Return the process pool, started once and kept until the process
    ends; a different size replaces it.

    Parameters:
    n_workers (int): processes.

    Returns:
    pool (ProcessPoolExecutor): the pool.
    """
    global _POOL, _POOL_N
    if _POOL is None or _POOL_N != n_workers:
        if _POOL is not None:
            _POOL.shutdown(wait = False)
        _POOL = concurrent.futures.ProcessPoolExecutor(
            max_workers = n_workers,
            mp_context = multiprocessing.get_context('spawn'))
        _POOL_N = n_workers
        atexit.register(_POOL.shutdown, wait = False)
    return _POOL

def map_in_pool(func, items, n_workers = 1, progress = False, desc = '',
                total = None):
    """
    Run a function on items, in worker processes or here.

    Parameters:
    func (callable): a module-level function (picklable) of the item's
        arguments.
    items (iterable): (key, args) pairs, ``args`` a tuple; read one at a
        time, so a generator keeps the memory bounded: at most twice
        ``n_workers`` items are in flight.
    n_workers (int): processes; 1 (default) runs here, as does a single
        item.
    progress (bool): If True, show a progress bar. Default False.
    desc (str): the bar's label. Default ''.
    total (int or None): the bar's length. Default None: unknown.

    Returns:
    out (dict): key: ``func`` output.
    """
    items = iter(items)
    out = {}
    bar = tqdm(total = total, disable = not progress, desc = desc)
    first = next(items, None)
    if first is None:
        bar.close()
        return out
    second = next(items, None)
    pool = worker_pool(n_workers) if n_workers > 1 and second is not None \
        else None
    pending = {}

    def take(done):
        """
        Keep finished workers' results.
        """
        for fut in done:
            out[pending.pop(fut)] = fut.result()
            bar.update(1)

    def rest():
        """
        Yield the items, the two read first included.
        """
        yield first
        if second is not None:
            yield second
            yield from items

    for key, args in rest():
        if pool is None:
            out[key] = func(*args)
            bar.update(1)
            continue
        if len(pending) >= 2 * n_workers:
            done, _ = concurrent.futures.wait(
                pending, return_when = concurrent.futures.FIRST_COMPLETED)
            take(done)
        pending[pool.submit(func, *args)] = key
    if pending:
        take(concurrent.futures.wait(pending)[0])
    bar.close()
    return out
