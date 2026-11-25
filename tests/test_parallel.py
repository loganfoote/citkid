"""
Tests of the worker pool (citkid.parallel).
"""
import os
import pytest
from citkid import parallel


def square(x, k = 1):
    """
    Return x squared times k (a picklable job).
    """
    return x * x * k


def fail(x):
    """
    Raise for the pool to pass on.
    """
    raise ValueError(f'no {x}')


def test_resolve_workers():
    assert parallel.resolve_workers(None) == max(
        1, min(os.cpu_count() or 1, 4))
    assert parallel.resolve_workers(0) == 1 and parallel.resolve_workers(3) == 3


def test_map_in_pool():
    """
    Items run here with one worker, a single item or none, and in the
    pool otherwise, with the same results; the items are read one at a
    time; an error in a worker is raised here; the pool is kept.
    """
    items = [(i, (i, 2)) for i in range(7)]
    want = {i: 2 * i * i for i in range(7)}
    assert parallel.map_in_pool(square, items, 1) == want
    assert parallel.map_in_pool(square, iter(items), 2) == want
    assert parallel.map_in_pool(square, items[:1], 2) == {0: 0}
    assert parallel.map_in_pool(square, [], 2) == {}
    assert parallel.worker_pool(2) is parallel.worker_pool(2)
    read = []

    def lazy():
        """
        Yield items, recording how far the pool has read.
        """
        for i in range(12):
            read.append(i)
            yield i, (i,)

    out = parallel.map_in_pool(square, lazy(), 2, progress = True,
                               desc = 'squares', total = 12)
    assert out == {i: i * i for i in range(12)} and len(read) == 12
    with pytest.raises(ValueError, match = 'no 1'):
        parallel.map_in_pool(fail, [(1, (1,))] * 2, 2)
