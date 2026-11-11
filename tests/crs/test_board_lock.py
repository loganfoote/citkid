"""
Tests for ``citkid.crs.board_lock`` (exclusive use of a CRS board).
"""
import asyncio
import json
import os
import numpy as np
import psutil
import pytest
from citkid.crs import board_lock as bl
from citkid.crs.sim import SimulatedCRS
from citkid.vna import simulate as sim

SERIAL = 77


@pytest.fixture(autouse = True)
def lock_dir(tmp_path, monkeypatch):
    """
    Use a fresh lock directory for each test.

    Returns:
    path (pathlib.Path): the directory.
    """
    monkeypatch.setenv('CITKID_LOCK_DIR', str(tmp_path))
    return tmp_path


def other_pid():
    """
    Return the id of a live process other than this one.

    Returns:
    pid (int): the parent process's id.
    """
    return psutil.Process().ppid()


def dead_pid():
    """
    Return a process id that doesn't run.

    Returns:
    pid (int): an unused id.
    """
    pid = 999999
    while psutil.pid_exists(pid):
        pid += 1
    return pid


def write(path, pid, owner = 'other script'):
    """
    Write a lock or marker file as another process would.

    Parameters:
    path (pathlib.Path): the file.
    pid (int): its process id.
    owner (str): its description.
    """
    path.write_text(json.dumps(dict(pid = pid, owner = owner,
                                    since = '2026-11-10 12:00:00')))


def test_exclusive_and_release(lock_dir):
    assert bl.holder(SERIAL) is None
    bl.check_free(SERIAL)
    with bl.exclusive(SERIAL, owner = 'me') as info:
        assert info['pid'] == os.getpid()
        assert bl.holder(SERIAL)['owner'] == 'me'
        # This process may still use the board
        bl.check_free(SERIAL)
        with bl.in_use(SERIAL):
            pass
    assert bl.holder(SERIAL) is None
    # Released on errors too
    with pytest.raises(ValueError):
        with bl.exclusive(SERIAL):
            raise ValueError('boom')
    assert not (lock_dir / f'crs{SERIAL}.lock').exists()


def test_other_holder_blocks_everything(lock_dir):
    write(lock_dir / f'crs{SERIAL}.lock', other_pid(), '04_stream.py')
    with pytest.raises(bl.BoardBusyError, match = '04_stream.py'):
        bl.check_free(SERIAL)
    with pytest.raises(bl.BoardBusyError, match = 'exclusive use'):
        with bl.in_use(SERIAL):
            pass
    with pytest.raises(bl.BoardBusyError, match = 'exclusive use'):
        with bl.exclusive(SERIAL):
            pass
    # Other boards are free
    bl.check_free(SERIAL + 1)


def test_stale_lock_and_markers_are_removed(lock_dir):
    write(lock_dir / f'crs{SERIAL}.lock', dead_pid())
    write(lock_dir / f'crs{SERIAL}.inuse.{dead_pid()}', dead_pid())
    assert bl.holder(SERIAL) is None and bl.users(SERIAL) == []
    assert not list(lock_dir.iterdir())
    with bl.exclusive(SERIAL):
        pass


def test_board_in_use_blocks_exclusive(lock_dir):
    write(lock_dir / f'crs{SERIAL}.inuse.{other_pid()}', other_pid(),
          '03_tune.py')
    with pytest.raises(bl.BoardBusyError, match = '03_tune.py'):
        with bl.exclusive(SERIAL):
            pass
    # This process's own marker doesn't block it
    (lock_dir / f'crs{SERIAL}.inuse.{other_pid()}').unlink()
    with bl.in_use(SERIAL):
        assert (lock_dir / f'crs{SERIAL}.inuse.{os.getpid()}').exists()
        with bl.in_use(SERIAL):
            pass
        assert (lock_dir / f'crs{SERIAL}.inuse.{os.getpid()}').exists()
        with bl.exclusive(SERIAL):
            pass
    assert not (lock_dir / f'crs{SERIAL}.inuse.{os.getpid()}').exists()


def test_crs_methods_fail_while_another_script_streams(lock_dir):
    rng = np.random.default_rng(0)
    crs = SimulatedCRS(sim.draw_device(rng, n_banks = 1, n_per_bank = 5),
                       sim.draw_gain(rng), serial_number = SERIAL)
    asyncio.run(crs.attach())
    write(lock_dir / f'crs{SERIAL}.lock', other_pid(), '04_stream.py')
    freqs = np.full((2, 3), 1.0e9)
    with pytest.raises(bl.BoardBusyError, match = '04_stream.py'):
        asyncio.run(crs.attach())
    with pytest.raises(bl.BoardBusyError):
        asyncio.run(crs.sweep(freqs, np.full(2, -55.), 10,
                              ch_map = {1: [0, 1]}))
    (lock_dir / f'crs{SERIAL}.lock').unlink()
    asyncio.run(crs.set_nco({1: 1.0e9}))
    asyncio.run(crs.sweep(freqs, np.full(2, -55.), 10, ch_map = {1: [0, 1]}))
    # No marker is left after the sweep
    assert not list(lock_dir.glob(f'crs{SERIAL}.inuse.*'))
