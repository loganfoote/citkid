"""
Exclusive use of a CRS board by one script, for timestreams.

Streaming needs the board's whole network bandwidth, so while a script
streams (``procedures/04_stream.py`` holds the lock for as long as its
window is open), no other script may use the board. The lock is a file on
the DAQ computer, one per board (``<lock dir>/crs<serial>.lock``, holding
the owner's process id, a description, and the start time):

- ``exclusive`` takes it. It fails if another script holds it, or if
  another script is using the board right now (sweeping or streaming:
  each such call leaves an "in use" marker while it runs).
- Every other use of the board fails immediately while another process
  holds it: ``CRS.attach``, ``configure_system``, sweeps, tone writes, and
  streams call ``check_free`` (or ``in_use``, which also leaves the
  marker).
- A lock or marker left by a process that no longer runs (e.g. after a
  crash) is ignored and removed.

The lock directory is ``~/.citkid/locks``, or ``CITKID_LOCK_DIR`` if set.
It only protects scripts on this computer that use citkid's CRS class.
"""
import contextlib
import json
import os
import time
import psutil

class BoardBusyError(RuntimeError):
    """
    Another script has (or is using) the board.
    """

################################################################################
#################################### Files #####################################
################################################################################
def lock_dir():
    """
    Return the lock directory (created if needed).

    Returns:
    path (str): ``CITKID_LOCK_DIR``, or ``~/.citkid/locks``.
    """
    path = os.environ.get('CITKID_LOCK_DIR') or os.path.join(
        os.path.expanduser('~'), '.citkid', 'locks')
    os.makedirs(path, exist_ok = True)
    return path

def _lock_path(serial):
    """
    Return a board's lock file path.

    Parameters:
    serial (int): board serial number.

    Returns:
    path (str): the lock file.
    """
    return os.path.join(lock_dir(), f'crs{int(serial)}.lock')

def _marker_path(serial, pid):
    """
    Return the "in use" marker path of one process.

    Parameters:
    serial (int): board serial number.
    pid (int): process id.

    Returns:
    path (str): the marker file.
    """
    return os.path.join(lock_dir(), f'crs{int(serial)}.inuse.{int(pid)}')

def _read(path):
    """
    Read a lock or marker file.

    Parameters:
    path (str): the file.

    Returns:
    info (dict or None): its contents (``pid``, ``owner``, ``since``), or
        None if it is missing or unreadable.
    """
    try:
        with open(path, encoding = 'utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None

def _alive(pid):
    """
    Check whether a process runs.

    Parameters:
    pid (int): process id.

    Returns:
    alive (bool): True if it runs.
    """
    try:
        return psutil.pid_exists(int(pid))
    except (TypeError, ValueError):
        return False

def _remove(path):
    """
    Remove a file if it exists.

    Parameters:
    path (str): the file.
    """
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)

################################################################################
##################################### State ####################################
################################################################################
def holder(serial):
    """
    Return who holds a board's exclusive lock, removing a stale lock.

    Parameters:
    serial (int): board serial number.

    Returns:
    info (dict or None): ``pid``, ``owner``, ``since`` of the live holder,
        or None if the board is not locked.
    """
    path = _lock_path(serial)
    info = _read(path)
    if info is None:
        if os.path.exists(path):
            _remove(path)
        return None
    if not _alive(info.get('pid')):
        _remove(path)
        return None
    return info

def users(serial):
    """
    Return the other processes using a board right now ("in use"
    markers), removing stale markers.

    Parameters:
    serial (int): board serial number.

    Returns:
    infos (list of dict): ``pid``, ``owner``, ``since`` of each.
    """
    prefix = f'crs{int(serial)}.inuse.'
    out = []
    for name in os.listdir(lock_dir()):
        if not name.startswith(prefix):
            continue
        path = os.path.join(lock_dir(), name)
        info = _read(path)
        if info is None or not _alive(info.get('pid')):
            _remove(path)
        elif int(info['pid']) != os.getpid():
            out.append(info)
    return out

def _describe(info):
    """
    Describe a lock holder or user.

    Parameters:
    info (dict): ``pid``, ``owner``, ``since``.

    Returns:
    text (str): e.g. '04_stream.py (pid 1234) since 2026-11-10 15:00:00'.
    """
    return (f"{info.get('owner', 'a script')} (pid {info.get('pid')}) "
            f"since {info.get('since', '?')}")

def check_free(serial):
    """
    Fail if another process holds a board's exclusive lock.

    Parameters:
    serial (int): board serial number.

    Raises:
    BoardBusyError: if another process holds it.
    """
    info = holder(serial)
    if info is not None and int(info['pid']) != os.getpid():
        raise BoardBusyError(
            f'CRS {serial} is in exclusive use by {_describe(info)} '
            '(streaming); try again when it ends')

################################################################################
#################################### Locking ###################################
################################################################################
# Nested "in use" calls of this process, per board
_in_use_depth = {}

@contextlib.contextmanager
def in_use(serial, owner = None):
    """
    Use a board (one sweep or stream): fail if another process holds its
    exclusive lock, and leave an "in use" marker while inside.

    Parameters:
    serial (int): board serial number.
    owner (str or None): description saved in the marker. None (default):
        the script's name.

    Yields:
    None

    Raises:
    BoardBusyError: if another process holds the exclusive lock.
    """
    check_free(serial)
    key = int(serial)
    path = _marker_path(serial, os.getpid())
    if not _in_use_depth.get(key):
        with open(path, 'w', encoding = 'utf-8') as fh:
            json.dump(dict(pid = os.getpid(), owner = owner or _script(),
                           since = time.strftime('%Y-%m-%d %H:%M:%S')), fh)
    _in_use_depth[key] = _in_use_depth.get(key, 0) + 1
    try:
        yield
    finally:
        _in_use_depth[key] -= 1
        if not _in_use_depth[key]:
            _remove(path)

def _script():
    """
    Return the running script's name.

    Returns:
    name (str): e.g. '03_tune.py'.
    """
    import sys
    return os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] \
        else 'python'

@contextlib.contextmanager
def exclusive(serial, owner = None):
    """
    Hold a board's exclusive lock: no other script may use the board until
    it is released (at the end of the ``with`` block, also on errors).

    Parameters:
    serial (int): board serial number.
    owner (str or None): description saved in the lock (e.g. the script
        and the tuning). None (default): the script's name.

    Yields:
    info (dict): the lock's contents.

    Raises:
    BoardBusyError: if another process holds the lock or is using the
        board right now.
    """
    others = users(serial)
    if others:
        raise BoardBusyError(
            f'CRS {serial} is in use by ' + '; '.join(
                _describe(i) for i in others) +
            ': stop it before taking the board for timestreams')
    path = _lock_path(serial)
    info = dict(pid = os.getpid(), owner = owner or _script(),
                since = time.strftime('%Y-%m-%d %H:%M:%S'))
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            current = holder(serial)
            if current is not None:
                raise BoardBusyError(
                    f'CRS {serial} is in exclusive use by '
                    f'{_describe(current)}') from None
            continue
        with os.fdopen(fd, 'w', encoding = 'utf-8') as fh:
            json.dump(info, fh)
        break
    else:
        raise BoardBusyError(f'could not take the lock {path}')
    try:
        yield info
    finally:
        current = _read(path)
        if current is not None and int(current.get('pid', -1)) == \
                os.getpid():
            _remove(path)
