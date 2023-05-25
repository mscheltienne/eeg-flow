from typing import List

from filelock import SoftFileLock

from ._checks import check_type, ensure_path
from ._docs import fill_doc


@fill_doc
def lock_files(*args, timeout: float = 10) -> List[SoftFileLock]:
    """Lock the files provided as positional arguments.

    Parameters
    ----------
    %(timeout)s

    Returns
    -------
    locks : List of SoftFileLock
        The list of locks. When deleted, a lock is automatically released.
    """
    check_type(timeout, ("numeric",), "timeout")
    files = [ensure_path(arg, must_exist=False) for arg in args]
    locks = list()
    failed = list()
    for file in files:
        lock = SoftFileLock(file.with_suffix(file.suffix + ".lock"))
        try:
            lock.acquire(timeout=timeout)
        except Exception:
            failed.append(str(file))
            continue
        locks.append(lock)
    if len(failed) != 0:
        for lock in locks:
            lock.release()
        raise RuntimeError(f"Could not lock all files. {failed} are already in-use.")
    return locks
