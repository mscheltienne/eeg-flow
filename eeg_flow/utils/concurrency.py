from filelock import SoftFileLock

from ._checks import check_type, ensure_path
from ._docs import fill_doc


@fill_doc
def lock_files(*args, timeout: float = 10) -> None:
    """Lock the files provided as positional arguments.

    Parameters
    ----------
    %(timeout)s
    """
    check_type(timeout, ("numerics",), "timeout")
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
        raise RuntimeError(
            f"Could not lock all files. {failed} are already in-use."
        )


def release_files(*args):
    """Release the lock on the existing files provided as arguments."""
    files = [ensure_path(arg, must_exist=True) for arg in args]
    for file in files:
        lock = SoftFileLock(file.with_suffix(file.suffix + ".lock"))
        lock.release()
