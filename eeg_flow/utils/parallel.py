import multiprocessing as mp
from typing import Any, Callable, List, Tuple

from tqdm import tqdm

from ._checks import check_type
from .logs import logger


def parallel(function: Callable, n_jobs: int, args: List[Any]) -> List[Any]:
    """Run function in parallel and display a TQDM progressbar.

    Parameters
    ----------
    function : callable
        The function to parallelize.
    n_jobs : int
        The number of jobs to run. The number of jobs will always be limited to the
        number of inputs in ``args`` and should usually not exceed the number of cores.
    args : list
        List of arguments to unpack in each function call. Hence an iterable of
        ``[(1,2), (3, 4)]`` results in ``[function(1,2), function(3,4)]``.

    Returns
    -------
    results : list
        List of results returned by each function call.
    """
    check_type(n_jobs, ("int",), "n_jobs")
    if mp.cpu_count() < n_jobs:
        logger.warning(
            "The number of requested jobs %i is superior to the number of CPU cores "
            "%i.",
            n_jobs,
            mp.cpu_count(),
        )
    # prepend the function to each input in args
    args = [[function] + list(arg) for arg in args]
    with mp.Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(_function_for_imap, args), total=len(args)))
    return results


def _function_for_imap(func_args: Tuple[Any]) -> Any:  # noqa: D401
    """Function passed to Pool.imap."""
    func = func_args[0]
    args = func_args[1:]
    return func(*args)
