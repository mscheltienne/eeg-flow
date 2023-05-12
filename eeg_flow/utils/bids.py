from pathlib import Path
from typing import Union

from ._checks import check_type, check_value, ensure_path
from ._docs import fill_doc


@fill_doc
def get_fname(
    participant: str,
    group: str,
    task: str,
    run: int,
) -> str:
    """Get the file name stem from the participant, group, task and run.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s

    Returns
    -------
    fname : str
        Stem of the file name, without any extension.

    Notes
    -----
    The file name stem can be used to create the derivatives file names.
    """
    check_type(participant, (str,), "participant")
    check_type(group, (str,), "group")
    check_type(task, (str,), "task")
    check_value(task, ("UT", "oddball"), "task")
    check_type(run, ("int",), "run")
    return f"sub-{participant}-{group}_task-{task}_run-{run}"


@fill_doc
def get_derivative_folder(
    root: Union[str, Path],
    participant: str,
    group: str,
    task: str = None,
    run: int = None,
) -> Path:
    """Get the folder from the participant and group.

    Parameters
    ----------
    root : path-like
        Path to the BIDS-like root folder.
    %(participant)s
    %(group)s
    %(task)s
    %(run)s

    Returns
    -------
    folder : Path
        Path to the folder.
    """
    root = ensure_path(root, must_exist=True)
    check_type(participant, (str,), "participant")
    check_type(group, (str,), "group")
    fname_stem = get_fname(participant, group, task, run)
    return root / f"sub-{participant}-{group}" / fname_stem


@fill_doc
def get_xdf_folder(
    root: Union[str, Path],
    participant: str,
    group: str,
) -> Path:
    """Get the XDF folder from the participant and group.

    Parameters
    ----------
    root : path-like
        Path to the BIDS-like root folder.
    %(participant)s
    %(group)s

    Returns
    -------
    folder : Path
        Path to the folder.
    """
    root = ensure_path(root, must_exist=True)
    check_type(participant, (str,), "participant")
    check_type(group, (str,), "group")
    return root / f"sub-{participant}-{group}"
