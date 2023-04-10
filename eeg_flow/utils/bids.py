from pathlib import Path
from typing import Union

from ._checks import check_type, check_value, ensure_path
from ._docs import fill_doc


@fill_doc
def get_fname(participant: int, group: int, task: str, run: int) -> str:
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
    check_type(participant, ("int",), "participant")
    check_type(group, ("int",), "group")
    check_type(task, (str,), "task")
    check_value(task, ("UT", "oddball"), "task")
    check_type(run, ("int",), "run")
    if 0 >= participant or 100 <= participant:
        raise ValueError(
            "The participant ID should be set between 1 and 100. "
            f"{participant} is not valid."
        )
    return f"sub-P{str(participant).zfill(2)}-G{group}_task-{task}_run-{run}"




@fill_doc
def get_folder(root: Union[str, Path], participant: int, group: int) -> Path:
    """Get the folder from the participant and group.

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
    check_type(participant, ("int",), "participant")
    check_type(group, ("int",), "group")
    if 0 >= participant or 100 <= participant:
        raise ValueError(
            "The participant ID should be set between 1 and 100. "
            f"{participant} is not valid."
        )
    return root / f"sub-P{str(participant).zfill(2)}-G{group}"
