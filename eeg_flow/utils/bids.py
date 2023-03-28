from pathlib import Path
from typing import Union

from ._checks import check_type, check_value, ensure_path


def get_fname(participant: int, group: int, task: str, run: int) -> str:
    """Get the file name stem from the participant, group, task and run.

    Parameters
    ----------
    participant : int
        ID of the participant.
    group : int
        ID of the group, 1 to 8.
    task : "oddball" | "UT"
        Task name.
    run : int
        ID of the run, 1 or 2.

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
    return f"sub-P{participant}-G{group}_task-{task}_run-{run}"


def get_folder(root: Union[str, Path], participant: int, group: int) -> Path:
    """Get the folder from the participant and group.

    Parameters
    ----------
    root : path-like
        Path to the BIDS-like root folder.
    participant : int
        ID of the participant.
    group : int
        ID of the group, 1 to 8.

    Returns
    -------
    folder : Path
        Path to the folder.
    """
    root = ensure_path(root, must_exist=True)
    check_type(participant, ("int",), "participant")
    check_type(group, ("int",), "group")
    return root / f"sub-{participant}-G{group}"
