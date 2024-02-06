# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from configparser import ConfigParser
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils._checks import check_type, ensure_path

if TYPE_CHECKING:
    from typing import Union


def load_triggers(
    fname: Union[str, Path] = files("eeg_flow.config") / "triggers.ini",
) -> dict[str, int]:
    """Load triggers from triggers.ini into a TriggerDef instance.

    Parameters
    ----------
    fname : str | Path
        Path to the configuration file. Default to ``'eeg_flow/config/triggers.ini'``.

    Returns
    -------
    triggers : dict
        Trigger definitiopn containing: standard, target, novel.
    """
    fname = ensure_path(fname, must_exist=True)
    config = ConfigParser(inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    config.read(str(fname))

    triggers = dict()
    for name, value in config.items("events"):
        triggers[name] = int(value)

    # verification
    keys = (
        "standard",
        "target",
        "novel",
    )
    for key in keys:
        if key not in triggers:
            raise ValueError(f"Key '{key}' is missing from trigger definition file.")

    return triggers


def create_config(
    xdf_folder: Union[str, Path],
    derivatives_folder: Union[str, Path],
    username: str,
) -> None:
    """Create the package configuration.

    This function can be used to overwrite an existing configuration.

    Parameters
    ----------
    xdf_folder : path-like
        Path to the folder containing the recorded raw data.
    derivatives_folder : path-like
        Path to the folder where all the derivatives will be saved.
    username : str
        Name of the experimenter.

    Notes
    -----
    The xdf_folder and the derivatives_folder should be on a network share and should be
    the same for all experimenters.
    """
    xdf_folder = ensure_path(xdf_folder, must_exist=True)
    derivatives_folder = ensure_path(derivatives_folder, must_exist=True)
    check_type(username, (str,), "username")
    assert 0 < len(username)

    config = ConfigParser(inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    config.add_section("Folders")
    config.set("Folders", "xdf", str(xdf_folder))
    config.set("Folders", "derivatives", str(derivatives_folder))
    config.add_section("Experimenter")
    config.set("Experimenter", "username", username)

    with open(Path.home() / ".eeg_flow", "w") as file:
        config.write(file)


def load_config() -> tuple[Path, Path, str]:
    """Load the package configuration.

    Returns
    -------
    xdf_folder : path-like
        Path to the folder containing the recorded raw data.
    derivatives_folder : path-like
        Path to the folder where all the derivatives will be saved.
    experimenter : str
        Name of the experimenter.
    """
    fname = Path.home() / ".eeg_flow"
    if not fname.exists():
        raise RuntimeError(
            "The configuration file with the paths to the XDF and derivatives folder "
            "does not exists. Call eeg_flow.config.create_config() to create the file "
            "'~/.eeg_flow'."
        )

    config = ConfigParser(inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    config.read(str(fname))

    xdf_folder = config["Folders"]["xdf"]
    xdf_folder = ensure_path(xdf_folder, must_exist=True)
    derivatives_folder = config["Folders"]["derivatives"]
    derivatives_folder = ensure_path(derivatives_folder, must_exist=True)
    experimenter = config["Experimenter"]["username"]

    return xdf_folder, derivatives_folder, experimenter
