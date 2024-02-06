from __future__ import annotations  # c.f. PEP 563, PEP 649

from configparser import ConfigParser
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils._checks import ensure_path

if TYPE_CHECKING:
    from typing import Union


_TRIGGER_FNAME: Path = files("eeg_flow.config") / "triggers.ini"


def load_triggers(fname: Union[str, Path] = _TRIGGER_FNAME) -> dict[str, int]:
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
