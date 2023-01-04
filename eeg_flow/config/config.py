from pathlib import Path
from typing import Union

from bsl.triggers import TriggerDef


def load_triggers(
    fname: Union[str, Path] = Path(__file__).parent / "triggers.ini"
) -> TriggerDef:
    """Load triggers from triggers.ini into a TriggerDef instance.

    Parameters
    ----------
    fname : str | Path
        Path to the configuration file.
        Default to 'eeg_flow/config/triggers.ini'.

    Returns
    -------
    tdef : TriggerDef
        Trigger definitiopn containing: standard, target, novel.
    """
    tdef = TriggerDef(fname)

    keys = (
        "standard",
        "target",
        "novel",
    )
    for key in keys:
        if not hasattr(tdef, key):
            raise ValueError(
                f"Key '{key}' is missing from trigger definition."
            )

    return tdef
