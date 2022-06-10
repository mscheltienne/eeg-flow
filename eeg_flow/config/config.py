from pathlib import Path

from bsl.triggers import TriggerDef


def load_triggers():
    """Load triggers from triggers.ini into a TriggerDef instance.

    Returns
    -------
    tdef : TriggerDef
        Trigger definitiopn containing: standard, target, novel.
    """
    directory = Path(__file__).parent
    tdef = TriggerDef(directory / "triggers.ini")

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
