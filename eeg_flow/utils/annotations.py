import numpy as np
from mne import Annotations, find_events
from mne.io import BaseRaw

from ..config import load_triggers


def annotations_from_events(raw: BaseRaw, duration: float = 0.1) -> Annotations:
    """Create annotations from the events on the trigger channel.

    Parameters
    ----------
    raw : Raw
        Raw object with a TRIGGER channel.
    duration : float
        Duration of the events.

    Returns
    -------
    annotations : Annotations
        Created annotations.

    Notes
    -----
    The events are loaded with eeg_flow.config.loader_triggers().
    The annotations are not set on the raw object.
    """
    events = find_events(raw, "TRIGGER")
    tdef = load_triggers()

    annotations = None
    for name, event in tdef.items():
        stim = np.where(events[:, 2] == event)[0]
        onsets = [events[start, 0] / raw.info["sfreq"] for start in stim]
        if annotations is None:
            annotations = Annotations(onsets, duration, name)
        else:
            annotations += Annotations(onsets, duration, name)
    return annotations
