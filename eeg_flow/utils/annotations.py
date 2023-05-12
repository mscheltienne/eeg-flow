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


def merge_bad_annotations(raw: BaseRaw) -> Annotations:
    """Merge bad_ and other annotations.

    Parameters
    ----------
    raw : Raw
        Raw object with annotations.

    Returns
    -------
    annotations : Annotations
        Created annotations where the bad annotations have been removed and
        annotations overlapping with a bad annotation have been prefixed with
        bad_.
    """
    annotations = raw.annotations.copy()
    offset_ = None
    annotation2remove = list()
    onsets2edit = list()
    description2edit = list()
    for k, annotation in enumerate(annotations):
        if annotation["description"].lower().startswith("bad"):
            offset_ = annotation["onset"] + annotation["duration"]
            annotation2remove.append(k)
            continue
        if offset_ is None:
            continue
        if annotation["onset"] <= offset_:
            annotation2remove.append(k)
            onsets2edit.append(annotation["onset"])
            description2edit.append(annotation["description"])

    for idx in annotation2remove[::-1]:
        annotations.delete(idx)

    description2edit = ["bad_" + elt for elt in description2edit]
    new_annotations = Annotations(onsets2edit, 0.1, description2edit)
    annotations += new_annotations
    return annotations
