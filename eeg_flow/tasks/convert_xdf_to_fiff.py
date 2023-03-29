from mne import find_events

from ..config import load_config
from ..io import (
    add_game_events,
    add_keyboard_buttons,
    add_mouse_buttons,
    add_mouse_position,
    create_raw,
    find_streams,
    load_xdf,
)
from ..utils._docs import fill_doc
from ..utils.annotations import annotations_from_events
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files, release_files


@fill_doc
def convert_xdf_to_fiff(
    participant: int,
    group: int,
    task: str,
    run: int,
    overwrite: bool = False,
    *,
    timeout: float = 10,
) -> None:
    """Convert the XDF recording to a raw FIFF file.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    overwrite : bool
        If True, overwrites existing derivatives.
    %(timeout)s
    """
    # prepare folders
    xdf_folder, derivatives_folder, experimenter = load_config()
    xdf_folder = get_folder(xdf_folder, participant, group)
    derivatives_folder = get_folder(derivatives_folder, participant, group)
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / (fname_stem + "_stream_annot.fif"),
        derivatives_folder / (fname_stem + "_oddball_annot.fif"),
        derivatives_folder / (fname_stem + "_raw.fif"),
    )
    lock_files(*derivatives)
    try:
        _convert_xdf_to_fiff(participant, group, task, run, overwrite)
    finally:
        release_files(*derivatives)


@fill_doc
def _convert_xdf_to_fiff(
    participant: int,
    group: int,
    task: str,
    run: int,
    overwrite: bool = False,
) -> None:
    """Convert the XDF recording to a raw FIFF file.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    # prepare folders
    xdf_folder, derivatives_folder, experimenter = load_config()
    xdf_folder = get_folder(xdf_folder, participant, group)
    derivatives_folder = get_folder(derivatives_folder, participant, group)
    fname_stem = get_fname(participant, group, task, run)

    # load XDF file and create raw object
    fname = xdf_folder / (fname_stem + "_eeg.xdf")
    streams = load_xdf(fname)
    eeg_stream = find_streams(streams, "eego")[0][1]
    raw = create_raw(eeg_stream)

    # fix the AUX channel name/types
    raw.rename_channels(
        {"AUX7": "ECG", "AUX8": "hEOG", "EOG": "vEOG", "AUX4": "EDA"}
    )
    raw.set_channel_types(mapping={"ECG": "ecg", "vEOG": "eog", "hEOG": "eog"})

    # add the mouse position and the game events as channels
    mouse_pos_stream = find_streams(streams, "MousePosition")[0][1]
    add_mouse_position(raw, eeg_stream, mouse_pos_stream)
    game_events_stream = find_streams(streams, "UT_GameEvents")[0][1]
    add_game_events(raw, eeg_stream, game_events_stream)

    # add the keyboard and mouse button annotations
    keyboard_stream = find_streams(streams, "Keyboard")[0][1]
    add_keyboard_buttons(raw, eeg_stream, keyboard_stream)
    mouse_buttons_stream = find_streams(streams, "MouseButtons")[0][1]
    add_mouse_buttons(raw, eeg_stream, mouse_buttons_stream)

    # crop the recording
    events = find_events(raw, stim_channel="TRIGGER")
    tmin = max(events[0, 0] / raw.info["sfreq"] - 5, 0)
    tmax = min(events[-1, 0] / raw.info["sfreq"] + 5, raw.times[-1])
    raw.crop(tmin, tmax)

    # save stream-annotations to the derivatives folder
    fname = derivatives_folder / (fname_stem + "_stream_annot.fif")
    raw.annotations.save(fname, overwrite=overwrite)
    # x-ref: https://github.com/mne-tools/mne-qt-browser/issues/161
    raw.set_annotations(None)

    # add the annotations of the oddball paradigm
    annotations = annotations_from_events(raw, duration=0.1)
    fname = derivatives_folder / (fname_stem + "_oddball_annot.fif")
    annotations.save(fname, overwrite=overwrite)
    raw.set_annotations(annotations)

    # save the raw recording
    fname = derivatives_folder / (fname_stem + "_raw.fif")
    raw.save(fname, overwrite=overwrite)
