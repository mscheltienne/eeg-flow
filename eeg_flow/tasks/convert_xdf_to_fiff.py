# ########################################
# Modified on Fri Apr 28 02:19:23 2023
# @anguyen


import os
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
from ..utils.concurrency import lock_files


@fill_doc
def convert_xdf_to_fiff(
    participant: str,
    group: str,
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
    """
    # prepare folders
    XDF_FOLDER_ROOT, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # create derivatives preprocessed subfolder
    os.makedirs(DERIVATIVES_SUBFOLDER, exist_ok=True)
    print("Created folder", DERIVATIVES_SUBFOLDER)

    # lock the output derivative files
    derivatives = [
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_oddball_annot.fif"),
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_raw.fif"),
    ]
    if task == "UT":
        derivatives.append(
            DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_stream_annot.fif")
            )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        _convert_xdf_to_fiff(participant, group, task, run, overwrite)
    finally:
        for lock in locks:
            lock.release()
        del locks
    return


def convert_xdf_to_fiff_star(args):
    return convert_xdf_to_fiff(*args)


@fill_doc
def _convert_xdf_to_fiff(
    participant: str,
    group: str,
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
    XDF_FOLDER_ROOT, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    XDF_FOLDER = get_folder(XDF_FOLDER_ROOT, participant, group)
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # load XDF file and create raw object
    FNAME_XDF = XDF_FOLDER / (FNAME_STEM + "_eeg.xdf")
    streams = load_xdf(FNAME_XDF)
    eeg_stream = find_streams(streams, "eego")[0][1]
    raw = create_raw(eeg_stream)

    # fix the AUX channel name/types
    raw.rename_channels(
        {"AUX7": "ECG", "AUX8": "hEOG", "EOG": "vEOG", "AUX4": "EDA"}
    )
    raw.set_channel_types(
        mapping={"ECG": "ecg", "vEOG": "eog", "hEOG": "eog", 'EDA': 'gsr'}
        )
    if task == "UT":
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
    # get the last event that is not a button response (ie that is a stim)
    for i in range(events.shape[0]-1, -1, -1):
        if events[i, 2] != 64:
            last_stim_onset = events[i, 0]
            break
    tmin = max(events[0, 0] / raw.info["sfreq"] - 5, 0)
    tmax = min(last_stim_onset / raw.info['sfreq'] + 5, raw.times[-1])
    raw.crop(tmin, tmax)

    # save stream-annotations to the derivatives folder
    if task == "UT":
        FNAME_STREAM_ANNOT = (
            DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_stream_annot.fif")
        )
        raw.annotations.save(FNAME_STREAM_ANNOT, overwrite=False)
        print("Saved: ", FNAME_STREAM_ANNOT)
        # x-ref: https://github.com/mne-tools/mne-qt-browser/issues/161
        raw.set_annotations(None)

    # add the annotations of the oddball paradigm
    annotations = annotations_from_events(raw, duration=0.1)
    FNAME_OB_ANNOT = (
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_oddball_annot.fif")
    )
    annotations.save(FNAME_OB_ANNOT, overwrite=overwrite)
    print("Saved: ", FNAME_OB_ANNOT)
    raw.set_annotations(annotations)

    raw.set_montage("standard_1020")
    # save the raw recording
    FNAME_RAW = DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_raw.fif")
    raw.save(FNAME_RAW, overwrite=False)
    print("Saved: ", FNAME_RAW)
