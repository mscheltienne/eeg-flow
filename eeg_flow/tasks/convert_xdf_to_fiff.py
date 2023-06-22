import os

from mne import find_events

from .. import logger
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
from ..utils.bids import get_derivative_folder, get_fname, get_xdf_folder
from ..utils.concurrency import lock_files


@fill_doc
def convert_xdf_to_fiff(
    participant: str,
    group: str,
    task: str,
    run: int,
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
    """
    # prepare folders
    _, derivatives_folder_root, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # create derivatives preprocessed subfolder
    if derivatives_folder.exists():
        logger.info(
            "The derivatives subfolder %s already exists.", derivatives_folder.name
        )
    else:
        os.makedirs(derivatives_folder, exist_ok=False)
        logger.info("Derivatives subfolder %s created.", derivatives_folder.name)

    # lock the output derivative files
    derivatives = [
        derivatives_folder / f"{fname_stem}_step1_oddball_annot.fif",
        derivatives_folder / f"{fname_stem}_step1_raw.fif",
    ]
    if task == "UT":
        derivatives.append(
            derivatives_folder / f"{fname_stem}_step1_stream_annot.fif",
        )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        _convert_xdf_to_fiff(participant, group, task, run)
    except FileNotFoundError:
        logger.error(
            "The requested file for participant %s, group %s, task %s, run %i does "
            "not exist and will be skipped.",
            participant,
            group,
            task,
            run,
        )
    except FileExistsError:
        logger.error(
            "The destination file for participant %s, group %s, task %s, run %i "
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    except Exception as error:
        logger.error(
            "The file for participant %s, group %s, task %s, run %i could not be "
            "processed.",
            participant,
            group,
            task,
            run,
        )
        logger.exception(error)
    finally:
        for lock in locks:
            lock.release()
        del locks


@fill_doc
def _convert_xdf_to_fiff(
    participant: str,
    group: str,
    task: str,
    run: int,
) -> None:
    """Convert the XDF recording to a raw FIFF file.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    """
    # prepare folders
    xdf_folder_root, derivatives_folder_root, _ = load_config()
    xdf_foler = get_xdf_folder(xdf_folder_root, participant, group)
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # load XDF file and create raw object
    streams = load_xdf(xdf_foler / (fname_stem + "_eeg.xdf"))
    eeg_stream = find_streams(streams, "eego")[0][1]
    raw = create_raw(eeg_stream)

    # fix the AUX channel name/types
    raw.rename_channels({"AUX7": "ECG", "AUX8": "hEOG", "EOG": "vEOG", "AUX4": "EDA"})
    raw.set_channel_types(
        mapping={"ECG": "ecg", "vEOG": "eog", "hEOG": "eog", "EDA": "gsr"},
        on_unit_change="ignore",
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
    for i in range(events.shape[0] - 1, -1, -1):
        if events[i, 2] != 64:
            last_stim_onset = events[i, 0]
            break
    tmin = max(events[0, 0] / raw.info["sfreq"] - 5, 0)
    tmax = min(last_stim_onset / raw.info["sfreq"] + 5, raw.times[-1])
    raw.crop(tmin, tmax)

    # save stream-annotations to the derivatives folder
    if task == "UT":
        fname = derivatives_folder / f"{fname_stem}_step1_stream_annot.fif"
        raw.annotations.save(fname, overwrite=False)
        logger.info("Saved: %s", fname.name)
        raw.set_annotations(None)  # to speed-up browsing

    # add the annotations of the oddball paradigm
    annotations = annotations_from_events(raw, duration=0.1)
    fname = derivatives_folder / f"{fname_stem}_step1_oddball_annot.fif"
    annotations.save(fname, overwrite=False)
    logger.info("Saved: %s", fname.name)
    raw.set_annotations(annotations)

    raw.set_montage("standard_1020")
    # save the raw recording
    fname = derivatives_folder / f"{fname_stem}_step1_raw.fif"
    raw.save(fname, overwrite=False)
    logger.info("Saved: %s", fname.name)
