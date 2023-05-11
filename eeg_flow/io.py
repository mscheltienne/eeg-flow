# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pyxdf
from mne import Annotations, create_info, rename_channels
from mne.io import BaseRaw, RawArray
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline

from .utils._checks import check_type, ensure_path
from .utils._docs import fill_doc

if TYPE_CHECKING:
    from pathlib import Path
    from typing import List, Tuple, Union


# ------------------------------------ Load streams ------------------------------------
@fill_doc
def load_xdf(fname: Union[str, Path]) -> List[dict]:
    """Load XDF file.

    Parameters
    ----------
    fname : str | Path
        Path to the existing .xdf dataset to load.

    Returns
    -------
    %(streams)s
    """
    fname = ensure_path(fname, must_exist=True)
    streams, _ = pyxdf.load_xdf(fname)
    return streams


@fill_doc
def find_streams(
    streams: List[dict], stream_name: str
) -> List[Tuple[int, dict]]:
    """Find the stream including 'stream_name' in the name attribute.

    Parameters
    ----------
    %(streams)s
    stream_name : str
        Substring that has to be present in the name attribute.

    Returns
    -------
    list of tuples : (k: int, stream: dict)
        k is the idx of stream in streams.
        stream is the stream that contains stream_name in its name.
    """
    return [
        (k, stream)
        for k, stream in enumerate(streams)
        if stream_name in stream["info"]["name"][0]
    ]


@fill_doc
def stream_names(streams: List[dict]):
    """Return the list of stream names.

    Parameters
    ----------
    %(streams)s
    """
    return [stream["info"]["name"][0] for stream in streams]


def _get_stream_timestamps(stream: dict):
    """Retrieve the LSL timestamp array."""
    return stream["time_stamps"]


def _get_stream_data(stream: dict):
    """Retrieve the time series."""
    return stream["time_series"]


# ------------------------------------- EEG stream -------------------------------------
@fill_doc
def create_raw(eeg_stream: dict) -> BaseRaw:
    """Create raw from EEG stream.

    Parameters
    ----------
    %(eeg_stream)s

    Returns
    -------
    %(raw)s
    """
    ch_names, ch_types = _get_eeg_ch_info(eeg_stream)
    sfreq = _get_eeg_sfreq(eeg_stream)
    data = _get_stream_data(eeg_stream).T

    info = create_info(ch_names, sfreq, ch_types)
    raw = RawArray(data, info, first_samp=0)

    mapping = {
        "FP1": "Fp1",
        "FPZ": "Fpz",
        "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "POZ": "POz",
        "FCZ": "FCz",
        "OZ": "Oz",
        "FPz": "Fpz",
    }
    for key, value in mapping.items():
        try:
            rename_channels(raw.info, {key: value})
        except Exception:
            pass

    # scaling
    def uVolt2Volt(timearr: NDArray[float]) -> NDArray[float]:
        """Convert from uV to Volts."""
        return timearr * 1e-6

    raw.apply_function(
        uVolt2Volt, picks=["eeg", "eog", "ecg", "misc"], channel_wise=True
    )

    return raw


def _get_eeg_ch_info(stream: dict) -> Tuple[List[str], List[str]]:
    """Extract the info for each eeg channels (label, type and unit)."""
    ch_names, ch_types = [], []

    # get channels labels, types and units
    for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
        ch_type = ch["type"][0].lower()
        if ch_type not in _DATA_CH_TYPES_ORDER_DEFAULT:
            # to be changed to a dict if to many entries exist.
            ch_type = "stim" if ch_type == "markers" else ch_type
            ch_type = "misc" if ch_type == "aux" else ch_type

        ch_names.append(ch["label"][0])
        ch_types.append(ch_type)

    return ch_names, ch_types


def _get_eeg_sfreq(stream: dict) -> float:
    """Retrieve the nominal sampling rate from the stream."""
    return float(Decimal(stream["info"]["nominal_srate"][0]))


# ------------------------------------ MousePosition -----------------------------------
@fill_doc
def add_mouse_position(
    raw: BaseRaw, eeg_stream: dict, mouse_pos_stream: dict, *, k: int = 1
) -> None:
    """Add the mouse position stream as 2 misc channels to the raw instance.

    Parameters
    ----------
    %(raw)s
    %(eeg_stream)s
    mouse_pos_stream : dict
        Stream containing the mouse position data.
    k : int
        Degree of the smoothing spline. Must be 1 <= k <= 5.

    Notes
    -----
    Operates in-place.
    """
    check_type(raw, (BaseRaw,), item_name="raw")
    check_type(k, ("int",), item_name="k")
    _add_misc_channel(raw, eeg_stream, mouse_pos_stream, k)


# ------------------------------------- GameEvents -------------------------------------
@fill_doc
def add_game_events(
    raw: BaseRaw, eeg_stream: dict, game_events_stream: dict, *, k: int = 1
) -> None:
    """Add the game events as misc channels to the raw instance.

    Parameters
    ----------
    %(raw)s
    %(eeg_stream)s
    game_events_stream : dict
        Stream containing the game event data.
    k : int
        Degree of the smoothing spline. Must be 1 <= k <= 5.

    Notes
    -----
    Operates in-place.
    """
    check_type(raw, (BaseRaw,), item_name="raw")
    check_type(k, ("int",), item_name="k")
    _add_misc_channel(raw, eeg_stream, game_events_stream, k)


# ----------------------------- Misc channel interpolated ------------------------------
def _add_misc_channel(
    raw: BaseRaw, eeg_stream: dict, stream: dict, k: int = 1
) -> None:
    """Add data from stream to the raw as a misc channel.

    The data from stream is interpolated on the timestamps of eeg_stream.

    Notes
    -----
    Operates in-place.
    """
    eeg_timestamps = _get_stream_timestamps(eeg_stream)
    timestamps = _get_stream_timestamps(stream)
    data = _get_stream_data(stream)

    ch_names = [
        elt["label"][0]
        for elt in stream["info"]["desc"][0]["channels"][0]["channel"]
    ]

    # interpolate spline on mouse position
    spl = {
        ch: UnivariateSpline(timestamps, data.T[i, :], k=k)
        for i, ch in enumerate(ch_names)
    }

    # find tmin/tmax compared to raw
    tmin_idx, tmax_idx = np.searchsorted(
        eeg_timestamps, (timestamps[0], timestamps[-1])
    )
    xs = np.linspace(timestamps[0], timestamps[-1], tmax_idx - tmin_idx)

    # create array
    raw_array = np.zeros(shape=(len(ch_names), len(raw.times)))
    for i, ch in enumerate(ch_names):
        raw_array[i, tmin_idx:tmax_idx] = spl[ch](xs)

    # add channel
    info = create_info(ch_names, sfreq=raw.info["sfreq"], ch_types="misc")
    misc_raw = RawArray(raw_array, info)
    raw.add_channels([misc_raw], force_update_info=True)


# ------------------------------------ MouseButtons ------------------------------------
@fill_doc
def add_mouse_buttons(
    raw: BaseRaw, eeg_stream: dict, mouse_buttons_stream: dict
) -> None:
    """Add the mouse buttons press/release to the raw as annotations.

    Parameters
    ----------
    %(raw)s
    %(eeg_stream)s
    mouse_buttons_stream : dict
        Loaded stream containing the mouse button data.

    Notes
    -----
    Operates in-place.
    """
    check_type(raw, (BaseRaw,), item_name="raw")

    eeg_timestamps = _get_stream_timestamps(eeg_stream)
    timestamps = _get_stream_timestamps(mouse_buttons_stream)
    data = _get_stream_data(mouse_buttons_stream)

    # convert
    data = np.array(data)[:, 0]
    assert data.shape == timestamps.shape  # sanity-check

    # find the unique set of buttons
    unique_buttons = set(elt.split()[0] for elt in set(data))

    # create annotations-like in LSL time
    onset_lsl = list()
    duration = list()
    description = list()

    for button in unique_buttons:
        for k, event in enumerate(data):
            if button not in event:
                continue

            # pressed defines onset and description
            if "pressed" in event:
                onset_lsl.append(timestamps[k])
                description.append(button)
            # released defines durations for right/left click
            if "released" in event:
                duration.append(timestamps[k] - onset_lsl[-1])
            # durations is set to 0 for wheel motions
            if "MouseWheel" in event:
                duration.append(0)

    # convert onset to time relative to raw instance
    insert_idx = np.searchsorted(eeg_timestamps, onset_lsl)
    onset = raw.times[insert_idx]

    # create annotations
    annotations = Annotations(onset, duration, description)
    raw.set_annotations(raw.annotations + annotations)


# -------------------------------------- Keyboard --------------------------------------
@fill_doc
def add_keyboard_buttons(
    raw: BaseRaw, eeg_stream: dict, keyboard_stream: dict
) -> None:
    """Add the keyboard buttons press/release to the raw as annotations.

    Parameters
    ----------
    %(raw)s
    %(eeg_stream)s
    keyboard_stream : dict
        Loaded stream containing the keyboard button data.

    Notes
    -----
    Operates in-place.
    """
    check_type(raw, (BaseRaw,), item_name="raw")

    eeg_timestamps = _get_stream_timestamps(eeg_stream)
    timestamps = _get_stream_timestamps(keyboard_stream)
    data = _get_stream_data(keyboard_stream)

    # convert
    data = np.array(data)[:, 0]
    assert data.shape == timestamps.shape  # sanity-check

    # find the unique set of buttons
    unique_buttons = set(elt.split()[0] for elt in set(data))

    # create annotations-like in LSL time
    onset_lsl = list()
    duration = list()
    description = list()

    for button in unique_buttons:
        for k, event in enumerate(data):
            if button not in event:
                continue

            # pressed defines onset and description
            if "pressed" in event:
                onset_lsl.append(timestamps[k])
                description.append(button)
            # released defines durations
            if "released" in event:
                duration.append(timestamps[k] - onset_lsl[-1])

    # convert onset to time relative to raw instance
    insert_idx = np.searchsorted(eeg_timestamps, onset_lsl)
    onset = raw.times[insert_idx]

    # create annotations
    annotations = Annotations(onset, duration, description)
    raw.set_annotations(raw.annotations + annotations)
