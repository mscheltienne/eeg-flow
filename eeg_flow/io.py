from pathlib import Path

import mne
import pyxdf
import numpy as np
from scipy.interpolate import UnivariateSpline
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT

from .utils._checks import _check_type


# ------------------------------- Load streams -------------------------------
def load_xdf(fname):
    """Load XDF file."""
    assert Path(fname).exists()
    streams, _ = pyxdf.load_xdf(fname)
    return streams


def find_streams(streams, stream_name):
    """
    Find the stream including 'stream_name' in the name attribute.
    """
    return [(k, stream) for k, stream in enumerate(streams)
            if stream_name in stream['info']['name'][0]]


def stream_names(streams):
    """
    Return the list of stream names.
    """
    return [stream['info']['name'][0] for stream in streams]


def _get_stream_timestamps(stream):
    """
    Retrieve the LSL timestamp array.
    """
    return stream['time_stamps']


def _get_stream_data(stream):
    """
    Retrieve the time series.
    """
    return stream['time_series']


# ------------------------------- EEG stream --------------------------------
def create_raw(stream):
    """
    Create raw array from EEG stream.
    """
    ch_names, ch_types, units = _get_eeg_ch_info(stream)
    sfreq = _get_eeg_sfreq(stream)
    data = _get_stream_data(stream).T

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, first_samp=0)

    # fix channel names/types
    mne.rename_channels(
        raw.info, {'AUX7': 'ECG',
                   'AUX8': 'hEOG',
                   'EOG': 'vEOG',
                   'AUX4': 'EDA'})
    # AUX5 to be defined
    raw.set_channel_types(mapping={'ECG': 'ecg', 'vEOG': 'eog', 'hEOG': 'eog'})

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
            mne.rename_channels(raw.info, {key: value})
        except Exception:
            pass

    # scaling
    def uVolt2Volt(timearr):
        """Converts from uV to Volts."""
        return timearr * 1e-6
    raw.apply_function(uVolt2Volt, picks=['eeg', 'eog', 'ecg', 'misc'],
                       channel_wise=True)

    # add reference and set montage
    raw.add_reference_channels(ref_channels='CPz')
    raw.set_montage('standard_1020')  # only after adding ref channel

    return raw


def _get_eeg_ch_info(stream):
    """
    Extract the info for each eeg channels (label, type and unit)
    """
    ch_names, ch_types, units = [], [], []

    # get channels labels, types and units
    for ch in stream["info"]["desc"][0]['channels'][0]['channel']:
        ch_type = ch['type'][0].lower()
        if ch_type not in _DATA_CH_TYPES_ORDER_DEFAULT:
            # to be changed to a dict if to many entries exist.
            ch_type = 'stim' if ch_type == 'markers' else ch_type
            ch_type = 'misc' if ch_type == 'aux' else ch_type

        ch_names.append(ch["label"][0])
        ch_types.append(ch_type)
        units.append(ch['unit'][0])

    return ch_names, ch_types, units


def _get_eeg_sfreq(stream):
    """
    Retrieve the nominal sampling rate from the stream.
    """
    return int(stream['info']['nominal_srate'][0])


# ------------------------------- MousePosition ------------------------------
def add_mouse_position(raw, eeg_stream, mouse_pos_stream, k=1):
    """
    Add the mouse position stream as 2 misc channels to the raw instance.
    """
    _check_type(raw, (mne.io.BaseRaw, ), item_name='raw')
    _check_type(k, ('int', ), item_name='k')

    eeg_timestamps = _get_stream_timestamps(eeg_stream)
    timestamps = _get_stream_timestamps(mouse_pos_stream)
    data = _get_stream_data(mouse_pos_stream)

    ch_names = [
        elt['label'][0] for elt in
        mouse_pos_stream['info']['desc'][0]['channels'][0]['channel']]
    assert ch_names == ['MouseX', 'MouseY']  # sanity-check

    # interpolate spline on mouse position
    spl = {ch: UnivariateSpline(timestamps, data.T[i, :], k=k)
           for i, ch in enumerate(ch_names)}

    # find tmin/tmax compared to raw
    tmin_idx, tmax_idx = np.searchsorted(eeg_timestamps,
                                         (timestamps[0], timestamps[-1]))
    xs = np.linspace(timestamps[0], timestamps[-1],
                     tmax_idx - tmin_idx)

    # create array
    mouse_pos_raw_array = np.zeros(shape=(len(ch_names), len(raw.times)))
    for i, ch in enumerate(ch_names):
        mouse_pos_raw_array[i, tmin_idx:tmax_idx] = spl[ch](xs)

    # add to raw
    info = mne.create_info(['mouseX', 'mouseY'], sfreq=raw.info['sfreq'],
                           ch_types='misc')
    mouse_raw = mne.io.RawArray(mouse_pos_raw_array, info)
    raw.add_channels([mouse_raw], force_update_info=True)


# -------------------------------- GameEvents --------------------------------
def add_game_events(raw, eeg_stream, game_events_stream, k=1):
    """
    Add the game events as misc channels to the raw instance.
    """
    _check_type(raw, (mne.io.BaseRaw, ), item_name='raw')
    _check_type(k, ('int', ), item_name='k')

    eeg_timestamps = _get_stream_timestamps(eeg_stream)
    timestamps = _get_stream_timestamps(game_events_stream)
    data = _get_stream_data(game_events_stream)

    ch_names = [
        elt['label'][0] for elt in
        game_events_stream['info']['desc'][0]['channels'][0]['channel']]
    assert ch_names == ['Health', 'Death', 'Primary_Assault_Ammo',
                        'Secondary_Assault_Ammo', 'Shield_Gun_Ammo', 'Ammo',
                        'Pick_Health_Pack', 'Pick_Assault_Ammo']

    # interpolate splines
    spl = {ch: UnivariateSpline(timestamps, data.T[i, :], k=k)
           for i, ch in enumerate(ch_names)}

    # find tmin/tmax compared to raw
    tmin_idx, tmax_idx = np.searchsorted(eeg_timestamps,
                                         (timestamps[0], timestamps[-1]))
    xs = np.linspace(timestamps[0], timestamps[-1],
                     tmax_idx - tmin_idx)

    # create array
    game_event_raw_array = np.zeros(shape=(len(ch_names), len(raw.times)))
    for i, ch in enumerate(ch_names):
        game_event_raw_array[i, tmin_idx:tmax_idx] = spl[ch](xs)

    # add to raw
    info = mne.create_info(ch_names, sfreq=raw.info['sfreq'], ch_types='misc')
    game_event_raw = mne.io.RawArray(game_event_raw_array, info)
    raw.add_channels([game_event_raw], force_update_info=True)


# ------------------------------- MouseButtons -------------------------------
def add_mouse_buttons(raw, eeg_stream, mouse_buttons_stream):
    """
    Add the mouse buttons press/release to the raw instance as annotations.
    """
    _check_type(raw, (mne.io.BaseRaw, ), item_name='raw')

    eeg_timestamps = _get_stream_timestamps(eeg_stream)
    timestamps = _get_stream_timestamps(mouse_buttons_stream)
    data = _get_stream_data(mouse_buttons_stream)

    # convert
    data = np.array(data)[:, 0]
    assert data.shape == timestamps.shape  # sanity-check

    # find the unique set of buttons
    unique_buttons = set(elt.split()[0] for elt in set(data))
    assert len(unique_buttons) == 4  # sanity-check

    # create annotations-like in LSL time
    onset_lsl = list()
    duration = list()
    description = list()

    for button in unique_buttons:
        for k, event in enumerate(data):
            if button not in event:
                continue

            # pressed defines onset and description
            if 'pressed' in event:
                onset_lsl.append(timestamps[k])
                description.append(button)
            # released defines durations for right/left click
            if 'released' in event:
                duration.append(timestamps[k] - onset_lsl[-1])
            # durations is set to 0 for wheel motions
            if 'MouseWheel' in event:
                duration.append(0)

    # convert onset to time relative to raw instance
    insert_idx = np.searchsorted(eeg_timestamps, onset_lsl)
    onset = raw.times[insert_idx]

    # create annotations
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)


# --------------------------------- Keyboard ---------------------------------
def add_keyboard_buttons(raw, eeg_stream, keyboard_stream):
    """
    Add the keyboard buttons press/release to the raw instance as annotations.
    """
    _check_type(raw, (mne.io.BaseRaw, ), item_name='raw')

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
            if 'pressed' in event:
                onset_lsl.append(timestamps[k])
                description.append(button)
            # released defines durations
            if 'released' in event:
                duration.append(timestamps[k] - onset_lsl[-1])

    # convert onset to time relative to raw instance
    insert_idx = np.searchsorted(eeg_timestamps, onset_lsl)
    onset = raw.times[insert_idx]

    # create annotations
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)
