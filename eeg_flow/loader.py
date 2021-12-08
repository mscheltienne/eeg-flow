from pathlib import Path

import mne
import pyxdf
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT

from bad_channels import PREP_bads_suggestion


# Trigger on EEG amplifier
EVENTS = dict(pedal_release=64, son_standard=1, son_target=2, son_novel=3)


def _find_streams(streams, stream_name):
    """
    Find the stream including 'stream_name' in the name attribute.
    """
    return [(k, stream) for k, stream in enumerate(streams)
            if stream_name in stream['info']['name'][0]]


def _stream_names(streams):
    """
    Return the list of stream names.
    """
    return [stream['info']['name'][0] for stream in streams]


def create_raw(stream):
    """
    Create raw array from EEG stream.
    """
    ch_names, ch_types, units = _get_eeg_ch_info(stream)
    sfreq = _get_eeg_sfreq(stream)
    data = stream['time_series'].T

    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, first_samp=0)

    # fix channel names/types
    raw.rename_channels(
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

    # mark bad eeg channels
    bads = PREP_bads_suggestion(raw)  # operates on a copy
    raw.info['bads'] = bads

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
