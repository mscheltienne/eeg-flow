import mne
import pyprep
import numpy as np
from autoreject import Ransac


# Until 0.4 release, make sure to use the development version.
if '0.3.1' in pyprep.__version__:
    assert pyprep.__version__.split('0.3.1')[1] != ''
else:
    assert 4 <= int(pyprep.__version__.split('.')[1])


def _prepapre_raw(raw):
    """
    Copy the raw instance and prepares it for RANSAC/PyPREP.
    Set the montage as 'standard_1020'. The reference 'CPz' is not added.
    """
    raw = raw.copy()
    raw.filter(
        l_freq=1.,
        h_freq=45.,
        picks='eeg',
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge")
    raw.notch_filter(np.arange(50, 151, 50), picks='eeg')
    raw.set_montage('standard_1020')

    return raw


def RANSAC_bads_suggestion(raw):
    """
    Create fix length-epochs and apply a RANSAC algorithm to detect bad
    channels using autoreject.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw instance.

    Returns
    -------
    bads : list
        List of bad channels.
    """
    raw = _prepapre_raw(raw)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, reject_by_annotation=True)
    picks = mne.pick_types(raw.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
    ransac.fit(epochs)

    return ransac.bad_chs_


def PREP_bads_suggestion(raw):
    """
    Apply the PREP pipeline to detect bad channels:
        - SNR
        - Correlation
        - Deviation
        - HF Noise
        - NaN flat
        - RANSAC

    Parameters
    ----------
    raw : mne.io.Raw
        Raw instance.

    Returns
    -------
    bads : list
        List of bad channels.
    """
    raw = _prepapre_raw(raw)
    raw.pick_types(eeg=True)
    nc = pyprep.find_noisy_channels.NoisyChannels(raw)
    nc.find_all_bads()
    return nc.get_bads()
