import mne
import pyprep
import numpy as np
from autoreject import Ransac

from . import logger
from .utils._docs import fill_doc


def _prepapre_raw(raw):
    """
    Copy the raw instance and prepares it for RANSAC/PyPREP.
    Set the montage as 'standard_1020'. The reference 'CPz' is not added.
    """
    logger.info('Applying BP filter (1., 45.) Hz and notch filter '
                '(50, 100, 150) Hz on a copy of raw..')
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


@fill_doc
def RANSAC_bads_suggestion(raw):
    """
    Create fix length-epochs and apply a RANSAC algorithm to detect bad
    channels using autoreject.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(bads)s
    """
    raw = _prepapre_raw(raw)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, reject_by_annotation=True)
    picks = mne.pick_types(raw.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
    ransac.fit(epochs)
    bads = ransac.bad_chs_
    if len(bads) == 0:
        logger.info('There are no bad channels found.')
    else:
        logger.info('Found %s bad channels: %s.', len(bads), ', '.join(bads))
    return bads


@fill_doc
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
    %(raw)s

    Returns
    -------
    %(bads)s
    """
    raw = _prepapre_raw(raw)
    raw.pick_types(eeg=True)
    nc = pyprep.find_noisy_channels.NoisyChannels(raw)
    nc.find_all_bads()
    bads = nc.get_bads()
    if len(bads) == 0:
        logger.info('There are no bad channels found.')
    else:
        logger.info('Found %s bad channels: %s.', len(bads), ', '.join(bads))
    return bads
