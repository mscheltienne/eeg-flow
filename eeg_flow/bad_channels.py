from typing import List

import mne
import numpy as np
import pyprep
from autoreject import Ransac
from mne.io import BaseRaw

from . import logger
from .utils._docs import fill_doc


def _prepapre_raw(raw: BaseRaw) -> BaseRaw:
    """Copy the raw instance and prepares it for RANSAC/PyPREP.

    Notes
    -----
    The preprocessing includes:
    - Filter between 1. and 45. Hz the EEG channels.
    - Notch at (50, 100, 150) Hz the EEG channels.
    - Set the montage as 'standard_1020'.
    The reference 'CPz' is not added.
    """
    raw = raw.copy()
    raw.filter(
        l_freq=1.0,
        h_freq=45.0,
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )
    raw.notch_filter(np.arange(50, 151, 50), picks="eeg")
    raw.set_montage("standard_1020")
    return raw


@fill_doc
def RANSAC_bads_suggestion(raw: BaseRaw) -> List[str]:
    """Detect bad channels with autoreject RANSAC implementation.

    Parameters
    ----------
    %(raw)s

    Returns
    -------
    %(bads)s
    """
    raw = _prepapre_raw(raw)
    epochs = mne.make_fixed_length_epochs(
        raw, duration=1.0, preload=True, reject_by_annotation=True
    )
    picks = mne.pick_types(raw.info, eeg=True)
    ransac = Ransac(verbose=False, picks=picks, n_jobs=1)
    ransac.fit(epochs)
    bads = ransac.bad_chs_
    if len(bads) == 0:
        logger.info("There are no bad channels found.")
    else:
        logger.info("Found %s bad channels: %s.", len(bads), ", ".join(bads))
    return bads


@fill_doc
def PREP_bads_suggestion(raw: BaseRaw) -> List[str]:
    """Detect bad channels with the PREP pipeline.

    The PREP pipeline includes:
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
        logger.info("There are no bad channels found.")
    else:
        logger.info("Found %s bad channels: %s.", len(bads), ", ".join(bads))
    return bads
