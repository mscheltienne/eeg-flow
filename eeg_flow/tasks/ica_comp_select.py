# ########################################
# Modified on Sat May 06 01:50:00 2023
# @anguyen

import os

from mne import read_annotations
from mne.io import read_info, read_raw_fif
from mne.preprocessing import read_ica

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files


@fill_doc
def prep_ica_selection(
    participant: str,
    group: str,
    task: str,
    run: int,
    overwrite: bool = False,
    *,
    timeout: float = 10,
) -> None:
    """XXXXXXXXXX.

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
    _, DERIVATIVES_FOLDER_ROOT, EXPERIMENTER = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )
    DERIVATIVES_ICA = DERIVATIVES_SUBFOLDER / "plots" / "ica"

    # create derivatives ica plots subfolder
    os.makedirs(DERIVATIVES_ICA, exist_ok=True)

    # lock the output derivative files
    # create locks
    derivatives = [
        DERIVATIVES_SUBFOLDER
        / (f"{FNAME_STEM}_step4_reviewed-1st-{EXPERIMENTER}-ica.fif"),
        DERIVATIVES_SUBFOLDER
        / (f"{FNAME_STEM}_step4_reviewed-2nd-{EXPERIMENTER}-ica.fif"),
    ]

    locks = lock_files(*derivatives, timeout=timeout)
    ica1, ica2, raw_ica_fit1, raw = _prep_ica_selection(participant, group, task, run, overwrite)
    return (ica1, ica2, DERIVATIVES_SUBFOLDER, FNAME_STEM, EXPERIMENTER, raw_ica_fit1, raw, locks)


@fill_doc
def _prep_ica_selection(
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
    _, DERIVATIVES_FOLDER_ROOT, EXPERIMENTER = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # load previous steps
    # # load raw recording
    raw = read_raw_fif(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_raw.fif"),
        preload=True,
    )
    # # load following annots
    info = read_info(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step2_info.fif")
    )
    annot = read_annotations(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step2_oddball_with_bads_annot.fif")
    )

    # merge info and annots into current raw
    raw.info["bads"] = info["bads"]
    raw.set_annotations(annot)

    # load ICA
    fname_ica1 = DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_1st-ica.fif")
    fname_ica2 = DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_2nd-ica.fif")

    ica1 = read_ica(fname_ica1)
    ica2 = read_ica(fname_ica2)

    # Filter to final BP (1, 40) Hz
    raw_ica_fit1 = raw.copy()
    raw_ica_fit1.filter(
        l_freq=1.0,
        h_freq=40.0,
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )

    return (ica1, ica2, raw_ica_fit1, raw)
