# ########################################
# Modified on Sun May 07 05:04:00 2023
# @anguyen

from mne import read_annotations
from mne.io import read_info, read_raw_fif
from mne.preprocessing import read_ica

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files


@fill_doc
def prep_ica_compare(
    participant: str,
    group: str,
    task: str,
    run: int,
    ica_nb: int,
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
    %(ica_nb)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    # prepare folders
    _, DERIVATIVES_FOLDER_ROOT, EXPERIMENTER = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # lock the output derivative files
    # create locks

    derivatives = [
        DERIVATIVES_SUBFOLDER
        / (f"{FNAME_STEM}_step5_reviewed-1st-ica.fif"),
        DERIVATIVES_SUBFOLDER
        / (f"{FNAME_STEM}_step5_reviewed-2nd-ica.fif"),
        ]

    locks = lock_files(*derivatives, timeout=timeout)
    raw, raw_ica_fit = _prep_ica_compare(participant, group, task, run, ica_nb, overwrite)
    return DERIVATIVES_SUBFOLDER, FNAME_STEM, raw, raw_ica_fit, locks


@fill_doc
def _prep_ica_compare(
    participant: str,
    group: str,
    task: str,
    run: int,
    ica_nb: int,
    overwrite: bool = False,
) -> None:
    """Convert the XDF recording to a raw FIFF file.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(ica_nb)s
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

    if ica_nb == 1:
        # Filter to final BP (1, 40) Hz
        raw_ica_fit = raw.copy()
        raw_ica_fit.filter(
            l_freq=1.0,
            h_freq=40.0,
            picks="eeg",
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )
    elif ica_nb == 2:
        raw.drop_channels(["M1", "M2"])
        # filter
        raw_ica_fit = raw
        raw_ica_fit.filter(
            l_freq=1.0,
            h_freq=100.0,  # Note the higher frequency
            picks="eeg",
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )
        # change the reference to a common average reference (CAR)
        raw_ica_fit.set_montage(None)
        raw_ica_fit.add_reference_channels(ref_channels="CPz")
        raw_ica_fit.set_montage("standard_1020")
        raw_ica_fit.set_eeg_reference("average", projection=False)
        # Note that the CAR is excluding the bad channels.

    return raw, raw_ica_fit


def load_ica_rev(
        DERIVATIVES_SUBFOLDER, FNAME_STEM, REVIEWER1, REVIEWER2, ica_nb
        ):
    # load ICA

    if ica_nb == 1:
        fname_ica_rev1 = DERIVATIVES_SUBFOLDER / (f"{FNAME_STEM}_step4_reviewed-1st-{REVIEWER1}-ica.fif")
        fname_ica_rev2 = DERIVATIVES_SUBFOLDER / (f"{FNAME_STEM}_step4_reviewed-1st-{REVIEWER2}-ica.fif")
    elif ica_nb == 2:
        fname_ica_rev1 = DERIVATIVES_SUBFOLDER / (f"{FNAME_STEM}_step4_reviewed-2nd-{REVIEWER1}-ica.fif")
        fname_ica_rev2 = DERIVATIVES_SUBFOLDER / (f"{FNAME_STEM}_step4_reviewed-2nd-{REVIEWER2}-ica.fif")

    ica_rev1 = read_ica(fname_ica_rev1)
    ica_rev2 = read_ica(fname_ica_rev2)

    return (ica_rev1, ica_rev2)


def compare_two_revs(ica_rev1, ica_rev2):
    # find components which have been excluded in both ICAs
    exclude_common = list(set(ica_rev1.exclude).intersection(set(ica_rev2.exclude)))
    exclude_diff = list(
        set(
            list(set(ica_rev1.exclude) - set(ica_rev2.exclude))
            + list(set(ica_rev2.exclude) - set(ica_rev1.exclude))
        )
    )
    return exclude_common, exclude_diff
