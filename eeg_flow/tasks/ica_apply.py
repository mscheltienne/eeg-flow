# ########################################
# Modified on Mon May 08 01:01:00 2023
# @anguyen

from mne import read_annotations
from mne.io import read_info, read_raw_fif
from mne.io.constants import FIFF
from mne.preprocessing import read_ica

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files


@fill_doc
def ica_apply_prep(
    participant: str,
    group: str,
    task: str,
    run: int,
    overwrite: bool = False,
    *,
    timeout: float = 10,
) -> None:
    """Apply ICAs.

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
    _, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # lock the output derivative files
    # create locks
    derivatives = [
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step6_preprocessed-raw.fif")
    ]

    locks = lock_files(*derivatives, timeout=timeout)
    try:
        _ica_apply_prep(participant, group, task, run, overwrite)
    finally:
        for lock in locks:
            lock.release()
        del locks
    return


@fill_doc
def ica_apply_prep_star(args):
    """Modification so that the function accepts *args instead.

    https://stackoverflow.com/a/67845088

    Parameters
    ----------
    %(args)s
    Reuse the args of convert_xdf_to_fiff
    """
    return ica_apply_prep(*args)


@fill_doc
def _ica_apply_prep(
    participant: str,
    group: str,
    task: str,
    run: int,
    overwrite: bool = False,
) -> None:
    """Apply ICAs

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

    # load previous steps
    # # load raw recording
    raw = read_raw_fif(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_raw.fif"),
        preload=True,
    )
    # # load following annots
    info = read_info(DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step2_info.fif"))
    annot = read_annotations(
        DERIVATIVES_SUBFOLDER
        / (FNAME_STEM + "_step2_oddball_with_bads_annot.fif")
    )

    # merge info and annots into current raw
    raw.info["bads"] = info["bads"]
    raw.set_annotations(annot)

    # load ICAs
    # --- TODO: load the final ICAs with reviewed labels
    FNAME_ICA1 = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step5_reviewed-1st-ica.fif"
    )
    FNAME_ICA2 = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step5_reviewed-2nd-ica.fif"
    )

    ica1 = read_ica(FNAME_ICA1)
    ica2 = read_ica(FNAME_ICA2)

    # 5.1 Apply ICA1 fit on raw_mastoids
    # %%% Filter the mastoids, apply ICA, and keep the mastoids
    # Final step in the preparation of our future reference, we need to filter
    # those channels to the desired frequencies, and apply the ICA.

    raw_mastoids = raw.copy()
    raw_mastoids.filter(
        l_freq=0.5,
        h_freq=40.0,
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )
    ica1.apply(raw_mastoids)
    raw_mastoids.pick_channels(["M1", "M2"])

    # Trick MNE in thinking that a custom-ref has been applied
    with raw_mastoids.info._unlock():
        raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

    """
    At this stage, the reference have been denoised.
    We have in 'raw_mastoids' the 2 mastoids M1 and M2 referenced to CPz.
    Now, let's clean the rest.
    """
    # %% Clean the other channels
    """
    The first step is to prepare the raw object for an ICA, and for suggestions
    from ICLabel. The steps are very similar to the previous ones.
    """
    raw.drop_channels(["M1", "M2"])

    # %% Apply ICA
    """
    At this stage, we have an ICA decomposition with labeled components.
    Now, we can apply it on the initial raw object, filtered between the final
    frequencies.
    But for this operation to be valid, it needs to be referenced as
    raw_ica_fit.
    """

    raw.filter(
        l_freq=0.5,
        h_freq=40.0,
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )

    raw.set_montage(None)
    raw.add_reference_channels(ref_channels="CPz")
    raw.set_eeg_reference("average", projection=False)
    ica2.apply(raw)

    # %% Rereferenced to mastoids
    # At this stage, we have:
    # - raw_mastoids, the mastoids cleaned and referenced to CPz
    # - raw, the other electrodes, cleaned + bads and referenced to CAR

    raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
    raw.add_channels([raw_mastoids])
    raw.set_montage("standard_1020")  # add montage for non-mastoids
    raw.set_eeg_reference(["M1", "M2"])

    # raw.interpolate_bads(reset_bads=True, mode='accurate')
    # raw.plot(title="ICA applied on raw")

    # Last visual inspection
    raw.plot(theme="light")

    # save deratives

    FNAME_FILT_RAW = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step6_preprocessed-raw.fif"
    )
    raw.save(FNAME_FILT_RAW, overwrite=False)
