# ########################################
# Modified on Sun May 07 03:41:00 2023
# @anguyen

from copy import deepcopy
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from mne import pick_types, read_annotations
from mne.io import read_info, read_raw_fif
from mne.preprocessing import ICA
from mne.viz.ica import _prepare_data_ica_properties
from mne_icalabel import label_components

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files


@fill_doc
def fit_two_icas(
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
    _, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # lock the output derivative files
    # create locks
    derivatives = (
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_1st-ica.fif"),
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_2nd-ica.fif"),
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_iclabel.xlsx"),
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        _fit_two_icas(participant, group, task, run, overwrite)
    finally:
        for lock in locks:
            lock.release()
        del locks
    return


def fit_two_icas_star(args):
    return fit_two_icas(*args)


@fill_doc
def _fit_two_icas(
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
    XDF_FOLDER_ROOT, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # load previous steps
    # load raw recording
    raw = read_raw_fif(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_raw.fif"),
        preload=True,
    )
    # load following annots
    info = read_info(DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step2_info.fif"))
    annot = read_annotations(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step2_oddball_with_bads_annot.fif")
    )

    # merge info and annots into current raw
    raw.info["bads"] = info["bads"]
    raw.set_annotations(annot)

    # ICA1 For mastoids
    raw_ica_fit1 = raw.copy()
    # Filter to final BP 40 Hz lowpass
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
    # ICA2 For EEG
    # %% Clean the other channels
    # The first step is to prepare the raw object for an ICA, & for suggestions
    # from ICLabel. The steps are very similar to the previous ones.
    raw.drop_channels(["M1", "M2"])

    # filter
    raw_ica_fit2 = raw.copy()
    raw_ica_fit2.filter(
        l_freq=1.0,
        h_freq=100.0,  # Note the higher frequency
        picks=["eeg"],
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )

    # change the reference to a common average reference (CAR)
    raw_ica_fit2.set_montage(None)
    raw_ica_fit2.add_reference_channels(ref_channels="CPz")
    raw_ica_fit2.set_montage("standard_1020")
    raw_ica_fit2.set_eeg_reference("average", projection=False)

    # %%% Fit an ICA in parallel :)

    ica = ICA(
        n_components=None,
        method="picard",
        max_iter="auto",
        fit_params=dict(ortho=False, extended=True),
        random_state=888,
    )
    filtered_sessions = [raw_ica_fit1, raw_ica_fit2]
    session_picks = [pick_types(filtered_sessions[i].info, eeg=True, exclude="bads") for i in range(2)]
    fitted_icas = Parallel(n_jobs=2)(delayed(fit_ica_on_data)(filtered_sessions, session_picks, deepcopy(ica), i) for i in range(2))
    del raw_ica_fit1
    ica1 = fitted_icas[0][0]
    ica2 = fitted_icas[1][0]

    # ICALABEL

    # %% Label components
    # Let's start by getting suggestion from the ICLabel model
    component_dict = label_components(raw_ica_fit2, ica2, method="iclabel")
    print(component_dict)
    #
    data_icalabel = {'y_pred': component_dict['y_pred_proba'],
                     'labels': component_dict["labels"]}
    df_icalabel = pd.DataFrame.from_dict(data_icalabel)
    fname_icalabel = DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_iclabel.xlsx")
    df_icalabel.to_excel(fname_icalabel)
    # let's remove eye-blink and heart beat
    labels = component_dict["labels"]
    exclude = [
        k for k, name in enumerate(labels) if name in ("eye blink", "heart beat")
    ]
    # let's remove other non-brain components that occur often
    _, _, _, data = _prepare_data_ica_properties(
        raw_ica_fit2,
        ica2,
        reject_by_annotation=True,
        reject="auto",
    )

    ica_data = np.swapaxes(data, 0, 1)
    var = np.var(ica_data, axis=2)  # (n_components, n_epochs)
    var = np.var(var.T / np.linalg.norm(var, axis=1), axis=0)
    # linear fit to determine the variance thresholds
    z = np.polyfit(range(0, ica2.n_components_, 1), var, 1)
    threshold = [z[0] * x + z[1] for x in range(0, ica2.n_components_, 1)]
    # add non-brain ICs below-threshold to exclude
    for k, label in enumerate(labels):
        if label in ("brain", "eye blink", "heart beat"):
            continue
        if threshold[k] <= var[k]:
            continue
        exclude.append(k)
    ica2.exclude = exclude

    FNAME_ICA1 = DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_1st-ica.fif")
    FNAME_ICA2 = DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step3_2nd-ica.fif")

    ica1.save(FNAME_ICA1, overwrite=False)
    ica2.save(FNAME_ICA2, overwrite=False)


def fit_ica_on_data(filtered_sessions, session_picks, ica, i):
    ica = ica.fit(filtered_sessions[i], session_picks[i])
    return ica, i
