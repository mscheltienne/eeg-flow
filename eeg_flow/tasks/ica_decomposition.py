# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from mne import pick_types
from mne.io import read_raw_fif
from mne.preprocessing import ICA
from mne.viz.ica import _prepare_data_ica_properties
from mne_icalabel import label_components

from .. import logger
from ..config import load_config
from ..utils._checks import check_type
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files

if TYPE_CHECKING:
    from typing import Any, Dict

    from mne.io import BaseRaw


@fill_doc
def fit_icas(
    participant: str,
    group: str,
    task: str,
    run: int,
    overwrite: bool = False,
    *,
    timeout: float = 10,
) -> None:
    """Fit ICAs decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    overwrite : bool
        If True, overwrites existing derivatives.
    %(timeout)s
    """
    check_type(overwrite, (bool,), "overwrite")
    # prepare folders
    _, derivatives_folder_root, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step3_1st_ica.fif",
        derivatives_folder / f"{fname_stem}_step3_2nd_ica.fif",
        derivatives_folder / f"{fname_stem}_step3_iclabel.xlsx",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        # However, it is not filtered.
        raw1 = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step2_raw.fif", preload=True
        )
        raw2 = raw1.copy()

        # ICA 1 for mastoids, referenced to CPz and filtered between 1 and 40 Hz
        raw1.filter(
            l_freq=1.0,
            h_freq=40.0,
            picks="eeg",
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )

        # ICA 2 for EEG channels, referenced to CAR and filtered between 1 and 100 Hz
        raw2.filter(
            l_freq=1.0,
            h_freq=100.0,
            picks=["eeg"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )
        raw2.set_montage(None)
        raw2.add_reference_channels(ref_channels="CPz")
        raw2.set_montage("standard_1020")
        raw2.set_eeg_reference("average", projection=False)

        # define ICAs argument, simpler to serialize than ICas classes
        kwargs = dict(
            n_components=None,
            method="picard",
            max_iter="auto",
            fit_params=dict(ortho=False, extended=True),
            random_state=888,
        )

        # run on 2 threads
        logger.info("Running ICA decomposition. It may take some time.")
        with ThreadPool(processes=2) as pool:
            results = pool.starmap(_fit_ica, [(raw1, kwargs), (raw2, kwargs)])
        ica1, ica2 = results
        # sanity-checks
        assert ica1.info["lowpass"] == 40.0
        assert ica1.info["custom_ref_applied"] == 0
        assert ica2.info["lowpass"] == 100.0
        assert ica2.info["custom_ref_applied"] == 1

        # label components
        logger.info("Running ICLabel.")
        df_iclabel = _label_components(raw2, ica2)

        # save deriatives
        logger.info("Saving derivatives.")
        ica1.save(
            derivatives_folder / f"{fname_stem}_step3_1st_ica.fif", overwrite=overwrite
        )
        ica2.save(
            derivatives_folder / f"{fname_stem}_step3_2nd_ica.fif", overwrite=overwrite
        )
        df_iclabel.to_excel(derivatives_folder / f"{fname_stem}_step3_iclabel.xlsx")
    except FileNotFoundError:
        logger.error(
            "The requested file for participant %s, group %s, task %s, run %i does "
            "not exist and will be skipped.",
            participant,
            group,
            task,
            run,
        )
    except FileExistsError:
        logger.error(
            "The destination file for participant %s, group %s, task %s, run %i "
            "already exists. Please use 'overwrite=True' to force overwriting.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def _fit_ica(raw: BaseRaw, ica_kwargs: Dict[str, Any]) -> ICA:
    """Create and fit an ICA decomposition on the provided raw recoridng."""
    ica = ICA(**ica_kwargs)
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    ica.fit(raw, picks=picks)
    return ica


def _label_components(raw: BaseRaw, ica: ICA) -> pd.DataFrame:
    """Label components with ICLabel."""
    component_dict = label_components(raw, ica, method="iclabel")
    data_icalabel = {
        "y_pred": component_dict["y_pred_proba"],
        "labels": component_dict["labels"],
    }
    df_iclabel = pd.DataFrame.from_dict(data_icalabel)

    # mark exclusion in-place in ica.exclude
    labels = component_dict["labels"]
    exclude = [
        k for k, name in enumerate(labels) if name in ("eye blink", "heart beat")
    ]
    _, _, _, data = _prepare_data_ica_properties(
        raw,
        ica,
        reject_by_annotation=True,
        reject="auto",
    )

    ica_data = np.swapaxes(data, 0, 1)
    var = np.var(ica_data, axis=2)  # (n_components, n_epochs)
    var = np.var(var.T / np.linalg.norm(var, axis=1), axis=0)
    # linear fit to determine the variance thresholds
    z = np.polyfit(range(0, ica.n_components_, 1), var, 1)
    threshold = [z[0] * x + z[1] for x in range(0, ica.n_components_, 1)]
    # add non-brain ICs below-threshold to exclude
    for k, label in enumerate(labels):
        if label in ("brain", "eye blink", "heart beat"):
            continue
        if threshold[k] <= var[k]:
            continue
        exclude.append(k)
    ica.exclude = exclude

    return df_iclabel
