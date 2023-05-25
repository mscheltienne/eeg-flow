# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

import os
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mne import pick_types
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne.preprocessing import ICA, read_ica
from mne.viz.ica import _prepare_data_ica_properties
from mne_icalabel import label_components as label_components_iclabel

from .. import logger
from ..config import load_config
from ..utils._checks import check_type, check_value
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Dict, Tuple

    from mne.io import BaseRaw


@fill_doc
def fit_icas(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
    overwrite: bool = False,
) -> None:
    """Fit ICAs decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    overwrite : bool
        If True, overwrites existing derivatives.
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
        raw1, raw2 = _load_and_filter_raws(
            derivatives_folder / f"{fname_stem}_step2_raw.fif"
        )

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
        df_iclabel = _auto_label_components(raw2, ica2)

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


def _load_and_filter_raws(fname: Path) -> Tuple[BaseRaw, BaseRaw]:
    """Load raw recording and filter for ICA fits."""
    raw1 = read_raw_fif(fname, preload=True)
    raw2 = raw1.copy()
    raw2.drop_channels(["M1", "M2"])

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
    return raw1, raw2


def _fit_ica(raw: BaseRaw, ica_kwargs: Dict[str, Any]) -> ICA:
    """Create and fit an ICA decomposition on the provided raw recoridng."""
    ica = ICA(**ica_kwargs)
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    ica.fit(raw, picks=picks)
    return ica


def _auto_label_components(raw: BaseRaw, ica: ICA) -> pd.DataFrame:
    """Label components with ICLabel."""
    component_dict = label_components_iclabel(raw, ica, method="iclabel")
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


@fill_doc
def label_components(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
    overwrite: bool = False,
) -> None:
    """Label both ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    check_type(overwrite, (bool,), "overwrite")
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)
    os.makedirs(derivatives_folder / "plots" / "ica", exist_ok=True)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step4_reviewed_1st_{username}_ica.fif",
        derivatives_folder / f"{fname_stem}_step4_reviewed_2nd_{username}_ica.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        # However, it is not filtered.
        raw1, raw2 = _load_and_filter_raws(
            derivatives_folder / f"{fname_stem}_step2_raw.fif"
        )

        # define ICAs argument, simpler to serialize than ICas classes
        ica1 = read_ica(derivatives_folder / f"{fname_stem}_step3_1st_ica.fif")
        ica2 = read_ica(derivatives_folder / f"{fname_stem}_step3_2nd_ica.fif")
        # sanity-checks
        assert ica1.info["lowpass"] == 40.0
        assert ica1.info["custom_ref_applied"] == 0
        assert ica2.info["lowpass"] == 100.0
        assert ica2.info["custom_ref_applied"] == 1

        # annotate ICA 1 for mastoids
        figs = ica1.plot_components(
            inst=raw1,
            title=f"{fname_stem} | ICA1 components Mastoids | {username}",
            show=True,
        )
        _disconnect_onclick_title(figs)
        plt.pause(0.1)
        ica1.plot_sources(
            inst=raw1,
            title=f"{fname_stem} | ICA1 sources Mastoids | {username}",
            show=True,
            block=True,
        )
        plt.close("all")
        del figs

        # annotate ICA 2 for EEG
        figs = ica2.plot_components(
            inst=raw2,
            title=f"{fname_stem} | ICA2 compoments | {username}",
            show=True,
        )
        _disconnect_onclick_title(figs)
        plt.pause(0.1)
        ica2.plot_sources(
            inst=raw2,
            title=f"{fname_stem} | ICA2 sources | {username}",
            show=True,
            block=True,
        )
        plt.close("all")
        del figs

        # save deriatives
        logger.info("Saving derivatives.")
        ica1.save(
            derivatives_folder / f"{fname_stem}_step4_reviewed_1st_{username}_ica.fif",
            overwrite=overwrite,
        )
        ica2.save(
            derivatives_folder / f"{fname_stem}_step4_reviewed_2nd_{username}_ica.fif",
            overwrite=overwrite,
        )

        # save figures after ICAs to catch the except FileExistsError first if needed
        figs = ica1.plot_components(inst=raw1, show=False)
        plt.pause(0.1)
        for k, fig in enumerate(figs):
            fig.savefig(
                derivatives_folder / "plots" / "ica" / f"ICA1_{username}_fig{k}.svg",
                transparent=True,
            )
        plt.close("all")  # because show=False does not work at the moment
        del figs
        figs = ica2.plot_components(inst=raw2, show=False)
        plt.pause(0.1)
        for k, fig in enumerate(figs):
            fig.savefig(
                derivatives_folder / "plots" / "ica" / f"ICA2_{username}_fig{k}.svg",
                transparent=True,
            )
        plt.close("all")  # because show=False does not work at the moment
        del figs
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


def _disconnect_onclick_title(figs):
    """Disconnect the onclick_title events added by MNE to select/deselect ICs."""
    if not isinstance(figs, list):
        figs = [figs]
    for fig in figs:
        for cid, func in fig.canvas.callbacks.callbacks["button_press_event"].items():
            if func().__name__ == "onclick_title":
                fig.canvas.mpl_disconnect(cid)
                break


def compare_labels(
    participant: str,
    group: str,
    task: str,
    run: int,
    ica_id: int,
    reviewers: Tuple[str, str],
    *,
    timeout: float = 10,
    overwrite: bool = False,
):
    """Compare labels assigned to ICs by 2 reviewers.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    ica_id : int
        ID of the ICA, 1 (mastoids) or 2 (EEG).
    reviewers : tuple of length (2,)
        Username of the reviewers to load.
    %(timeout)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    check_type(ica_id, ("int",), "ica_id")
    check_value(ica_id, (1, 2), "ica_id")
    check_type(reviewers, (tuple,), "reviewers")
    assert len(reviewers) == 2
    for reviewer in reviewers:
        check_type(reviewer, (str,), "reviewer")
    assert reviewers[0] != reviewers[1]  # sanity-check
    check_type(overwrite, (bool,), "overwrite")
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    idx = "1st" if ica_id == 1 else "2nd"
    derivatives = (derivatives_folder / f"{fname_stem}_step5_reviewed_{idx}_ica.fif",)
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        # However, it is not filtered.
        raw1, raw2 = _load_and_filter_raws(
            derivatives_folder / f"{fname_stem}_step2_raw.fif"
        )
        # keep only the one we need for this function to free resources
        if ica_id == 1:
            raw = raw1
            del raw2
        else:
            raw = raw2
            del raw1

        # load ICAs
        icas = [
            read_ica(
                derivatives_folder
                / f"{fname_stem}_step4_reviewed_{idx}_{username}_ica.fif"
            )
            for username in reviewers
        ]

        # compare the sets of excluded components
        exclude_common = list(set(icas[0].exclude).intersection(set(icas[1].exclude)))
        exclude_diff = list(
            set(
                list(set(icas[0].exclude) - set(icas[1].exclude))
                + list(set(icas[1].exclude) - set(icas[0].exclude))
            )
        )
        logger.info("Set of common excluded ICs: %s", exclude_common)
        logger.info("Set of different excluded ICs: %s", exclude_diff)

        # check if we need to go further
        if len(exclude_diff) == 0:
            logger.critical(
                "Congratulation! There is no difference between the labels of both "
                "reviewers! This function aborts early. The ICA topography plot is "
                "replaced with a .txt file for traceback."
            )
            with open(
                derivatives_folder / "plots" / "ica" / f"ICA{ica_id}_rev-diff.txt", "w"
            ) as file:
                file.write("No difference between the labels of both reviewers!")
            # save derivatives
            logger.info("Saving derivatives.")
            icas[0].save(
                derivatives_folder / f"{fname_stem}_step5_reviewed_{idx}_ica.fif",
                overwrite=overwrite,
            )
            return None

        # keep only one ICA to free resources
        ica = icas[0]
        del icas

        # create figures to review the set of differently labelled ICs
        ica.exclude = []  # reset for render
        figs = ica.plot_components(
            inst=raw,
            show=True,
            picks=exclude_diff,
            title=f"{fname_stem} | ICA{ica_id} mismatch labels",
        )
        _disconnect_onclick_title(figs)
        plt.pause(0.1)
        ica.plot_sources(
            inst=raw,
            picks=exclude_diff,
            title=f"{fname_stem} | ICA{ica_id} sources",
            show=True,
            block=True,
        )
        plt.close("all")
        del figs

        # merge exclude_common which contains ICs commonly excluded with the ICs
        # selected now for exclusion
        ica.exclude += exclude_common
        ica.exclude = sorted(ica.exclude)  # for readability
        assert len(ica.exclude) == len(set(ica.exclude)), "Please contact a developer."

        # save derivatives
        logger.info("Saving derivatives.")
        ica.save(
            derivatives_folder / f"{fname_stem}_step5_reviewed_{idx}_ica.fif",
            overwrite=overwrite,
        )

        # save figures after ICAs to catch the except FileExistsError first if needed
        figs = ica.plot_components(inst=raw, show=False)
        plt.pause(0.1)
        for k, fig in enumerate(figs):
            fig.savefig(
                derivatives_folder
                / "plots"
                / "ica"
                / f"ICA{ica_id}_rev-diff_fig{k}.svg",
                transparent=True,
            )
        plt.close("all")  # because show=False does not work at the moment
        del figs
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


def apply_ica(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
    overwrite: bool = False,
):
    """Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    check_type(overwrite, (bool,), "overwrite")
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (derivatives_folder / f"{fname_stem}_step6_preprocessed_raw.fif",)
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step2_raw.fif", preload=True
        )

        # apply ICA for mastoids
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step5_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick_channels(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        # apply ICA for EEG channels
        raw.drop_channels(["M1", "M2"])
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
        raw.set_montage(None)  # just in case we have a montage left
        raw.add_reference_channels(ref_channels="CPz")
        raw.set_eeg_reference("average", projection=False)
        ica = read_ica(derivatives_folder / f"{fname_stem}_step5_reviewed_2nd_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources

        raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
        raw.add_channels([raw_mastoids])
        del raw_mastoids
        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])

        # plot for inspection
        raw.plot(theme="light", block=True)

        # save derivative
        fname = derivatives_folder / f"{fname_stem}_step6_preprocessed_raw.fif"
        raw.save(fname, overwrite=overwrite)
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
