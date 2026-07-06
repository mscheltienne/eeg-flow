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
from mne import create_info, pick_types
from mne.io import RawArray, read_raw_fif
from mne.io.constants import FIFF
from mne.preprocessing import ICA, read_ica
from mne.viz.ica import _prepare_data_ica_properties

from ..config import load_config
from ..utils._checks import check_type, check_value
from ..utils._docs import fill_doc
from ..utils._imports import import_optional_dependency
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files
from ..utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from mne.io import BaseRaw


@fill_doc
def fit_icas(
    participant: str,
    group: str,
    task: str,
    run: int,
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
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step4_1st_ica.fif",
        derivatives_folder / f"{fname_stem}_step4_2nd_ica.fif",
        derivatives_folder / f"{fname_stem}_step4_iclabel.xlsx",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        # However, it is not filtered.
        raw1, raw2 = _load_and_filter_raws(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif"
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
        if import_optional_dependency("mne_icalabel", raise_error=False) is not None:
            logger.info("Running ICLabel.")
            df_iclabel = _auto_label_components(raw2, ica2)
        else:
            logger.info("MNE-ICAlabel is not installed. Skipping.")
            df_iclabel = None

        # save deriatives
        logger.info("Saving derivatives.")
        ica1.save(
            derivatives_folder / f"{fname_stem}_step4_1st_ica.fif", overwrite=False
        )
        ica2.save(
            derivatives_folder / f"{fname_stem}_step4_2nd_ica.fif", overwrite=False
        )
        if df_iclabel is not None:
            df_iclabel.to_excel(derivatives_folder / f"{fname_stem}_step4_iclabel.xlsx")
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    except Exception as error:
        logger.error(
            "The file for participant %s, group %s, task %s, run %i could not be "
            "processed.",
            participant,
            group,
            task,
            run,
        )
        logger.exception(error)
    finally:
        for lock in locks:
            lock.release()
        del locks


def _load_and_filter_raws(fname: Path) -> tuple[BaseRaw, BaseRaw]:
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


def _fit_ica(raw: BaseRaw, ica_kwargs: dict[str, Any]) -> ICA:
    """Create and fit an ICA decomposition on the provided raw recoridng."""
    ica = ICA(**ica_kwargs)
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    ica.fit(raw, picks=picks)
    return ica


def _auto_label_components(raw: BaseRaw, ica: ICA) -> pd.DataFrame:
    """Label components with ICLabel."""
    from mne_icalabel import label_components as label_components_iclabel

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
) -> None:
    """Label both ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)
    os.makedirs(derivatives_folder / "plots" / "ica", exist_ok=True)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step5_reviewed_1st_{username}_ica.fif",
        derivatives_folder / f"{fname_stem}_step5_reviewed_2nd_{username}_ica.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError

        # define ICAs argument, simpler to serialize than ICas classes
        ica1 = read_ica(derivatives_folder / f"{fname_stem}_step4_1st_ica.fif")
        ica2 = read_ica(derivatives_folder / f"{fname_stem}_step4_2nd_ica.fif")
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        # However, it is not filtered.
        raw1, raw2 = _load_and_filter_raws(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif"
        )
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
            title=f"{fname_stem} | ICA2 components | {username}",
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
            derivatives_folder / f"{fname_stem}_step5_reviewed_1st_{username}_ica.fif",
            overwrite=False,
        )
        ica2.save(
            derivatives_folder / f"{fname_stem}_step5_reviewed_2nd_{username}_ica.fif",
            overwrite=False,
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
            "already exists.",
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
    reviewers: tuple[str, str],
    *,
    timeout: float = 10,
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
    """
    check_type(ica_id, ("int-like",), "ica_id")
    check_value(ica_id, (1, 2), "ica_id")
    check_type(reviewers, (tuple,), "reviewers")
    assert len(reviewers) == 2
    for reviewer in reviewers:
        check_type(reviewer, (str,), "reviewer")
    assert reviewers[0] != reviewers[1]  # sanity-check
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    idx = "1st" if ica_id == 1 else "2nd"
    derivatives = (derivatives_folder / f"{fname_stem}_step6_reviewed_{idx}_ica.fif",)
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        # However, it is not filtered.
        raw1, raw2 = _load_and_filter_raws(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif"
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
                / f"{fname_stem}_step5_reviewed_{idx}_{username}_ica.fif"
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
                derivatives_folder / f"{fname_stem}_step6_reviewed_{idx}_ica.fif",
                overwrite=False,
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
            derivatives_folder / f"{fname_stem}_step6_reviewed_{idx}_ica.fif",
            overwrite=False,
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
            "already exists.",
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
):
    """Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (  # derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step22a_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step22b_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step22c_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step22d_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step22e_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step42mast1_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step42mast2_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step42e1-noICA1-noICA2_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step42e2-wICA1-noICA2_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step42e3-wICA1-wICA2_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step52mast1_preCAR_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step52mast2_postCAR_52c_EEG_postICA2_preprocessed_raw.fif",
        derivatives_folder
        / f"{fname_stem}_step52ee_postMastavg_preprocessed_raw.fif",  # skip CPZ
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")

        # fname = derivatives_folder / f"{fname_stem}_step42mast1_preprocessed_raw.fif"
        # raw_mastoids.save(fname, overwrite=False)

        # raw_mastoids_noICA1 = raw_mastoids.copy()
        # raw_mastoids_noICA1.pick(["M1", "M2"])

        ica.apply(raw_mastoids)

        # fname = derivatives_folder / f"{fname_stem}_step42mast2_preprocessed_raw.fif"
        # raw_mastoids.save(fname, overwrite=False)

        del ica  # free resources
        raw_mastoids.pick(["M1", "M2", "TRIGGER"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        ## trick MNE in thinking that a custom-ref has been applied
        # with raw_mastoids_noICA1.info._unlock():
        #    raw_mastoids_noICA1.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

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

        # fname = derivatives_folder / f"{fname_stem}_step22a_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        raw.set_montage(None)  # just in case we have a montage left
        raw.add_reference_channels(ref_channels="CPz")

        # raw.drop_channels(raw.info['bads'])
        # raw.drop_channels(["vEOG",'EDA','ECG','hEOG'])

        # manual reref for mastoids
        # Apply Common Average Reference (CAR) by subtracting the average of all channels
        # Calculate the average of all EEG channels (excluding vEOG for this step)
        all_eeg_data = raw.get_data(picks="eeg")
        avg_eeg_data = all_eeg_data.mean(axis=0)

        processed_channels = []

        sfreq = raw.info["sfreq"]

        # fname = derivatives_folder / f"{fname_stem}_step52mast1_preCAR_preprocessed_raw.fif"
        # raw_mastoids.save(fname, overwrite=False)

        # Process both mastoids
        for ch in ["M1", "M2"]:
            if ch != "TRIGGER":
                mastoid_data = raw_mastoids.get_data(picks=ch)
                mastoid_data_car = mastoid_data - avg_eeg_data  # shape: (1, n_times)

                ch_info = create_info([ch], sfreq, ch_types=["eeg"])
                ch_raw = RawArray(mastoid_data_car, ch_info)
                processed_channels.append(ch_raw)

        # reref EEG to CAR
        del raw_mastoids
        raw.set_eeg_reference("average", projection=False)

        # fname = derivatives_folder / f"{fname_stem}_step22b_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_2nd_ica.fif")
        ica.apply(raw)
        del ica  # free resources

        # fname = derivatives_folder / f"{fname_stem}_step22c_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        # Combine EEG and mastoids
        # Add all the re-referenced channels
        for ch_raw in processed_channels:
            raw.add_channels([ch_raw], force_update_info=True)

        # fname = derivatives_folder / f"{fname_stem}_step52mast2_postCAR_52c_EEG_postICA2_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        # raw.set_eeg_reference(["CPz"], projection=False)  # change reference back

        # raw_for_noICA1 = raw.copy()

        # raw.add_channels([raw_mastoids])
        # raw_for_noICA1.add_channels([raw_mastoids_noICA1])
        # del raw_mastoids
        # del raw_mastoids_noICA1

        # fname = derivatives_folder / f"{fname_stem}_step22d_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])
        raw.drop_channels(["M1", "M2"])

        # fname = derivatives_folder / f"{fname_stem}_step42e3-wICA1-wICA2_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        fname = (
            derivatives_folder
            / f"{fname_stem}_step52ee_postMastavg_preprocessed_raw.fif"
        )
        raw.save(fname, overwrite=False)

        # raw_for_noICA1.set_montage("standard_1020")  # add montage for non-mastoids
        # raw_for_noICA1.set_eeg_reference(["M1", "M2"])
        # raw_for_noICA1.drop_channels(["M1", "M2"])

        # save derivative
        # fname = derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        # fname = derivatives_folder / f"{fname_stem}_step42e2-wICA1-noICA2_preprocessed_raw.fif"
        # raw.save(fname, overwrite=False)

        # fname = derivatives_folder / f"{fname_stem}_step42e1-noICA1-noICA2_preprocessed_raw.fif"
        # raw_for_noICA1.save(fname, overwrite=False)

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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_original(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder
        / f"{fname_stem}_step70dd_oldwithoutautoreject_preprocessed_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_2nd_ica.fif")
        ica.apply(raw)
        del ica  # free resources

        raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
        raw.add_channels([raw_mastoids])
        del raw_mastoids
        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])
        raw.drop_channels(["M1", "M2"])

        # save derivative
        fname = (
            derivatives_folder
            / f"{fname_stem}_step70dd_oldwithoutautoreject_preprocessed_raw.fif"
        )
        raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_interpolate(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """Apply the reviewed ICA decomposition, then interpolate bads.

    This is a dirty copy to not break the previous steps for now.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (derivatives_folder / f"{fname_stem}_step10_preprocessed_raw.fif",)
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_2nd_ica.fif")
        ica.apply(raw)
        del ica  # free resources

        raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
        raw.add_channels([raw_mastoids])
        del raw_mastoids
        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])
        raw.drop_channels(["M1", "M2"])

        # interpolate bads
        raw.interpolate_bads()
        # warning, tmin will be t=0 for all subsequent functions. New first_samp and
        # last_samp are set accordingly.
        raw.crop(tmin=30.0)

        # save derivative
        fname = derivatives_folder / f"{fname_stem}_step10_preprocessed_raw.fif"
        raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step12_preprocessed_EOG_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_2nd_ica.fif")
        ica.apply(raw)
        del ica  # free resources

        raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
        raw.add_channels([raw_mastoids])

        # resume the final rereference specifically to the EEG channels
        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])

        # Let's assume you want to subtract M1 and M2 from the vEOG
        # First, get the data for the vEOG and M1/M2
        vEOG_data = raw.get_data(picks="vEOG")
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")

        # Combine M1 and M2 as a reference (e.g., average or difference)
        average_mastoids = (M1_data + M2_data) / 2

        # Subtract the average mastoid reference from the vEOG
        vEOG_data_ref = vEOG_data - average_mastoids

        # Put the re-referenced vEOG data back into the raw object
        raw._data[raw.info["ch_names"].index("vEOG")] = vEOG_data_ref

        del raw_mastoids

        # Filter both EOG
        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks="eog",
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )

        raw.drop_channels(["M1", "M2"])

        # save derivative
        fname = derivatives_folder / f"{fname_stem}_step12_preprocessed_EOG_raw.fif"
        raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_correct(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step12bb_preprocessed_EOG_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        # apply ICA for EEG channels
        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )

        raw.set_montage(None)  # just in case we have a montage left
        raw.add_reference_channels(ref_channels="CPz")
        raw.set_eeg_reference("average", projection=False)

        # Get the CPz data from the raw object after it's been referenced to CAR
        CPz_CARref_data = raw.get_data(picks="CPz")

        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_2nd_ica.fif")
        ica.apply(raw)
        del ica  # free resources

        raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
        raw.add_channels([raw_mastoids])

        # resume the final rereference specifically to the EEG channels
        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])

        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")

        # Apply Common Average Reference (CAR) by subtracting the average of all channels
        # Calculate the average of all EEG channels (excluding vEOG for this step)
        all_eeg_data = raw.get_data(picks="eeg")
        avg_eeg_data = all_eeg_data.mean(axis=0)

        # Subtract the average EEG data from the vEOG to perform CAR
        vEOG_data_car = vEOG_data - avg_eeg_data

        ###############################
        # Subtract CPz from when it was rerefenced to CAR from the vEOG to re-reference to CPz
        vEOG_data_cpz = vEOG_data_car - CPz_CARref_data

        ###############################

        # Let's assume you want to subtract M1 and M2 from the vEOG
        #  M1/M2
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")

        # Combine M1 and M2 as a reference (e.g., average or difference)
        average_mastoids = (M1_data + M2_data) / 2

        # Subtract the average mastoid reference from the vEOG
        vEOG_data_Mref = vEOG_data_cpz - average_mastoids

        # Put the re-referenced vEOG data back into the raw object
        raw._data[raw.info["ch_names"].index("vEOG")] = vEOG_data_Mref

        del raw_mastoids

        ## Filter both EOG
        # raw.filter(
        #    l_freq=0.5,
        #    h_freq=40.0,
        #    picks='eog',
        #    method='fir',
        #    phase='zero-double',
        #    fir_window='hamming',
        #    fir_design='firwin',
        #    pad='edge',
        # )

        raw.drop_channels(["M1", "M2"])

        # save derivative
        fname = derivatives_folder / f"{fname_stem}_step12bb_preprocessed_EOG_raw.fif"
        raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_correct_noICA2(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder
        / f"{fname_stem}_step12cc_preprocessed_EOG_EEGnoICA2_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        # apply ICA for EEG channels
        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )
        raw.set_montage(None)  # just in case we have a montage left
        raw.add_reference_channels(ref_channels="CPz")
        raw.set_eeg_reference("average", projection=False)

        # Get the CPz data from the raw object after it's been referenced to CAR
        CPz_CARref_data = raw.get_data(picks="CPz")

        raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
        raw.add_channels([raw_mastoids])

        # resume the final rereference specifically to the EEG channels
        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])

        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")

        # Apply Common Average Reference (CAR) by subtracting the average of all channels
        # Calculate the average of all EEG channels (excluding vEOG for this step)
        all_eeg_data = raw.get_data(picks="eeg")
        avg_eeg_data = all_eeg_data.mean(axis=0)

        # Subtract the average EEG data from the vEOG to perform CAR
        vEOG_data_car = vEOG_data - avg_eeg_data

        ###############################
        # Subtract CPz from when it was rerefenced to CAR from the vEOG to re-reference to CPz
        vEOG_data_cpz = vEOG_data_car - CPz_CARref_data

        ###############################

        # Let's assume you want to subtract M1 and M2 from the vEOG
        #  M1/M2
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")

        # Combine M1 and M2 as a reference (e.g., average or difference)
        average_mastoids = (M1_data + M2_data) / 2

        # Subtract the average mastoid reference from the vEOG
        vEOG_data_Mref = vEOG_data_cpz - average_mastoids

        # Put the re-referenced vEOG data back into the raw object
        raw._data[raw.info["ch_names"].index("vEOG")] = vEOG_data_Mref

        del raw_mastoids

        ## Filter both EOG
        # raw.filter(
        #    l_freq=0.5,
        #    h_freq=40.0,
        #    picks='eog',
        #    method='fir',
        #    phase='zero-double',
        #    fir_window='hamming',
        #    fir_design='firwin',
        #    pad='edge',
        # )

        raw.drop_channels(["M1", "M2"])

        # save derivative
        fname = (
            derivatives_folder
            / f"{fname_stem}_step12cc_preprocessed_EOG_EEGnoICA2_raw.fif"
        )
        raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_final(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step12d_preprocessed_EOG_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )
        raw.set_montage(None)  # just in case we have a montage left
        raw.set_montage("standard_1020")  # add montage for non-mastoids

        # assuming all good for Fp1
        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")
        Fp1_data = raw.get_data(picks="Fp1")

        ###############################
        # Let's assume you want to subtract M1 and M2 from the vEOG
        #  M1/M2
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")

        # Combine M1 and M2 as a reference (e.g., average or difference)
        average_mastoids = (M1_data + M2_data) / 2

        # Subtract the average EEG data from the vEOG to perform CAR
        vEOG_data_Mref = vEOG_data - average_mastoids
        Fp1_data_Mref = Fp1_data - average_mastoids

        hEOG_raw = raw.copy().pick(
            ["hEOG", "TRIGGER"]
        )  # or use pick(picks='hEOG') if needed

        sfreq = raw.info["sfreq"]

        # vEOG
        veog_info = create_info(["vEOG"], sfreq, ch_types=["eog"])
        veog_raw = RawArray(vEOG_data_Mref, veog_info)

        # FPA
        Fp1_info = create_info(["Fp1"], sfreq, ch_types=["eeg"])
        Fp1_raw = RawArray(Fp1_data_Mref, Fp1_info)

        EOG_raw = hEOG_raw.copy().add_channels(
            [veog_raw, Fp1_raw], force_update_info=True
        )

        del raw_mastoids

        # save derivative
        fname = derivatives_folder / f"{fname_stem}_step12d_preprocessed_EOG_raw.fif"
        EOG_raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_final_extend(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step20zz_preprocessed_EOGEEG3_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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
        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )
        raw.set_montage(None)  # just in case we have a montage left
        raw.set_montage("standard_1020")  # add montage for non-mastoids

        # assuming all good for Fp1
        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")

        # Define EEG channels to process (excluding Fp1 — handled separately)
        eeg_channels = [
            "Fp1",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "Fz",
            "F7",
            "F5",
            "F3",
            "F1",
            "F2",
            "F4",
            "F6",
            "F8",
        ]

        # Sampling frequency
        sfreq = raw.info["sfreq"]

        # Get mastoid reference (average of M1 and M2)
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")
        average_mastoids = (M1_data + M2_data) / 2

        # Process vEOG (EOG type)
        vEOG_data = raw.get_data(picks="vEOG")
        vEOG_data_Mref = vEOG_data - average_mastoids
        veog_info = create_info(["vEOG"], sfreq, ch_types=["eog"])
        veog_raw = RawArray(vEOG_data_Mref, veog_info)

        # List to collect all processed EEG RawArrays
        processed_channels = [veog_raw]

        # Process each EEG channel
        for ch in eeg_channels:
            ch_data = raw.get_data(picks=ch)
            ch_data_Mref = ch_data - average_mastoids
            ch_info = create_info([ch], sfreq, ch_types=["eeg"])
            ch_raw = RawArray(ch_data_Mref, ch_info)
            processed_channels.append(ch_raw)

        # Start with hEOG and TRIGGER channels
        EOG_raw = raw.copy().pick(["hEOG", "TRIGGER"])

        # Add all the re-referenced channels
        for ch_raw in processed_channels:
            EOG_raw.add_channels([ch_raw], force_update_info=True)

        # Clean up
        del raw_mastoids
        ###############################

        # save derivative
        fname = (
            derivatives_folder / f"{fname_stem}_step20zz_preprocessed_EOGEEG3_raw.fif"
        )
        EOG_raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_final_extend_all_eeg(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step30zz_preprocessed_EOGEEGall_raw.fif",
    )

    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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

        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")
        ica.apply(raw_mastoids)
        del ica  # free resources
        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )

        raw.set_montage(None)  # just in case we have a montage left
        raw.set_montage("standard_1020")  # add montage for non-mastoids

        # assuming all good for Fp1
        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")

        # Define EEG channels to process (excluding Fp1 — handled separately)
        raw.add_reference_channels(ref_channels="CPz")

        eeg_channels = [
            "Fp1",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "Fz",
            "F7",
            "F5",
            "F3",
            "F1",
            "F2",
            "F4",
            "F6",
            "F8",
        ]
        eeg_channels = [
            "Fp1",
            "Fpz",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "CP5",
            "CP1",
            "CP2",
            "CP6",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "POz",
            "O1",
            "O2",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "F5",
            "F1",
            "F2",
            "F6",
            "FC3",
            "FCz",
            "FC4",
            "C5",
            "C1",
            "C2",
            "C6",
            "CP3",
            "CP4",
            "P5",
            "P1",
            "P2",
            "P6",
            "PO5",
            "PO3",
            "PO4",
            "PO6",
            "FT7",
            "FT8",
            "TP7",
            "TP8",
            "PO7",
            "PO8",
            "Oz",
            "CPz",
        ]

        # Sampling frequency
        sfreq = raw.info["sfreq"]
        bads = raw.info["bads"]

        # Get mastoid reference (average of M1 and M2)
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")
        average_mastoids = (M1_data + M2_data) / 2

        # Process vEOG (EOG type)
        vEOG_data = raw.get_data(picks="vEOG")
        vEOG_data_Mref = vEOG_data - average_mastoids
        veog_info = create_info(["vEOG"], sfreq, ch_types=["eog"])
        veog_raw = RawArray(vEOG_data_Mref, veog_info)

        # List to collect all processed EEG RawArrays
        processed_channels = [veog_raw]

        # Process each EEG channel
        for ch in eeg_channels:
            ch_data = raw.get_data(picks=ch)
            ch_data_Mref = ch_data - average_mastoids
            ch_info = create_info([ch], sfreq, ch_types=["eeg"])
            ch_raw = RawArray(ch_data_Mref, ch_info)
            processed_channels.append(ch_raw)

        # Start with hEOG and TRIGGER channels
        EOG_raw = raw.copy().pick(["hEOG", "TRIGGER"])

        # Add all the re-referenced channels
        for ch_raw in processed_channels:
            EOG_raw.add_channels([ch_raw], force_update_info=True)

        EOG_raw.info["bads"] = bads
        # Clean up
        del raw_mastoids
        ###############################

        # save derivative
        fname = (
            derivatives_folder / f"{fname_stem}_step30zz_preprocessed_EOGEEGall_raw.fif"
        )
        EOG_raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_final_extend_all_eeg_noICA1(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder
        / f"{fname_stem}_step30zz_preprocessed_EOGEEGall_rerefmastwithoutICA1raw.fif",
    )

    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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

        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )

        raw.set_montage(None)  # just in case we have a montage left
        raw.set_montage("standard_1020")  # add montage for non-mastoids

        # assuming all good for Fp1
        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")

        # Define EEG channels to process (excluding Fp1 — handled separately)
        raw.add_reference_channels(ref_channels="CPz")

        eeg_channels = [
            "Fp1",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "Fz",
            "F7",
            "F5",
            "F3",
            "F1",
            "F2",
            "F4",
            "F6",
            "F8",
        ]
        eeg_channels = [
            "Fp1",
            "Fpz",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "CP5",
            "CP1",
            "CP2",
            "CP6",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "POz",
            "O1",
            "O2",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "F5",
            "F1",
            "F2",
            "F6",
            "FC3",
            "FCz",
            "FC4",
            "C5",
            "C1",
            "C2",
            "C6",
            "CP3",
            "CP4",
            "P5",
            "P1",
            "P2",
            "P6",
            "PO5",
            "PO3",
            "PO4",
            "PO6",
            "FT7",
            "FT8",
            "TP7",
            "TP8",
            "PO7",
            "PO8",
            "Oz",
            "CPz",
        ]

        # Sampling frequency
        sfreq = raw.info["sfreq"]

        # Get mastoid reference (average of M1 and M2)
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")
        average_mastoids = (M1_data + M2_data) / 2

        # Process vEOG (EOG type)
        vEOG_data = raw.get_data(picks="vEOG")
        vEOG_data_Mref = vEOG_data - average_mastoids
        veog_info = create_info(["vEOG"], sfreq, ch_types=["eog"])
        veog_raw = RawArray(vEOG_data_Mref, veog_info)

        # List to collect all processed EEG RawArrays
        processed_channels = [veog_raw]

        # Process each EEG channel
        for ch in eeg_channels:
            ch_data = raw.get_data(picks=ch)
            ch_data_Mref = ch_data - average_mastoids
            ch_info = create_info([ch], sfreq, ch_types=["eeg"])
            ch_raw = RawArray(ch_data_Mref, ch_info)
            processed_channels.append(ch_raw)

        # Start with hEOG and TRIGGER channels
        EOG_raw = raw.copy().pick(["hEOG", "TRIGGER"])

        # Add all the re-referenced channels
        for ch_raw in processed_channels:
            EOG_raw.add_channels([ch_raw], force_update_info=True)

        # Clean up
        del raw_mastoids
        ###############################

        # save derivative
        fname = (
            derivatives_folder
            / f"{fname_stem}_step30zz_preprocessed_EOGEEGall_rerefmastwithoutICA1raw.fif"
        )
        EOG_raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_reref_EOG_final_extend_all_eeg_onlyfilter(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """include a step to preprocess the vEOG, as we need the mastoids, go from this step just to be clean. Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder
        / f"{fname_stem}_step30zz_preprocessed_EOGEEGall_originalonCPZ_raw.fif",
    )

    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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

        raw_mastoids.pick(["M1", "M2"])

        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        raw.drop_channels(["M1", "M2"])

        raw.filter(
            l_freq=0.5,
            h_freq=40.0,
            picks=["eeg", "eog"],
            method="fir",
            phase="zero-double",
            fir_window="hamming",
            fir_design="firwin",
            pad="edge",
        )

        raw.set_montage(None)  # just in case we have a montage left
        raw.set_montage("standard_1020")  # add montage for non-mastoids

        # assuming all good for Fp1
        ###############################
        # Get the raw data for vEOG
        vEOG_data = raw.get_data(picks="vEOG")

        # Define EEG channels to process (excluding Fp1 — handled separately)
        raw.add_reference_channels(ref_channels="CPz")

        eeg_channels = [
            "Fp1",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "Fz",
            "F7",
            "F5",
            "F3",
            "F1",
            "F2",
            "F4",
            "F6",
            "F8",
        ]
        eeg_channels = [
            "Fp1",
            "Fpz",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "CP5",
            "CP1",
            "CP2",
            "CP6",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "POz",
            "O1",
            "O2",
            "AF7",
            "AF3",
            "AF4",
            "AF8",
            "F5",
            "F1",
            "F2",
            "F6",
            "FC3",
            "FCz",
            "FC4",
            "C5",
            "C1",
            "C2",
            "C6",
            "CP3",
            "CP4",
            "P5",
            "P1",
            "P2",
            "P6",
            "PO5",
            "PO3",
            "PO4",
            "PO6",
            "FT7",
            "FT8",
            "TP7",
            "TP8",
            "PO7",
            "PO8",
            "Oz",
            "CPz",
        ]

        # Sampling frequency
        sfreq = raw.info["sfreq"]

        # Get mastoid reference (average of M1 and M2)
        M1_data = raw_mastoids.get_data(picks="M1")
        M2_data = raw_mastoids.get_data(picks="M2")
        average_mastoids = (M1_data + M2_data) / 2

        # Process vEOG (EOG type)
        vEOG_data = raw.get_data(picks="vEOG")
        veog_info = create_info(["vEOG"], sfreq, ch_types=["eog"])
        veog_raw = RawArray(vEOG_data, veog_info)

        # List to collect all processed EEG RawArrays
        processed_channels = [veog_raw]

        # Process each EEG channel
        for ch in eeg_channels:
            ch_data = raw.get_data(picks=ch)
            ch_info = create_info([ch], sfreq, ch_types=["eeg"])
            ch_raw = RawArray(ch_data, ch_info)
            processed_channels.append(ch_raw)

        # Start with hEOG and TRIGGER channels
        EOG_raw = raw.copy().pick(["hEOG", "TRIGGER"])

        # Add all the re-referenced channels
        for ch_raw in processed_channels:
            EOG_raw.add_channels([ch_raw], force_update_info=True)

        # Clean up
        del raw_mastoids
        ###############################

        # save derivative
        fname = (
            derivatives_folder
            / f"{fname_stem}_step30zz_preprocessed_EOGEEGall_originalonCPZ_raw.fif"
        )
        EOG_raw.save(fname, overwrite=False)
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks


def apply_ica_journey_mastoids(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
):
    """Apply the reviewed ICA decomposition.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, username = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step60-1-originalmast_preprocessed_raw.fif",
        # derivatives_folder / f"{fname_stem}_step60-2-mastICA1_preprocessed_raw.fif"
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        # The raw saved after interpolation of bridges already contains bad channels and
        # segments. No need to reload the "info" and "oddball_with_bads" annotations.
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step3_with-bads_raw.fif", preload=True
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

        fname = (
            derivatives_folder
            / f"{fname_stem}_step60-1-originalmast_preprocessed_raw.fif"
        )
        raw_mastoids.save(fname, overwrite=False)

        # ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_1st_ica.fif")

        # ica.apply(raw_mastoids)

        # fname = derivatives_folder / f"{fname_stem}_step60-2-mastICA1_preprocessed_raw.fif"
        # raw_mastoids.save(fname, overwrite=False)

        # del ica  # free resources
        """raw_mastoids.pick(["M1", "M2","TRIGGER"])

        
        # trick MNE in thinking that a custom-ref has been applied
        with raw_mastoids.info._unlock():
            raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

        ## trick MNE in thinking that a custom-ref has been applied
        #with raw_mastoids_noICA1.info._unlock():
        #    raw_mastoids_noICA1.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON


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

        #fname = derivatives_folder / f"{fname_stem}_step22a_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)

        raw.set_montage(None)  # just in case we have a montage left
        raw.add_reference_channels(ref_channels="CPz")




        #raw.drop_channels(raw.info['bads']) 
        #raw.drop_channels(["vEOG",'EDA','ECG','hEOG'])

        #manual reref for mastoids
        # Apply Common Average Reference (CAR) by subtracting the average of all channels
        # Calculate the average of all EEG channels (excluding vEOG for this step)
        all_eeg_data = raw.get_data(picks='eeg')
        avg_eeg_data = all_eeg_data.mean(axis=0)

        processed_channels = []

        sfreq = raw.info['sfreq']

        fname = derivatives_folder / f"{fname_stem}_step52mast1_preCAR_preprocessed_raw.fif"
        raw_mastoids.save(fname, overwrite=False)

        # Process both mastoids
        for ch in ["M1", "M2"]:
            if ch != "TRIGGER":
                mastoid_data = raw_mastoids.get_data(picks=ch)
                mastoid_data_car = mastoid_data - avg_eeg_data  # shape: (1, n_times)

                ch_info = create_info([ch], sfreq, ch_types=['eeg'])
                ch_raw = RawArray(mastoid_data_car, ch_info)
                processed_channels.append(ch_raw)

        #reref EEG to CAR
        del raw_mastoids
        raw.set_eeg_reference("average", projection=False)

        

        #fname = derivatives_folder / f"{fname_stem}_step22b_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)

        ica = read_ica(derivatives_folder / f"{fname_stem}_step6_reviewed_2nd_ica.fif")
        ica.apply(raw)
        del ica  # free resources

        #fname = derivatives_folder / f"{fname_stem}_step22c_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)

        # Combine EEG and mastoids
        # Add all the re-referenced channels
        for ch_raw in processed_channels:
            raw.add_channels([ch_raw], force_update_info=True)

        fname = derivatives_folder / f"{fname_stem}_step52mast2_postCAR_52c_EEG_postICA2_preprocessed_raw.fif"
        raw.save(fname, overwrite=False)

       


        #raw.set_eeg_reference(["CPz"], projection=False)  # change reference back

        #raw_for_noICA1 = raw.copy()

        #raw.add_channels([raw_mastoids])
        #raw_for_noICA1.add_channels([raw_mastoids_noICA1])
        #del raw_mastoids
        #del raw_mastoids_noICA1

        #fname = derivatives_folder / f"{fname_stem}_step22d_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)

        raw.set_montage("standard_1020")  # add montage for non-mastoids
        raw.set_eeg_reference(["M1", "M2"])
        raw.drop_channels(["M1", "M2"])

        #fname = derivatives_folder / f"{fname_stem}_step42e3-wICA1-wICA2_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)   

        fname = derivatives_folder / f"{fname_stem}_step52e_postMastavg_preprocessed_raw.fif"
        raw.save(fname, overwrite=False)

        #raw_for_noICA1.set_montage("standard_1020")  # add montage for non-mastoids
        #raw_for_noICA1.set_eeg_reference(["M1", "M2"])
        #raw_for_noICA1.drop_channels(["M1", "M2"])

        # save derivative
        #fname = derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)

        #fname = derivatives_folder / f"{fname_stem}_step42e2-wICA1-noICA2_preprocessed_raw.fif"
        #raw.save(fname, overwrite=False)

        #fname = derivatives_folder / f"{fname_stem}_step42e1-noICA1-noICA2_preprocessed_raw.fif"
        #raw_for_noICA1.save(fname, overwrite=False)
"""
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
            "already exists.",
            participant,
            group,
            task,
            run,
        )
    finally:
        for lock in locks:
            lock.release()
        del locks
