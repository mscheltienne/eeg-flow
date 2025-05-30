from __future__ import annotations  # c.f. PEP 563, PEP 649

import time
from collections import Counter
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import pandas as pd
from autoreject import get_rejection_threshold
from mne import Epochs, find_events, read_epochs, read_evokeds
from mne.epochs import make_metadata as make_metadata_mne
from mne.io import read_raw_fif
from mne.preprocessing import compute_current_source_density
from mne.utils import check_version

if check_version("mne", "1.6"):
    from mne._fiff.pick import _picks_to_idx
else:
    from mne.io.pick import _picks_to_idx

from ..config import load_config, load_triggers
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files
from ..utils.logs import logger

if TYPE_CHECKING:
    from typing import Optional

    from matplotlib import pyplot as plt
    from mne import Evoked
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw
    from numpy.typing import DTypeLike, NDArray

    ScalarIntType: tuple[DTypeLike, ...] = (np.int8, np.int16, np.int32, np.int64)


_TOO_QUICK_THRESHOLD: float = 0.2


def stimlocked_to_CSD(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
) -> None:
    """Transform the final epochs with a CSD when needed.

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
    fname_stem = get_fname(participant, group, task, run)
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )

    # lock the output derivative files
    # create locks

    derivatives = [
        derivatives_folder / f"{fname_stem}_step8b_stimlocked-novel_CSD-ave.fif",
        derivatives_folder / f"{fname_stem}_step8b_stimlocked-standard_CSD-ave.fif",
        derivatives_folder / f"{fname_stem}_step8b_stimlocked-target_CSD-ave.fif",
    ]

    locks = lock_files(*derivatives, timeout=timeout)

    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError

        # load previous steps (normal epoching)
        evoked_novel = read_evokeds(
            derivatives_folder / f"{fname_stem}_step8_stimlocked-novel-ave.fif",
        )[0]

        # evoked_standard = read_evokeds(
        #     derivatives_folder / f"{fname_stem}_step8_stimlocked-standard-ave.fif",
        # )[0]

        # evoked_target = read_evokeds(
        #     derivatives_folder / f"{fname_stem}_step8_stimlocked-target-ave.fif",
        # )[0]

        evoked_novel_CSD = compute_current_source_density(
            evoked_novel.interpolate_bads(),
            stiffness=3,
            n_legendre_terms=15,
            verbose=100,
        )
        evoked_standard_CSD = compute_current_source_density(
            evoked_standard.interpolate_bads(),
            stiffness=3,
            n_legendre_terms=15,
            verbose=100,
        )
        evoked_target_CSD = compute_current_source_density(
            evoked_target.interpolate_bads(),
            stiffness=3,
            n_legendre_terms=15,
            verbose=100,
        )

        # save epochs, drop-log and evoked files
        if evoked_novel_CSD is not None:
            evoked_novel_CSD.save(
                derivatives_folder / f"{fname_stem}_step8b_stimlocked-novel_CSD-ave.fif"
            )
        if evoked_standard_CSD is not None:
            evoked_standard_CSD.save(
                derivatives_folder
                / f"{fname_stem}_step8b_stimlocked-standard_CSD-ave.fif"
            )
        if evoked_target_CSD is not None:
            evoked_target_CSD.save(
                derivatives_folder
                / f"{fname_stem}_step8b_stimlocked-target_CSD-ave.fif"
            )

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


def response_to_CSD(
    participant: str,
    group: str,
    task: str,
    run: int,
    BC_response: str,
    *,
    timeout: float = 10,
) -> None:
    """Transform the final epochs with a CSD when needed.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(BC_response)s
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder_root, _ = load_config()
    fname_stem = get_fname(participant, group, task, run)
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )

    # lock the output derivative files
    # create locks
    # "400200resp","2000resp","400200resp_CSD","2000resp_CSD"

    derivatives = [
        derivatives_folder
        / f"{fname_stem}_step8b_responselocked-{BC_response}_CSD-cleaned-epo.fif",
        derivatives_folder
        / f"{fname_stem}_step8b_responselocked-{BC_response}_CSD-ave.fif",
    ]

    locks = lock_files(*derivatives, timeout=timeout)

    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError

        # load previous steps (normal epoching)

        if BC_response == "2000stim":
            epochs = read_epochs(
                derivatives_folder
                / f"{fname_stem}_step8_responselocked-cleaned-epo.fif",
            )

            evoked = read_evokeds(
                derivatives_folder / f"{fname_stem}_step8_responselocked-ave.fif",
            )[0]
        else:
            epochs = read_epochs(
                derivatives_folder
                / f"{fname_stem}_step8_responselocked-{BC_response}-cleaned-epo.fif",
            )

            evoked = read_evokeds(
                derivatives_folder
                / f"{fname_stem}_step8_responselocked-{BC_response}-ave.fif",
            )[0]

        epochs_CSD = compute_current_source_density(
            epochs.interpolate_bads(), stiffness=3, n_legendre_terms=15, verbose=100
        )
        evoked_CSD = compute_current_source_density(
            evoked.interpolate_bads(), stiffness=3, n_legendre_terms=15, verbose=100
        )

        # save epochs, drop-log and evoked files
        if epochs_CSD is not None and evoked_CSD is not None:
            epochs_CSD.save(
                derivatives_folder
                / f"{fname_stem}_step8b_responselocked-{BC_response}_CSD-cleaned-epo.fif"  # noqa: E501
            )
            evoked_CSD.save(
                derivatives_folder
                / f"{fname_stem}_step8b_responselocked-{BC_response}_CSD-ave.fif"
            )

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


@fill_doc
def create_epochs_evoked_and_behavioral_metadata(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
) -> None:
    """Compute behavioral report and output metadata.

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
    fname_stem = get_fname(participant, group, task, run)
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )

    # lock the output derivative files
    # create locks
    derivatives = [
        derivatives_folder / f"{fname_stem}_step8_stimlocked-cleaned-epo.fif",
        derivatives_folder
        / "plots"
        / f"{fname_stem}_step8_stimlocked-epochs-rejected.svg",
        derivatives_folder / f"{fname_stem}_step8_stimlocked-cleaned-epo-drop-log.csv",
        derivatives_folder / f"{fname_stem}_step8_stimlocked-standard-ave.fif",
        derivatives_folder / f"{fname_stem}_step8_stimlocked-target-ave.fif",
        derivatives_folder / f"{fname_stem}_step8_stimlocked-novel-ave.fif",
        # derivatives_folder / f"{fname_stem}_step8_responselocked-cleaned-epo.fif",
        # derivatives_folder
        # / "plots"
        # / f"{fname_stem}_step8_responselocked-epochs-rejected.svg",
        # derivatives_folder
        # / f"{fname_stem}_step8_responselocked-cleaned-epo-drop-log.csv",
        # derivatives_folder / f"{fname_stem}_step8_responselocked-ave.fif",
    ]

    # window_response_bc = ["2000resp_freq","400200resp","2000resp",]
    # for window in window_response_bc:
    #     derivatives.extend([derivatives_folder / f"{fname_stem}_step8_responselocked-{window}-cleaned-epo.fif",  # noqa: E501
    #     derivatives_folder
    #     / "plots"
    #     / f"{fname_stem}_step8_responselocked-{window}-epochs-rejected.svg",
    #     derivatives_folder
    #     / f"{fname_stem}_step8_responselocked-{window}-cleaned-epo-drop-log.csv",
    #     derivatives_folder / f"{fname_stem}_step8_responselocked-{window}-ave.fif",
    # ])

    locks = lock_files(*derivatives, timeout=timeout)

    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError

        # load previous steps (raw_fit recording)
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif",
            preload=True,
        )
        # prepare epoch and behavioral data
        (
            epochs,
            count_stim_before,
            count_stim_after,
            drop_reasons,
            fig_drops,
            evokeds,
            # epochs_response,
            # evoked_response,
            # drop_reasons_response,
            # fig_drops_response,
            # count_stim_before_response,
            # count_stim_after_response,
            # epochs_response_400200resp,
            # evoked_response_400200resp,
            # drop_reasons_response_400200resp,
            # fig_drops_response_400200resp,
            # count_stim_before_response_400200resp,
            # count_stim_after_response_400200resp,
            # epochs_response_2000resp,
            # evoked_response_2000resp,
            # drop_reasons_response_2000resp,
            # fig_drops_response_2000resp,
            # count_stim_before_response_2000resp,
            # count_stim_after_response_2000resp,
            # epochs_response_2000resp_freq,
            # evoked_response_2000resp_freq,
            # drop_reasons_response_2000resp_freq,
            # fig_drops_response_2000resp_freq,
            # count_stim_before_response_2000resp_freq,
            # count_stim_after_response_2000resp_freq,
        ) = _create_epochs_evoked_and_behavioral_metadata(raw)

        # save epochs, drop-log and evoked files
        epochs.save(
            derivatives_folder / f"{fname_stem}_step8_stimlocked-cleaned-epo.fif"
        )
        fig_drops.get_axes()[0].set_title(
            f"{fname_stem}: {fig_drops.get_axes()[0].get_title()}"
        )
        fig_drops.savefig(
            derivatives_folder
            / "plots"
            / f"{fname_stem}_step8_stimlocked-epochs-rejected.svg",
            transparent=True,
        )
        for cond in epochs.event_id:
            evokeds[cond].save(
                derivatives_folder / f"{fname_stem}_step8_stimlocked-{cond}-ave.fif"
            )
        events_mapping = load_triggers()
        with open(
            derivatives_folder
            / f"{fname_stem}_step8_stimlocked-cleaned-epo-drop-log.csv",
            "w",
        ) as file:
            file.write(
                ",Total,Rejected,Bad,PTP,After response,Too short\n"
                f"Standard,{count_stim_before[events_mapping['standard']]},{count_stim_before[events_mapping['standard']] - count_stim_after[events_mapping['standard']]},{drop_reasons['standard']['bad_segment']},{drop_reasons['standard']['ptp']},{drop_reasons['standard']['after_response']},{drop_reasons['standard']['too_short']}\n"  # noqa: E501
                f"Target,{count_stim_before[events_mapping['target']]},{count_stim_before[events_mapping['target']] - count_stim_after[events_mapping['target']]},{drop_reasons['target']['bad_segment']},{drop_reasons['target']['ptp']},{drop_reasons['target']['after_response']},{drop_reasons['target']['too_short']}\n"  # noqa: E501
                f"Novel,{count_stim_before[events_mapping['novel']]},{count_stim_before[events_mapping['novel']] - count_stim_after[events_mapping['novel']]},{drop_reasons['novel']['bad_segment']},{drop_reasons['novel']['ptp']},{drop_reasons['novel']['after_response']},{drop_reasons['novel']['too_short']}\n"  # noqa: E501
            )

        # if epochs_response is not None and evoked_response is not None:
        #     epochs_response.save(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-cleaned-epo.fif"
        #     )
        #     evoked_response.save(
        #         derivatives_folder / f"{fname_stem}_step8_responselocked-ave.fif"
        #     )
        #     fig_drops_response.get_axes()[0].set_title(
        #         f"{fname_stem}: {fig_drops_response.get_axes()[0].get_title()}"
        #     )
        #     fig_drops_response.savefig(
        #         derivatives_folder
        #         / "plots"
        #         / f"{fname_stem}_step8_responselocked-epochs-rejected.svg",
        #         transparent=True,
        #     )
        #     with open(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-cleaned-epo-drop-log.csv",
        #         "w",
        #     ) as file:
        #         file.write(
        #             ",Total,Rejected,Bad,PTP\n"
        #             f"Response,{count_stim_before_response[64]},{count_stim_before_response[64] - count_stim_after_response[64]},{drop_reasons_response['response']['bad_segment']},{drop_reasons_response['response']['ptp']}\n"  # noqa: E501
        #         )

        # if epochs_response_400200resp is not None and evoked_response_400200resp is not None:  # noqa: E501
        #     epochs_response_400200resp.save(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-400200resp-cleaned-epo.fif"
        #     )
        #     evoked_response_400200resp.save(
        #         derivatives_folder / f"{fname_stem}_step8_responselocked-400200resp-ave.fif"  # noqa: E501
        #     )
        #     fig_drops_response_400200resp.get_axes()[0].set_title(
        #         f"{fname_stem}: {fig_drops_response_400200resp.get_axes()[0].get_title()}"  # noqa: E501
        #     )
        #     fig_drops_response_400200resp.savefig(
        #         derivatives_folder
        #         / "plots"
        #         / f"{fname_stem}_step8_responselocked-400200resp-epochs-rejected.svg",
        #         transparent=True,
        #     )
        #     with open(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-400200resp-cleaned-epo-drop-log.csv",  # noqa: E501
        #         "w",
        #     ) as file:
        #         file.write(
        #             ",Total,Rejected,Bad,PTP\n"
        #             f"Response,{count_stim_before_response_400200resp[64]},{count_stim_before_response_400200resp[64] - count_stim_after_response_400200resp[64]},{drop_reasons_response_400200resp['response']['bad_segment']},{drop_reasons_response_400200resp['response']['ptp']}\n"  # noqa: E501
        #          )
        # if epochs_response_2000resp is not None and evoked_response_2000resp is not None:  # noqa: E501
        #     epochs_response_2000resp.save(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-2000resp-cleaned-epo.fif"
        #     )
        #     evoked_response_2000resp.save(
        #         derivatives_folder / f"{fname_stem}_step8_responselocked-2000resp-ave.fif"  # noqa: E501
        #     )
        #     fig_drops_response_2000resp.get_axes()[0].set_title(
        #         f"{fname_stem}: {fig_drops_response_2000resp.get_axes()[0].get_title()}"  # noqa: E501
        #     )
        #     fig_drops_response_2000resp.savefig(
        #         derivatives_folder
        #         / "plots"
        #         / f"{fname_stem}_step8_responselocked-2000resp-epochs-rejected.svg",
        #         transparent=True,
        #     )
        #     with open(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-2000resp-cleaned-epo-drop-log.csv",  # noqa: E501
        #         "w",
        #     ) as file:
        #         file.write(
        #             ",Total,Rejected,Bad,PTP\n"
        #             f"Response,{count_stim_before_response_2000resp[64]},{count_stim_before_response_2000resp[64] - count_stim_after_response_2000resp[64]},{drop_reasons_response_2000resp['response']['bad_segment']},{drop_reasons_response_2000resp['response']['ptp']}\n"  # noqa: E501
        #         )
        # if epochs_response_2000resp_freq is not None and evoked_response_2000resp_freq is not None:  # noqa: E501
        #     epochs_response_2000resp_freq.save(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-2000resp_freq-cleaned-epo.fif"
        #     )
        #     evoked_response_2000resp_freq.save(
        #         derivatives_folder / f"{fname_stem}_step8_responselocked-2000resp_freq-ave.fif"  # noqa: E501
        #     )
        #     fig_drops_response_2000resp_freq.get_axes()[0].set_title(
        #         f"{fname_stem}: {fig_drops_response_2000resp_freq.get_axes()[0].get_title()}"  # noqa: E501
        #     )
        #     fig_drops_response_2000resp_freq.savefig(
        #         derivatives_folder
        #         / "plots"
        #         / f"{fname_stem}_step8_responselocked-2000resp_freq-epochs-rejected.svg",  # noqa: E501
        #         transparent=True,
        #     )
        #     with open(
        #         derivatives_folder
        #         / f"{fname_stem}_step8_responselocked-2000resp_freq-cleaned-epo-drop-log.csv",  # noqa: E501
        #         "w",
        #     ) as file:
        #         file.write(
        #             ",Total,Rejected,Bad,PTP\n"
        #             f"Response,{count_stim_before_response_2000resp_freq[64]},{count_stim_before_response_2000resp_freq[64] - count_stim_after_response_2000resp_freq[64]},{drop_reasons_response_2000resp_freq['response']['bad_segment']},{drop_reasons_response_2000resp_freq['response']['ptp']}\n"  # noqa: E501
        #         )

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


def _create_epochs_evoked_and_behavioral_metadata(
    raw: BaseRaw,
) -> tuple[
    Epochs,
    Counter,
    Counter,
    dict[str, dict[str, int]],
    plt.Figure,
    dict[str, Evoked],
    Optional[Epochs],
    # Optional[Evoked],
    # dict[str, dict[str, int]],
    # plt.Figure,
    # Counter,
    # Counter,
    # Optional[Epochs],
    # Optional[Evoked],
    # dict[str, dict[str, int]],
    # plt.Figure,
    # Counter,
    # Counter,
    # Optional[Epochs],
    # Optional[Evoked],
    # dict[str, dict[str, int]],
    # plt.Figure,
    # Counter,
    # Counter,
]:
    """Prepare epochs from a raw object."""
    events = find_events(raw, stim_channel="TRIGGER")
    events_id = load_triggers()
    if np.any(events[:, 2] == 64):
        mask1 = np.where(events[:, 2] == 2)[0]
        sel = np.array(
            [
                elt
                for elt in np.where(events[:, 2] == 64)[0]
                if elt - 1 in mask1
                and events[elt, 0] - events[elt - 1, 0] >= _TOO_QUICK_THRESHOLD * raw.info["sfreq"]  # noqa: E501
            ]
        )  # fmt: skip
        events_response = events[sel]
        events_id["response"] = 64
    if sorted(np.unique(events[:, 2])) != sorted(events_id.values()):
        warn(
            f"The events array contains unexpected triggers: {np.unique(events[:, 2])}",
            RuntimeWarning,
            stacklevel=2,
        )
    if "response" in events_id:
        metadata, events, events_id = _make_metadata(events, events_id, raw)
        metadata.drop(columns=["standard", "target", "novel"], inplace=True)
        logger.info("Creating response-lock epochs.")
        response_time = metadata["response"].values[
            metadata["response_type"].values == "Hits"
        ]
        assert response_time.size == events_response.shape[0]  # sanity-check
        metadata_reponse = dict(
            event_name=["response"] * events_response.shape[0],
            response_time=response_time,
        )
        metadata_reponse = pd.DataFrame.from_dict(metadata_reponse)
        epochs_response = Epochs(
            raw=raw,
            tmin=-0.2,
            tmax=0.8,
            events=events_response,
            event_id=dict(response=64),
            metadata=metadata_reponse,
            reject=None,
            preload=True,
            baseline=None,  # manual baseline
            picks="eeg",
        )
        # at this point, the only epochs dropped are due to bad segments, let's now
        # apply the mean baseline correction using the stimulus-locked baseline period.
        distance = np.full(
            (events[:, 0].size, epochs_response.events[:, 0].size),
            epochs_response.events[:, 0],
            dtype=float,
        )
        distance = distance.T - events[:, 0]  # (n_stimuli, n_response)
        distance[np.where(distance <= 0)] = np.nan
        idx = np.nanargmin(distance, axis=1)
        tf = events[idx, 0] - raw.first_samp  # stimuli onset of the preceding stimulus
        delta = int(np.round(raw.info["sfreq"] * 0.2))  # 200 ms before stimulus onset
        t0 = tf - delta
        baseline = np.zeros(
            (epochs_response._data.shape[0], epochs_response._data.shape[1], delta)
        )
        eegs = _picks_to_idx(raw.info, "eeg", exclude=())
        assert eegs.size == epochs_response._data.shape[1]  # sanity-check
        for k, (t1, t2) in enumerate(zip(t0, tf, strict=True)):
            baseline[k, :] = raw[eegs, t1:t2][0]
        baseline = np.mean(baseline, axis=-1, keepdims=True)  # average in time
        epochs_response._data -= baseline
        # and now that the baseline correction is applied, we can reject bad epochs
        reject = _get_rejection(epochs_response)
        (
            epochs_response,
            count_stim_before_response,
            count_stim_after_response,
            drop_reasons_response,
            fig_drops_response,
        ) = _drop_bad_epochs(epochs_response, events_response, reject, response=True)

        # redo for other baseline correction
        epochs_response_400200resp = Epochs(
            raw=raw,
            tmin=-0.4,
            tmax=0.8,
            events=events_response,
            event_id=dict(response=64),
            metadata=metadata_reponse,
            reject=None,
            preload=True,
            baseline=(-0.4, -0.2),  # manual baseline
            picks="eeg",
        )
        reject = _get_rejection(epochs_response_400200resp)
        (
            epochs_response_400200resp,
            count_stim_before_response_400200resp,
            count_stim_after_response_400200resp,
            drop_reasons_response_400200resp,
            fig_drops_response_400200resp,
        ) = _drop_bad_epochs(
            epochs_response_400200resp, events_response, reject, response=True
        )

        # redo for other baseline correction
        epochs_response_2000resp = Epochs(
            raw=raw,
            tmin=-0.2,
            tmax=0.8,
            events=events_response,
            event_id=dict(response=64),
            metadata=metadata_reponse,
            reject=None,
            preload=True,
            baseline=(-0.2, 0),  # manual baseline
            picks="eeg",
        )
        reject = _get_rejection(epochs_response_2000resp)
        (
            epochs_response_2000resp,
            count_stim_before_response_2000resp,
            count_stim_after_response_2000resp,
            drop_reasons_response_2000resp,
            fig_drops_response_2000resp,
        ) = _drop_bad_epochs(
            epochs_response_2000resp, events_response, reject, response=True
        )

        # redo for other baseline correction
        epochs_response_2000resp_freq = Epochs(
            raw=raw,
            tmin=-0.2,
            tmax=0.250,
            events=events_response,
            event_id=dict(response=64),
            metadata=metadata_reponse,
            reject=None,
            preload=True,
            baseline=(-0.2, 0),  # manual baseline
            picks="eeg",
        )
        reject = _get_rejection(epochs_response_2000resp_freq)
        (
            epochs_response_2000resp_freq,
            count_stim_before_response_2000resp_freq,
            count_stim_after_response_2000resp_freq,
            drop_reasons_response_2000resp_freq,
            fig_drops_response_2000resp_freq,
        ) = _drop_bad_epochs(
            epochs_response_2000resp_freq, events_response, reject, response=True
        )

    else:
        metadata = None
        epochs_response = None

    epochs = Epochs(
        raw=raw,
        tmin=-0.2,
        tmax=0.8,
        events=events,
        event_id=events_id,
        metadata=metadata,
        reject=None,
        preload=True,
        baseline=(None, 0),
        picks="eeg",
    )
    reject = _get_rejection(epochs)
    (
        epochs,
        count_stim_before,
        count_stim_after,
        drop_reasons,
        fig_drops,
    ) = _drop_bad_epochs(epochs, events, reject, response=False)

    if metadata is None:
        evokeds = dict((cond, epochs[cond].average()) for cond in epochs.event_id)
    else:
        evokeds = dict(
            (cond, epochs["response_correct == True"][cond].average())
            for cond in epochs.event_id
        )
    # Assuming `metadata`, `epochs`, and `event_id` are defined as in your original code

    return (
        epochs,
        count_stim_before,
        count_stim_after,
        drop_reasons,
        fig_drops,
        evokeds,
        # epochs_response,
        # None if epochs_response is None else epochs_response.average(),
        # drop_reasons_response,
        # fig_drops_response,
        # count_stim_before_response,
        # count_stim_after_response,
        # epochs_response_400200resp,
        # None if epochs_response_400200resp is None else epochs_response_400200resp.average(),  # noqa: E501
        # drop_reasons_response_400200resp,
        # fig_drops_response_400200resp,
        # count_stim_before_response_400200resp,
        # count_stim_after_response_400200resp,
        # epochs_response_2000resp,
        # None if epochs_response_2000resp is None else epochs_response_2000resp.average(),  # noqa: E501
        # drop_reasons_response_2000resp,
        # fig_drops_response_2000resp,
        # count_stim_before_response_2000resp,
        # count_stim_after_response_2000resp,
        # epochs_response_2000resp_freq,
        # None if epochs_response_2000resp_freq is None else epochs_response_2000resp_freq.average(),  # noqa: E501
        # drop_reasons_response_2000resp_freq,
        # fig_drops_response_2000resp_freq,
        # count_stim_before_response_2000resp_freq,
        # count_stim_after_response_2000resp_freq,
    )


def _make_metadata(
    events: NDArray[+ScalarIntType],
    events_id: dict[str, int],
    raw: BaseRaw,
) -> tuple[pd.DataFrame, NDArray[+ScalarIntType], dict[str, int]]:
    """Create metadata from events for each epoch.

    For each epoch, it shall include events from the range: [0.0, 1.5] s,
    i.e. starting with stimulus onset and expanding beyond the end of the epoch.
    Currently it includes [0.0, 0.999].

    Parameters
    ----------
    events : array of shape (n_events, 3)
    events_id : dict
    raw : Raw

    Returns
    -------
    metadata : DataFrame
        Metadata of the events.
    events : array of shape (n_events, 3)
        The events.
    event_id : dict
        Mapping str to int of the events
    """
    # MNE auto-generate metadata, which also returns a new events array and an event_id
    # dictionary. The metadata are created from response-locked events around target,
    # standard and novel stimuli; and include the events target, standard, novel and
    # response.
    metadata, events, event_id = make_metadata_mne(
        events=events,
        event_id=events_id,
        tmin=0.0,
        tmax=0.999,
        sfreq=raw.info["sfreq"],
        row_events=["standard", "target", "novel"],
    )
    # fmt: off
    conditions = [
        (metadata["event_name"].eq("target")) & (pd.notna(metadata["response"])) & (metadata["response"] < _TOO_QUICK_THRESHOLD),  # noqa: E501
        (metadata["event_name"].eq("target")) & (pd.notna(metadata["response"])) & (metadata["response"] >= _TOO_QUICK_THRESHOLD),  # noqa: E501
        (metadata["event_name"].eq("target")) & (pd.isna(metadata["response"])),
        (metadata["event_name"].eq("standard")) & (pd.notna(metadata["response"])),
        (metadata["event_name"].eq("standard")) & (pd.isna(metadata["response"])),
        (metadata["event_name"].eq("novel")) & (pd.notna(metadata["response"])),
        (metadata["event_name"].eq("novel")) & (pd.isna(metadata["response"])),
    ]
    choices = [
        "FalseAlarms_tooquick",
        "Hits",
        "Misses",
        "FalseAlarms-standard",
        "CorrectRejections-standard",
        "FalseAlarms-novel",
        "CorrectRejections-novel",
    ]
    assert len(conditions) == len(choices)  # sanity-check
    metadata["response_type"] = np.select(conditions, choices, default=0)
    metadata["response_correct"] = (
        (metadata["response_type"] == "Hits")
        | (metadata["response_type"] == "CorrectRejections-standard")
        | (metadata["response_type"] == "CorrectRejections-novel")
    )
    # fmt: on
    return metadata, events, event_id


def _get_rejection(epochs: BaseEpochs) -> dict[str, float]:
    """Epoching.

    Parameters
    ----------
    epochs : Epochs

    Returns
    -------
    reject : dict
        The rejection dictionary with keys as specified by ch_types.
    """
    starttime = time.time()
    reject = get_rejection_threshold(epochs, decim=1, ch_types="eeg", random_state=888)
    endtime = time.time()
    diff = endtime - starttime
    minutes = str(int(diff // 60)).zfill(2)
    seconds = str(int(diff % 60)).zfill(2)
    logger.info("\nPeak-to-peak rejection threshold computed: %s", reject)
    logger.info(f"Elapsed {minutes}:{seconds}\n")
    return reject


def _drop_bad_epochs(
    epochs: BaseEpochs, events: NDArray, reject: dict[str, float], response: bool
) -> tuple[BaseEpochs, Counter, Counter, dict[str, dict[str, int]], plt.Figure]:
    """Clean epochs from autoreject value."""
    count_stim_before = Counter(epochs.events[:, 2])
    if epochs.metadata is not None and not response:
        # drop epochs following a response
        idx_to_drop = np.where(~np.isnan(epochs.metadata["response"].values))[0] + 1
        idx_to_drop = idx_to_drop[np.where(idx_to_drop <= len(epochs))]
        epochs.drop(idx_to_drop, reason="epoch after response")
    epochs.drop_bad(reject=reject)
    count_stim_after = Counter(epochs.events[:, 2])
    # log dropped epochs
    events_mapping = {value: key for key, value in epochs.event_id.items()}
    events_mapping["response"] = 64
    if response:
        drop_reasons = dict(response=dict(bad_segment=0, ptp=0, too_short=0))
    else:
        drop_reasons = dict(
            standard=dict(bad_segment=0, ptp=0, too_short=0, after_response=0),
            target=dict(bad_segment=0, ptp=0, too_short=0, after_response=0),
            novel=dict(bad_segment=0, ptp=0, too_short=0, after_response=0),
        )
    for ev, drops in zip(events, epochs.drop_log):
        if len(drops) == 0:
            continue
        event_type = events_mapping[ev[2]]
        if all(elt in epochs.ch_names for elt in drops):
            drop_reasons[event_type]["ptp"] += 1
        elif any(elt.lower().startswith("bad") for elt in drops):
            drop_reasons[event_type]["bad_segment"] += 1
        elif any(elt == "epoch after response" for elt in drops) and not response:
            drop_reasons[event_type]["after_response"] += 1
        elif any(elt == "TOO_SHORT" for elt in drops):
            drop_reasons[event_type]["too_short"] += 1
        else:
            raise ValueError(f"Unknown drop reason: {drops}")
    fig = epochs.plot_drop_log()
    return epochs, count_stim_before, count_stim_after, drop_reasons, fig
