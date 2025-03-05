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



@fill_doc
def create_epochs_evoked_and_behavioral_metadata_response(
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
        derivatives_folder / f"{fname_stem}_step11_responselocked-interp-cleaned-epo.fif",
        derivatives_folder
        / "plots"
        / f"{fname_stem}_step11_responselocked-interp-epochs-rejected.svg",
        derivatives_folder
        / f"{fname_stem}_step11_responselocked-interp-cleaned-epo-drop-log.csv",
        derivatives_folder / f"{fname_stem}_step11_responselocked-interp-ave.fif",
    ]

    locks = lock_files(*derivatives, timeout=timeout)

    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError

        # load previous steps (raw_fit recording)
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step10_preprocessed_raw.fif",
            preload=True,
        )
        # prepare epoch and behavioral data
        (

            epochs_response,
            evoked_response,
            drop_reasons_response,
            fig_drops_response,
            count_stim_before_response,
            count_stim_after_response,
           
        ) = _create_epochs_evoked_and_behavioral_metadata_response(raw)

        # save epochs, drop-log and evoked files

        if epochs_response is not None and evoked_response is not None:
            epochs_response.save(
                derivatives_folder
                / f"{fname_stem}_step11_responselocked-interp-cleaned-epo.fif"
            )
            evoked_response.save(
                derivatives_folder / f"{fname_stem}_step11_responselocked-interp-ave.fif"
            )
            fig_drops_response.get_axes()[0].set_title(
                f"{fname_stem}: {fig_drops_response.get_axes()[0].get_title()}"
            )
            fig_drops_response.savefig(
                derivatives_folder
                / "plots"
                / f"{fname_stem}_step11_responselocked-interp-epochs-rejected.svg",
                transparent=True,
            )
            with open(
                derivatives_folder
                / f"{fname_stem}_step11_responselocked-interp-cleaned-epo-drop-log.csv",
                "w",
            ) as file:
                file.write(
                    ",Total,Rejected,Bad,PTP\n"
                    f"Response,{count_stim_before_response[64]},{count_stim_before_response[64] - count_stim_after_response[64]},{drop_reasons_response['response']['bad_segment']},{drop_reasons_response['response']['ptp']}\n"  # noqa: E501
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


def _create_epochs_evoked_and_behavioral_metadata_response(
    raw: BaseRaw,
) -> tuple[
    Epochs,
    Counter,
    Counter,
    dict[str, dict[str, int]],
    plt.Figure,
    # dict[str, Evoked],
    # Optional[Epochs],
    
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
            "The events array contains unexpected triggers: "
            f"{np.unique(events[:, 2])}",
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




        # redo for other baseline correction
        epochs_response = Epochs(
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
        reject = _get_rejection(epochs_response)
        (
            epochs_response,
            count_stim_before_response,
            count_stim_after_response,
            drop_reasons_response,
            fig_drops_response,
        ) = _drop_bad_epochs(
            epochs_response, events_response, reject, response=True
        )

   
    # Assuming `metadata`, `epochs`, and `event_id` are defined as in your original code

    return (

        epochs_response,
        None if epochs_response is None else epochs_response.average(),
        drop_reasons_response,
        fig_drops_response,
        count_stim_before_response,
        count_stim_after_response,
       
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
