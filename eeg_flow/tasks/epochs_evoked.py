from __future__ import annotations  # c.f. PEP 563, PEP 649

import math
import time
from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import pandas as pd
from autoreject import get_rejection_threshold
from mne import Epochs, find_events
from mne.epochs import make_metadata as make_metadata_mne
from mne.io import read_raw_fif
from scipy.stats import norm

from .. import logger
from ..config import load_config, load_triggers
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files

if TYPE_CHECKING:
    from matplotlib import pyplot as plt
    from mne import Evoked
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw
    from numpy.typing import DTypeLike, NDArray

    ScalarIntType: tuple[DTypeLike, ...] = (np.int8, np.int16, np.int32, np.int64)


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
        derivatives_folder / f"{fname_stem}_step8_c1-cleaned-epo.fif",
        derivatives_folder / f"{fname_stem}_step8_c2-drop-epochs-per-stim.csv",
        derivatives_folder / f"{fname_stem}_step8_c3-drop-epochs-per-reason.csv",
        derivatives_folder / f"{fname_stem}_step8_c4-dropped-total.csv",
        derivatives_folder / f"{fname_stem}_step8-standard-ave.fif",
        derivatives_folder / f"{fname_stem}_step8-target-ave.fif",
        derivatives_folder / f"{fname_stem}_step8-novel-ave.fif",
    ]
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
            metadata,
            epochs,
            count_stim_before,
            df_drops,
            df_total_drops,
            df_total_remaining,
            fig_drops,
            evokeds,
        ) = _create_epochs_evoked_and_behavioral_metadata(raw)

        # save epochs, drop-log and evoked files
        epochs.save(derivatives_folder / f"{fname_stem}_step8_c1-cleaned-epo.fif")
        df_counts = _count_stim_dropped(count_stim_before, epochs)
        df_counts.to_csv(derivatives_folder / f"{fname_stem}_step8_c2-drop-epochs-per-stim.csv")
        fig_drops.get_axes()[0].set_title(
            f"{fname_stem}: {fig_drops.get_axes()[0].get_title()}"
        )
        fig_drops.savefig(
            derivatives_folder / "plots" / f"{fname_stem}_step8_epochs-rejected.svg",
            transparent=True,
        )
        df_drops = df_drops.rename(columns={0: fname_stem})
        df_drops.to_csv(
            derivatives_folder / f"{fname_stem}_step8_c3-drop-epochs-per-reason.csv"
        )
        df_total_drops.to_csv(
            derivatives_folder / f"{fname_stem}_step8_c4-dropped-total.csv"
        )
        df_total_remaining.to_csv(
            derivatives_folder / f"{fname_stem}_step8_c5-total-remaining.csv"
        )


        for cond in epochs.event_id:
            evokeds[cond].save(
                derivatives_folder / f"{fname_stem}_step8_{cond}-ave.fif"
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


def _create_epochs_evoked_and_behavioral_metadata(
    raw: BaseRaw,
) -> tuple[
    pd.DataFrame,
    Epochs,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    plt.Figure,
    dict[str, Evoked],
]:
    """Prepare epochs from a raw object."""
    events = find_events(raw, stim_channel="TRIGGER")
    events_id = load_triggers()
    if np.any(events[:, 2] == 64):
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
      
    else:
        metadata = None

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
    epochs, count_stim_before, df_drops, df_total_drops, df_total_remaining, fig_drops = _drop_bad_epochs(epochs, reject)
    if metadata is None:
        evokeds = dict((cond, epochs[cond].average()) for cond in epochs.event_id)
    else:
        evokeds = dict(
            (cond, epochs["response_correct == True"][cond].average())
            for cond in epochs.event_id
        )
    return (
        metadata,
        epochs,
        count_stim_before,
        df_drops,
        df_total_drops,
        df_total_remaining,
        fig_drops,
        evokeds,
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
    conditions = [
        (metadata["event_name"].eq("target"))
        & (pd.notna(metadata["response"]))
        & (metadata["response"] < 0.2),
        (metadata["event_name"].eq("target"))
        & (pd.notna(metadata["response"]))
        & (metadata["response"] >= 0.2),
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
        "FalseAlarms",
        "CorrectRejections",
        "FalseAlarms",
        "CorrectRejections",
    ]
    metadata["response_type"] = np.select(conditions, choices, default=0)
    metadata["response_type"].value_counts()
    metadata["response_correct"] = (metadata["response_type"] == "CorrectRejections") | (metadata["response_type"] == "Hits")
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
    epochs: BaseEpochs, reject: dict[str, float]
) -> tuple[BaseEpochs, pd.DataFrame, pd.DataFrame, plt.Figure]:
    """Clean epochs from autoreject value.

    Parameters
    ----------
    epochs : Epochs
    df_counts : DataFrame
    df_drops : DataFrame
    fig_drops : Figure

    Returns
    -------
    epochs : Epochs
        Cleaned epochs, where epochs with supra-threshold PTP amplitude are dropped and
        where epochs following a response are dropped.
    """
    count_stim_before = Counter(epochs.events[:, 2])
    epochs.drop_bad(reject=reject)
    if epochs.metadata is not None:
        # drop epochs following a response
        response_arr = pd.notna(epochs.metadata["response"]).to_numpy()
        idx_to_drop = np.where(response_arr)[0] + 1
        for item in idx_to_drop:
            if item >= len(epochs):
                idx_to_drop = np.delete(idx_to_drop, np.where(idx_to_drop == item))
        epochs.drop(idx_to_drop, reason="epoch after response")
    # log dropped epochs
    totals = Counter(chain(*epochs.drop_log))
    df_drops = pd.DataFrame.from_dict(totals, orient="index")
    df_drops = df_drops.sort_values(by=[0], ascending=False)
    fig = epochs.plot_drop_log()
    df_total_drops = _log_total_drop(fig.get_axes()[0].get_title())
    df_total_remaining = _total_per_stim(epochs)
    return epochs, count_stim_before, df_drops, df_total_drops, df_total_remaining, fig

def _total_per_stim(epochs):
    """Return a dataframe with the count of remaining stims
    
    Parameters
    ----------
    epochs : BaseEpoch

    Returns
    ----------
    df_total_stim : DataFrame
        DataFrame with the relevant dropped epochs infos
    """
    unique, counts = np.unique(epochs.events, return_counts=True)
    all_counts = dict(zip(unique, counts))
    all_counts

    all_count_stim = {key: all_counts[key] for key in [1,2,3]}
    all_count_stim

    df_total_stim = pd.DataFrame(all_count_stim, index=[0])
    return df_total_stim


def _log_total_drop(drop_info):
    """Return a dataframe with the key infos dropped epochs
    
    Parameters
    ----------
    drop_info : str

    Returns
    ----------
    df_total_drops : DataFrame
        DataFrame with the relevant dropped epochs infos
    """
    temp = drop_info.split(" ")
    df_total_drops = pd.DataFrame(columns=["n_dropped", "n_original", "percent_dropped"])
    df_total_drops.loc[0] = [temp[0],temp[2],temp[5]]
    return df_total_drops

def _count_stim_dropped(count_stim_before, epochs):
    """Return a DataFrame with the number of epochs dropped per stimulus, for any reasons.

    Parameters
    ----------
    count_stim_before : list of int
    epochs : Epochs

    Returns
    -------
    out : DataFrame
        Total count of dropped epochs per stimulus.
    """
    count_stim_after = Counter(epochs.events[:, 2])
    data = [
        ["1", count_stim_before[1] - count_stim_after[1]],
        ["2", count_stim_before[2] - count_stim_after[2]],
        ["3", count_stim_before[3] - count_stim_after[3]],
    ]
    return pd.DataFrame(data, columns=["Stim", "n_dropped"])
