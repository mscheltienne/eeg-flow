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
        derivatives_folder / f"{fname_stem}_step8_a-metadata.csv",
        derivatives_folder / f"{fname_stem}_step8_b-behav.txt",
        derivatives_folder / f"{fname_stem}_step8_c1-cleaned-epo.fif",
        derivatives_folder / f"{fname_stem}_step8_c2-drop-epochs.csv",
        derivatives_folder / f"{fname_stem}_step8_c3-drop-channel-log.csv",
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
            fig_rt,
            behavioral_str,
            epochs,
            df_counts,
            df_drops,
            fig_drops,
            evokeds,
        ) = _create_epochs_evoked_and_behavioral_metadata(raw)

        # save metadata, response times and behavioral data
        if metadata is not None:
            assert fig_rt is not None
            assert behavioral_str is not None
            metadata.to_csv(derivatives_folder / f"{fname_stem}_step8_a-metadata.csv")
            fig_rt.suptitle(fname_stem, fontsize=16, y=1)
            fig_rt.savefig(
                derivatives_folder / "plots" / f"{fname_stem}_step8_RT.svg",
                transparent=True,
            )
            with open(
                derivatives_folder / f"{fname_stem}_step8_b-behav.txt", "w"
            ) as file:
                file.write(behavioral_str)

        # save epochs, drop-log and evoked files
        epochs.save(derivatives_folder / f"{fname_stem}_step8_c1-cleaned-epo.fif")
        df_counts.to_csv(derivatives_folder / f"{fname_stem}_step8_c2-drop-epochs.csv")
        fig_drops.get_axes()[0].set_title(
            f"{fname_stem}: {fig_drops.get_axes()[0].get_title()}"
        )
        df_drops = df_drops.rename(columns={0: fname_stem})
        df_drops.to_csv(
            derivatives_folder / f"{fname_stem}_step8_c3-drop-channel-log.csv"
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
    plt.Figure,
    str,
    Epochs,
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
        n_hits, n_correct_rejections, n_misses, n_false_alarms = _get_SDT_outcomes(
            metadata
        )
        hits = metadata[metadata["response_type"] == "Hits"]
        response_mean = round(hits["response"].mean(), 5)
        response_std = round(hits["response"].std(), 5)
        fig_rt, _ = _plot_reaction_time(hits, response_mean, response_std)
        behavioral_str = _repr_individual_behavioral(
            metadata,
            n_hits,
            n_correct_rejections,
            n_misses,
            n_false_alarms,
            response_mean,
            response_std,
        )
    else:
        metadata = None
        fig_rt = None
        behavioral_str = None

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
    epochs, df_counts, df_drops, fig_drops = _drop_bad_epochs(epochs, reject)
    if metadata is None:
        evokeds = dict((cond, epochs[cond].average()) for cond in epochs.event_id)
    else:
        evokeds = dict(
            (cond, epochs["response_correct == True"][cond].average())
            for cond in epochs.event_id
        )
    return (
        metadata,
        fig_rt,
        behavioral_str,
        epochs,
        df_counts,
        df_drops,
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
    metadata.response_correct = False
    metadata.loc[
        (metadata["response_type"] == "CorrectRejections"), "response_correct"
    ] = True
    metadata.loc[(metadata["response_type"] == "Hits"), "response_correct"] = True
    metadata.loc[
        (metadata["response_type"] == "FalseAlarms"), "response_correct"
    ] = False
    metadata.loc[(metadata["response_type"] == "Misses"), "response_correct"] = False
    return metadata, events, event_id


def _get_SDT_outcomes(metadata: pd.DataFrame) -> tuple[int, int, int, int]:
    """Compute proportions for Signal Detection Theory (SDT).

    Parameters
    ----------
    %(metadata)s

    Returns
    -------
    n_hits : int
        Number of hits
    n_correct_rejections : int
        Number of correct rejections
    n_misses : int
        Number of misses
    n_false_alarms : int
        Number of false alarms
    """
    n_hits = len(metadata[metadata["response_type"] == "Hits"])
    n_correct_rejections = len(
        metadata[metadata["response_type"] == "CorrectRejections"]
    )
    n_misses = len(metadata[metadata["response_type"] == "Misses"])
    n_false_alarms = len(metadata[metadata["response_type"] == "FalseAlarms"])
    return n_hits, n_correct_rejections, n_misses, n_false_alarms


def _plot_reaction_time(
    hits: pd.Series, response_mean: float, response_std: float
) -> tuple[plt.Figure, plt.Axes]:
    """Plot histogram of response times.

    Parameters
    ----------
    hits : Series
    response_mean : float
    response_std : float

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    ax_rt = hits["response"].plot.hist(
        bins=100,
        title=f"Response Times of TPs\nmean:{str(response_mean)} ({str(response_std)})",
    )
    return ax_rt.figure, ax_rt


def _repr_individual_behavioral(
    metadata: pd.DataFrame,
    n_hits: int,
    n_correct_rejections: int,
    n_misses: int,
    n_false_alarms: int,
    response_mean: float,
    response_std: float,
) -> str:
    """Create a string representation of the individual behavioral information.

    Parameters
    ----------
    metadata : DataFrame
    n_hits : int
    n_correct_rejections : int
    n_misses : int
    n_false_alarms : int
    response_mean : float
    response_std : float

    Returns
    -------
    behavioral_str : str
        String representation of the metadata.
    """
    correct_response_count = metadata["response_correct"].sum()
    logger.info(
        f"\nCorrect responses: {correct_response_count}\n"
        f"Incorrect responses: {len(metadata) - correct_response_count}\n"
    )
    metadata_count_correct = metadata.groupby(by="event_name").count()[
        "response_correct"
    ]
    behavioral_str = (
        "Stats:"
        f"\n\tHits: {n_hits}\n\tMisses: {n_misses}"
        f"\n\tCorrect Rejections: {n_correct_rejections}"
        f"\n\tFalse Alarms: {n_false_alarms}\n\n"
        "Stims:"
        f"\n\tStandard: {str(metadata_count_correct['standard'])}"
        f"\n\tTarget: {str(metadata_count_correct['target'])}"
        f"\n\tNovel: {str(metadata_count_correct['novel'])}\n\n"
        "Response mean, Response std:"
        f"\n\t{response_mean}, {response_std}\n\n"
        "D':\n\t"
        f"{str(_SDT_loglinear(n_hits, n_misses, n_false_alarms, n_correct_rejections))}"
    )
    logger.info(behavioral_str)
    return behavioral_str


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
    count_stim_after = Counter(epochs.events[:, 2])
    data = [
        ["1", count_stim_before[1] - count_stim_after[1]],
        ["2", count_stim_before[2] - count_stim_after[2]],
        ["3", count_stim_before[3] - count_stim_after[3]],
    ]
    df_count = pd.DataFrame(data, columns=["Stim", "n_dropped"])
    if epochs.metadata is not None:
        # drop epochs following a response
        response_arr = pd.notna(epochs.metadata["response"]).to_numpy()
        epochs.drop(np.where(response_arr)[0] + 1, reason="epoch after response")
    # log dropped epochs
    totals = Counter(chain(*epochs.drop_log))
    df_drops = pd.DataFrame.from_dict(totals, orient="index")
    df_drops = df_drops.sort_values(by=[0], ascending=False)
    fig = epochs.plot_drop_log()
    return epochs, df_count, df_drops, fig


def _SDT_loglinear(hits: int, misses: int, fas: int, crs: int) -> dict[str, float]:
    """Return a dict with d-prime measures, corrected with the log-linear rule.

    See Stanislaw & Todorov 1999 and Hautus 1995,
    https://lindeloev.net/calculating-d-in-python-and-php/

    Parameters
    ----------
    hits : int
    misses : int
    fas : int
    crs : int

    Returns
    -------
    out : dict
        D' measures.
    """
    Z = norm.ppf
    # calculate hit_rate and avoid d' infinity
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    # calculate false alarm rate and avoid d' infinity
    fa_rate = (fas + 0.5) / (fas + crs + 1)
    # return d', beta, c and Ad'
    out = {}
    out["d"] = Z(hit_rate) - Z(fa_rate)
    out["beta"] = math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)
    out["c"] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out["Ad"] = norm.cdf(out["d"] / math.sqrt(2))
    return out
