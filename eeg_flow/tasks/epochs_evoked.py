# ########################################
# Modified on Mon May 08 01:01:00 2023
# @anguyen

from __future__ import annotations  # c.f. PEP 563, PEP 649

import itertools
import math
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from autoreject import get_rejection_threshold
from mne import Epochs, find_events
from mne.epochs import make_metadata as make_metadata_mne
from mne.io import read_raw_fif
from scipy.stats import norm

from .. import logger
from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files

if TYPE_CHECKING:
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw
    from numpy.typing import DTypeLike, NDArray

    ScalarIntType: tuple[DTypeLike, ...] = (np.int8, np.int16, np.int32, np.int64)


@fill_doc
def behav_prep_epoching(
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
        derivatives_folder / f"{fname_stem}_step8_d1-standard_evoked-ave.fif",
        derivatives_folder / f"{fname_stem}_step8_d2-target_evoked-ave.fif",
        derivatives_folder / f"{fname_stem}_step8_d3-novel_evoked-ave.fif",
    ]

    locks = lock_files(*derivatives, timeout=timeout)

    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        _behav_prep_epoching(participant, group, task, run)
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
def _behav_prep_epoching(
    participant: str,
    group: str,
    task: str,
    run: int,
) -> tuple[BaseRaw, NDArray[+ScalarIntType], dict[str, int], pd.DataFrame]:
    """Compute behavioral report and output metadata.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s

    Returns
    -------
    raw : Raw
        Raw data.
    events : array of shape (n_events, 3)
        The events.
    event_id : dict
        Mapping str to int of the events
    metadata : DataFrame
        Metadata of the events.
    """
    # prepare folders
    _, derivatives_folder_root, _ = load_config()
    fname_stem = get_fname(participant, group, task, run)
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )

    # load previous steps (raw_fit recording)
    raw = read_raw_fif(
        derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif",
        preload=True,
    )

    events = find_events(raw, stim_channel="TRIGGER")
    events_id = dict(standard=1, target=2, novel=3, response=64)
    row_events = ["standard", "target", "novel"]
    metadata, events, event_id = make_metadata(events, events_id, raw, row_events)
    fname_metadata = derivatives_folder / f"{fname_stem}_step8_a-metadata.csv"
    metadata.to_csv(fname_metadata)
    num_hits, num_correct_rejections, num_misses, num_false_alarms = get_SDT_outcomes(
        metadata
    )

    hits = metadata[metadata["response_type"] == "Hits"]
    response_mean = round(hits["response"].mean(), 5)
    response_std = round(hits["response"].std(), 5)
    plot_RT(hits, fname_stem, derivatives_folder, response_mean, response_std)

    get_indiv_behav(
        metadata,
        num_hits,
        num_correct_rejections,
        num_misses,
        num_false_alarms,
        fname_stem,
        derivatives_folder,
        response_mean,
        response_std,
    )

    epochs = epoching(raw, events, event_id, metadata)
    reject = get_rejection(epochs)
    epochs = clean_epochs_from_rejection(epochs, reject, fname_stem, derivatives_folder)
    fname_clean_epochs = derivatives_folder / f"{fname_stem}_step8_c1-cleaned-epo.fif"
    epochs.save(fname_clean_epochs)
    save_evoked(epochs, event_id, fname_stem, derivatives_folder)

    return raw, events, event_id, metadata


def make_metadata(
    events: NDArray[+ScalarIntType],
    events_id: dict[str, int],
    raw: BaseRaw,
    row_events: list[str],
) -> tuple[pd.DataFrame, NDArray[+ScalarIntType], dict[str, int]]:
    """Create metadata from events for each epoch.

    For each epoch, it shall include events from the range: [0.0, 1.5] s,
    i.e. starting with stimulus onset and expanding beyond the end of the epoch.
    Currently it includes [0.0, 0.999]

    Parameters
    ----------
    %(events)s
    %(events_id)s
    %(raw)s
    %(row_events)s

    Returns
    -------
    metadata : DataFrame
        Metadata of the events.
    events : array of shape (n_events, 3)
        The events.
    event_id : dict
        Mapping str to int of the events
    """
    metadata_tmin, metadata_tmax = 0.0, 0.999

    # MNE auto-generate metadata, which also returns a new events array and an event_id
    # dictionary.
    metadata, events, event_id = make_metadata_mne(
        events=events,
        event_id=events_id,
        tmin=metadata_tmin,
        tmax=metadata_tmax,
        sfreq=raw.info["sfreq"],
        row_events=row_events,
    )

conditions = [
        (metadata["event_name"].eq("target")) & (pd.notna(metadata["response"])) & (metadata["response"]<0.2),
        (metadata["event_name"].eq("target")) & (pd.notna(metadata["response"])) & (metadata["response"]>=0.2),
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


def get_SDT_outcomes(metadata: pd.DataFrame) -> tuple[int, int, int, int]:
    """Compute proportions for SDT.

    Parameters
    ----------
    %(metadata)s

    Returns
    -------
    num_hits : int
        Number of hits
    num_correct_rejections : int
        Number of correct rejections
    num_misses : int
        Number of misses
    num_false_alarms : int
        Number of false alarms
    """
    num_hits = len(metadata[metadata["response_type"] == "Hits"])
    num_correct_rejections = len(
        metadata[metadata["response_type"] == "CorrectRejections"]
    )
    num_misses = len(metadata[metadata["response_type"] == "Misses"])
    num_false_alarms = len(metadata[metadata["response_type"] == "FalseAlarms"])
    return num_hits, num_correct_rejections, num_misses, num_false_alarms


def plot_RT(hits, fname_stem, derivatives_subfolder, response_mean, response_std):
    """Plot histogram of response times.

    Parameters
    ----------
    %(hits)s
    %(fname_stem)s
    %(derivatives_subfolder)s
    %(response_mean)s
    %(response_std)s
    """
    ax_rt = hits["response"].plot.hist(
        bins=100,
        title=f"Response Times of TPs\nmean:{str(response_mean)} ({str(response_std)})",
    )

    fname_rt_plot = derivatives_subfolder / "plots" / f"{fname_stem}_step8_RT.svg"
    ax_rt.figure.suptitle(fname_stem, fontsize=16, y=1)
    ax_rt.figure.savefig(fname_rt_plot, transparent=True)


def get_indiv_behav(
    metadata,
    num_hits,
    num_correct_rejections,
    num_misses,
    num_false_alarms,
    fname_stem,
    derivatives_subfolder,
    response_mean,
    response_std,
):
    """Write a file with the individual measures.

    Parameters
    ----------
    %(metadata)s
    %(num_hits)s
    %(num_correct_rejections)s
    %(num_misses)s
    %(num_false_alarms)s
    %(fname_stem)s
    %(derivatives_subfolder)s
    %(response_mean)s
    %(response_std)s
    """
    correct_response_count = metadata["response_correct"].sum()

    logger.info(
        f"\nCorrect responses: {correct_response_count}\n"
        f"Incorrect responses: {len(metadata) - correct_response_count}\n"
    )

    logger.info("Hits, Misses, Correct Rejections, False Alarms")
    logger.info(num_hits, num_misses, num_correct_rejections, num_false_alarms, "\n")
    SDT(num_hits, num_misses, num_false_alarms, num_correct_rejections)
    metadata.groupby(by="event_name").count()

    # write behav file
    FNAME_BEHAV = derivatives_subfolder / f"{fname_stem}_step8_b-behav.txt"

    file_behav = open(FNAME_BEHAV, "w")

    file_behav.write("Hits, Misses, Correct Rejections, False Alarms\n")
    file_behav.write(
        f"{str(num_hits)}\t{str(num_misses)}\t{str(num_correct_rejections)}\t{str(num_false_alarms)}"
    )

    file_behav.write("\n\nStandard, Novel, Target\n")
    metadata_count_correct = metadata.groupby(by="event_name").count()[
        "response_correct"
    ]
    count_corr_standard = str(metadata_count_correct["standard"])
    count_corr_target = str(metadata_count_correct["target"])
    count_corr_novel = str(metadata_count_correct["novel"])

    file_behav.write(f"{count_corr_standard}\t{count_corr_novel}\t{count_corr_target}")

    file_behav.write("\n\nResponse_mean, Response_std\n")
    file_behav.write(str(response_mean) + "\t" + str(response_std))

    file_behav.write("\n\nd'\n")
    file_behav.write(
        str(SDT(num_hits, num_misses, num_false_alarms, num_correct_rejections)["d"])
    )

    file_behav.close()  # to change file access modes


def epoching(
    raw: BaseRaw,
    events: np.NDArray[+ScalarIntType],
    event_id: dict[str, int],
    metadata: pd.DataFrame,
) -> BaseEpochs:
    """Epoching.

    Parameters
    ----------
    %(raw)s
    %(events)s
    %(event_id)s
    %(metadata)s

    Returns
    -------
    epochs : Epochs
        Epoched data, -0.2 to 0.8 seconds around the stimuli, with metadata.
    """
    epochs_tmin, epochs_tmax = -0.2, 0.8
    epochs = Epochs(
        raw=raw,
        tmin=epochs_tmin,
        tmax=epochs_tmax,
        events=events,
        event_id=event_id,
        metadata=metadata,
        reject=None,
        preload=True,
        baseline=(None, 0),
        picks="eeg",
    )
    return epochs


def get_rejection(epochs):
    """Epoching.

    Parameters
    ----------
    %(epochs)s

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


def clean_epochs_from_rejection(epochs, reject, fname_stem, derivatives_subfolder):
    """Clean epochs from autoreject value.

    Parameters
    ----------
    %(epochs)s
    %(reject)s
    %(fname_stem)s
    %(derivatives_subfolder)s

    Returns
    -------
    epochs : Epochs
        Cleaned epochs
    """

    """ 
    to fix the threshold:
    reject = dict(eeg=100e-6,      # unit: V (EEG channels)
                                    # unit: V (EOG channels)
                   )
    """
    stim_before = [el[2] for el in epochs.events]
    count_stim_before = Counter(stim_before)

    epochs.drop_bad(reject=reject)

    stim_after = [el[2] for el in epochs.events]
    count_stim_after = Counter(stim_after)

    data = [
        ["1", count_stim_before[1] - count_stim_after[1]],
        ["2", count_stim_before[2] - count_stim_after[2]],
        ["3", count_stim_before[3] - count_stim_after[3]],
    ]

    df_count = pd.DataFrame(data, columns=["Stim", "nb_dropped"])
    df_count.to_csv(derivatives_subfolder / f"{fname_stem}_step8_c2-drop-epochs.csv")
    fig = epochs.plot_drop_log(subject=fname_stem)
    fname_drop_log = (
        derivatives_subfolder / "plots" / f"{fname_stem}_step8_epochs-rejected.svg"
    )
    fig.savefig(fname_drop_log, transparent=True)
    totals = Counter(i for i in list(itertools.chain.from_iterable(epochs.drop_log)))
    df_drops = pd.DataFrame.from_dict(totals, orient="index")
    df_drops = df_drops.rename(columns={0: fname_stem})
    df_drops = df_drops.sort_values(by=[fname_stem], ascending=False)
    df_drops.to_csv(
        derivatives_subfolder / f"{fname_stem}_step8_c3-drop-channel-log.csv"
    )

    return epochs


def save_evoked(epochs, event_id, fname_stem, derivatives_subfolder):
    """Save individual evoked files.

    Parameters
    ----------
    %(epochs)s
    %(event_id)s
    %(fname_stem)s
    %(derivatives_subfolder)s
    """
    epochs.metadata.groupby(
        by=[
            "event_name",
            "response_correct",
        ]
    ).count()
    # this keeps correct responses only (hits and correct rejection)
    epochs["response_correct"]

    all_evokeds = dict(
        (cond, epochs["response_correct"][cond].average()) for cond in event_id
    )
    # all_evokeds = {cond: epochs["response_correct"][cond].average() for cond in event_id}
    all_evokeds

    fname_ev_standard = derivatives_subfolder / f"{fname_stem}_step8_d1-standard_evoked-ave.fif"
    
    fname_ev_target = derivatives_subfolder / f"{fname_stem}_step8_d2-target_evoked-ave.fif"

    fname_ev_novel = derivatives_subfolder / f"{fname_stem}_step8_d3-novel_evoked-ave.fif"
    

    all_evokeds["standard"].save(fname_ev_standard)
    all_evokeds["target"].save(fname_ev_target)
    all_evokeds["novel"].save(fname_ev_novel)


def SDT2(hits, misses, fas, crs):
    """Return a dict with d-prime measures.

    Parameters
    ----------
    %(hits)s
    %(misses)s
    %(fas)s
    %(crs)s

    Returns
    ----------
    out: dict
        d' measures
    """
    Z = norm.ppf

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)

    # Return d', beta, c and Ad'
    out = {}
    out["d"] = Z(hit_rate) - Z(fa_rate)
    out["beta"] = math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)
    out["c"] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out["Ad"] = norm.cdf(out["d"] / math.sqrt(2))

    return out


def SDT(hits, misses, fas, crs):
    """Return a dict with d-prime measures + tweeks to avoid d' infinity.
    https://lindeloev.net/calculating-d-in-python-and-php/

    Parameters
    ----------
    %(hits)s
    %(misses)s
    %(fas)s
    %(crs)s

    Returns
    ----------
    out: dict
        d' measures
    """
    Z = norm.ppf
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)

    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1:
        hit_rate = 1 - half_hit
    if hit_rate == 0:
        hit_rate = half_hit

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1:
        fa_rate = 1 - half_fa
    if fa_rate == 0:
        fa_rate = half_fa

    # Return d', beta, c and Ad'
    out = {}
    out["d"] = Z(hit_rate) - Z(fa_rate)
    out["beta"] = math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)
    out["c"] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out["Ad"] = norm.cdf(out["d"] / math.sqrt(2))

    return out
