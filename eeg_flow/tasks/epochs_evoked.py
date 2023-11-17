# ########################################
# Modified on Mon May 08 01:01:00 2023
# @anguyen

from collections import Counter
import math
import itertools
import time

import numpy as np
import pandas as pd
from autoreject import get_rejection_threshold
from mne import Epochs, find_events
from mne.epochs import make_metadata
from mne.io import read_raw_fif
from scipy.stats import norm

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_derivative_folder
from ..utils.concurrency import lock_files


@fill_doc
def behav_prep_epoching(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
) -> None:
    """Compute behav report and output metadata.
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
        derivatives_folder / (fname_stem + "_step9_a-metadata.csv"),
        derivatives_folder / (fname_stem + "_step9_b-behav.txt"),
        derivatives_folder / (fname_stem + "_step9_c1-cleaned-epo.fif"),
        derivatives_folder / (fname_stem + "_step9_c2-drop-epochs.csv"),
        derivatives_folder / (fname_stem + "_step9_c3-drop-channel-log.csv"),
        derivatives_folder / (fname_stem + "_step9_d1-standard_evoked-ave.fif"),
        derivatives_folder / (fname_stem + "_step9_d2-target_evoked-ave.fif"),
        derivatives_folder / (fname_stem + "_step9_d3-novel_evoked-ave.fif"),
    ]

    locks = lock_files(*derivatives, timeout=timeout)

    try:
        _behav_prep_epoching(participant, group, task, run)
    finally:
        for lock in locks:
            lock.release()
        del locks
    return


@fill_doc
def _behav_prep_epoching(
    participant: str,
    group: str,
    task: str,
    run: int,
) -> None:
    """Compute behav report and output metadata.
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

    # load previous steps
    # # load raw_fit recording
    raw = read_raw_fif(
        derivatives_folder / (fname_stem + "_step7_preprocessed_raw.fif"),
        preload=True,
    )

    events = find_events(raw, stim_channel="TRIGGER")

    events_id = dict(standard=1, target=2, novel=3, response=64)
    row_events = ["standard", "target", "novel"]

    metadata, events, event_id = create_metadata(events, events_id, raw, row_events)
    FNAME_METADATA = derivatives_folder / (fname_stem + "_step9_a-metadata.csv")
    metadata.to_csv(FNAME_METADATA)

    num_Hits, num_CorrectRejections, num_Misses, num_FalseAlarms = get_SDT_outcomes(metadata)

    hits = metadata[metadata["response_type"] == "Hits"]
    response_mean = round(hits["response"].mean(), 5)
    response_std = round(hits["response"].std(), 5)

    plot_RT(hits, fname_stem, derivatives_folder, response_mean, response_std)

    get_indiv_behav(metadata, num_Hits, num_CorrectRejections, num_Misses, num_FalseAlarms, 
                    fname_stem, derivatives_folder, response_mean, response_std)


    # %%
    epochs = epoching(raw, events, event_id, metadata)

    reject = get_rejection(epochs)

    epochs = clean_epochs_from_rejection(epochs, reject, fname_stem, derivatives_folder)

    
    # %%
    FNAME_CLEANED_EPOCHS = derivatives_folder / (
        fname_stem + "_step9_c1-cleaned-epo.fif"
    )
    epochs.save(FNAME_CLEANED_EPOCHS)

    save_evoked(epochs, event_id, fname_stem, derivatives_folder)

    return raw, events, event_id, metadata

def create_metadata(events, events_id, raw, row_events):
    """
    metadata for each epoch shall include events from the range: [0.0, 1.5] s,
    i.e. starting with stimulus onset and expanding beyond the end of the epoch
    """
    metadata_tmin, metadata_tmax = 0.0, 0.999

    """
    auto-create metadata
    this also returns a new events array and an event_id dictionary.
    we'll see later why this is important
    """

    metadata, events, event_id = make_metadata(
        events=events,
        event_id=events_id,
        tmin=metadata_tmin,
        tmax=metadata_tmax,
        sfreq=raw.info["sfreq"],
        row_events=row_events,
    )

    conditions = [
        metadata["event_name"].eq("target") & pd.notna(metadata["response"]),
        metadata["event_name"].eq("target") & pd.isna(metadata["response"]),
        metadata["event_name"].eq("standard") & pd.notna(metadata["response"]),
        metadata["event_name"].eq("standard") & pd.isna(metadata["response"]),
        metadata["event_name"].eq("novel") & pd.notna(metadata["response"]),
        metadata["event_name"].eq("novel") & pd.isna(metadata["response"]),
    ]
    choices = [
        "Hits",
        "Misses",
        "FalseAlarms",
        "CorrectRejections",
        "FalseAlarms",
        "CorrectRejections",
    ]

    metadata["response_type"] = np.select(conditions, choices, default=0)
    metadata["response_type"].value_counts()
    
    ########
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

def get_SDT_outcomes(metadata):
    
    num_Hits = len(metadata[metadata["response_type"] == "Hits"])
    num_CorrectRejections = len(
        metadata[metadata["response_type"] == "CorrectRejections"]
    )
    num_Misses = len(metadata[metadata["response_type"] == "Misses"])
    num_FalseAlarms = len(metadata[metadata["response_type"] == "FalseAlarms"])

    return(num_Hits, num_CorrectRejections, num_Misses, num_FalseAlarms)

def plot_RT(hits, fname_stem, derivatives_subfolder, response_mean, response_std):
    # visualize response times of TP
    ax_rt = hits["response"].plot.hist(
        bins=100,
        title=f"Response Times of TPs\nmean:{str(response_mean)} ({str(response_std)})",
    )

    fname_rt_plot = derivatives_subfolder / "plots" / (fname_stem + "_step9_RT.svg")
    ax_rt.figure.suptitle(fname_stem, fontsize=16, y=1)
    ax_rt.figure.savefig(fname_rt_plot, transparent=True)
    ax_rt

def get_indiv_behav(metadata, num_Hits, num_CorrectRejections, num_Misses, num_FalseAlarms, 
                    fname_stem, derivatives_subfolder, response_mean, response_std):
    correct_response_count = metadata["response_correct"].sum()

    print(
        f"\nCorrect responses: {correct_response_count}\n"
        f"Incorrect responses: {len(metadata) - correct_response_count}\n"
    )

    print("Hits, Misses, Correct Rejections, False Alarms")
    print(num_Hits, num_Misses, num_CorrectRejections, num_FalseAlarms,"\n")
    SDT(num_Hits, num_Misses, num_FalseAlarms, num_CorrectRejections)
    metadata.groupby(by="event_name").count()
    
    # write behav file
    FNAME_BEHAV = derivatives_subfolder / (fname_stem + "_step9_b-behav.txt")

    file_behav = open(FNAME_BEHAV, "w")

    file_behav.write("Hits, Misses, Correct Rejections, False Alarms\n")
    file_behav.write(f"{str(num_Hits)}\t{str(num_Misses)}\t{str(num_CorrectRejections)}\t{str(num_FalseAlarms)}")


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
        str(SDT(num_Hits, num_Misses, num_FalseAlarms, num_CorrectRejections)["d"])
    )

    file_behav.close()  # to change file access modes


def epoching(raw, events, event_id, metadata):
    """Epoching.
    Parameters
    ----------
    %(raw)s
    %(events)s
    %(event_id)s
    %(metadata)s
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
    starttime = time.time()
    reject = get_rejection_threshold(epochs, decim=1, ch_types="eeg", random_state=888)
    endtime = time.time()
    diff = endtime - starttime
    minutes = str(int(diff // 60)).zfill(2)
    seconds = str(int(diff % 60)).zfill(2)
    print("\nPeak-to-peak rejection threshold computed: %s", reject)
    print(f"Elapsed {minutes}:{seconds}\n")
    return reject

def clean_epochs_from_rejection(epochs, reject, FNAME_STEM, DERIVATIVES_SUBFOLDER):
    # reject = dict(eeg=100e-6,      # unit: V (EEG channels)
    #                  # unit: V (EOG channels)
    #               )

    stim_before = [el[2] for el in epochs.events]
    count_stim_before = Counter(stim_before)




    epochs.drop_bad(reject=reject)

    stim_after = [el[2] for el in epochs.events]
    count_stim_after = Counter(stim_after)

    data = [['1', count_stim_before[1]-count_stim_after[1]], 
            ['2', count_stim_before[2]-count_stim_after[2]], 
            ['3', count_stim_before[3]-count_stim_after[3]]] 

    df_count = pd.DataFrame(data, columns=['Stim', 'nb_dropped']) 

    df_count.to_csv(DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step9_c2-drop-epochs.csv"))

    fig = epochs.plot_drop_log(subject=FNAME_STEM)

    FNAME_DROP_LOG = (
        DERIVATIVES_SUBFOLDER / "plots" / (FNAME_STEM + "_step9_epochs-rejected.svg")
    )
    fig.savefig(FNAME_DROP_LOG, transparent=True)

    totals = Counter(i for i in list(itertools.chain.from_iterable(epochs.drop_log)))
    df_drops = pd.DataFrame.from_dict(totals, orient = "index")
    df_drops = df_drops.rename(columns={0: FNAME_STEM})
    df_drops = df_drops.sort_values(by=[FNAME_STEM], ascending = False)
    df_drops.to_csv(DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step9_c3-drop-channel-log.csv"))

    return(epochs)

def save_evoked(epochs, event_id, FNAME_STEM, DERIVATIVES_SUBFOLDER):

    epochs.metadata.groupby(
        by=[
            "event_name",
            "response_correct",
        ]
    ).count()
    # this keeps correct responses only (hits and correct rejection)
    epochs["response_correct"]

    # %%
    all_evokeds = dict(
        (cond, epochs["response_correct"][cond].average()) for cond in event_id
    )
    # all_evokeds = {cond: epochs["response_correct"][cond].average() for cond in event_id}
    all_evokeds

    FNAME_EV_STANDARD = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step9_d1-standard_evoked-ave.fif"
    )
    FNAME_EV_TARGET = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step9_d2-target_evoked-ave.fif"
    )
    FNAME_EV_NOVEL = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step9_d3-novel_evoked-ave.fif"
    )

    all_evokeds["standard"].save(FNAME_EV_STANDARD)
    all_evokeds["target"].save(FNAME_EV_TARGET)
    all_evokeds["novel"].save(FNAME_EV_NOVEL)


def SDT2(hits, misses, fas, crs):
    """Return a dict with d-prime measures.
    Parameters
    ----------
    %(hits)s
    %(misses)s
    %(fas)s
    %(crs)s
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
    Parameters
    ----------
    %(hits)s
    %(misses)s
    %(fas)s
    %(crs)s
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