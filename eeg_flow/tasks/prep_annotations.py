# ########################################
# Modified on Mon May 08 01:01:00 2023
# @anguyen

from itertools import chain
import os

from mne.io import read_raw_fif
from mne.preprocessing import (
    compute_bridged_electrodes,
    interpolate_bridged_electrodes,
)
from pyprep import NoisyChannels
from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files
from ..viz import plot_bridged_electrodes


@fill_doc
def load_for_annotations(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
) -> None:
    """Load the necessary files for the annotation.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    """
    # prepare folders
    _, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # create derivatives plots subfolder
    os.makedirs(DERIVATIVES_SUBFOLDER / "plots", exist_ok=True)

    # create locks
    derivatives = (
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step2_info.fif"),
        DERIVATIVES_SUBFOLDER
        / (FNAME_STEM + "_step2_oddball_with_bads_annot.fif"),
        DERIVATIVES_SUBFOLDER / "plots" / (FNAME_STEM + "_step2_bridges.svg"),
    )
    locks = lock_files(*derivatives, timeout=timeout)

    # load XDF file and create raw object
    raw = read_raw_fif(
        DERIVATIVES_SUBFOLDER / (FNAME_STEM + "_step1_raw.fif"), preload=True
    )
    # for lock in locks:
    #    lock.release()
    # del locks
    return (DERIVATIVES_SUBFOLDER, FNAME_STEM, raw, locks)


@fill_doc
def check_bridges(
        DERIVATIVES_SUBFOLDER, FNAME_STEM, raw):
    """Check for bridges.

    Parameters
    ----------
    %(DERIVATIVES_SUBFOLDER)s
    %(FNAME_STEM)s
    %(raw)s
    """
    FNAME_BRIDGE_PLOT = (
        DERIVATIVES_SUBFOLDER / "plots" / (FNAME_STEM + "_step2_bridges.svg")
    )

    if not FNAME_BRIDGE_PLOT.exists():
        fig, ax = plot_bridged_electrodes(raw)
        fig.suptitle(FNAME_STEM, fontsize=16, y=1.0)
        fig.savefig(FNAME_BRIDGE_PLOT, transparent=True)
    return


@fill_doc
def interpolate_bridge_try(raw):
    """Attempt to interpolate bridged channels.

    Parameters
    ----------
    %(raw)s
    """
    bridged_idx, _ = compute_bridged_electrodes(raw)
    try:
        raw = interpolate_bridged_electrodes(raw, bridged_idx)
    except RuntimeError:
        bads_idx = sorted(set(chain(*bridged_idx)))
        raw.info["bads"] = [raw.ch_names[k] for k in bads_idx]
        assert "M1" not in raw.info["bads"]
        assert "M2" not in raw.info["bads"]
    return


@fill_doc
def auto_bad_channels(raw):
    """Look for bad channels.

    Parameters
    ----------
    %(raw)s
    """
    raw.filter(
        l_freq=1.0,
        h_freq=100.0,
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )
    ns = NoisyChannels(raw, do_detrend=False)  # operates only on EEG
    ns.find_bad_by_SNR()
    ns.find_bad_by_correlation()
    ns.find_bad_by_hfnoise()
    ns.find_bad_by_nan_flat()
    # ns.find_bad_by_ransac()  # requires electrode position
    print(ns.get_bads())

    raw.info["bads"].extend(
        [ch for ch in ns.get_bads() if ch not in ("M1", "M2")]
    )
    raw.info["bads"] = list(set(raw.info["bads"]))
    return
