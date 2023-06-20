# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

import os
from itertools import chain
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from mne.io import read_raw_fif
from mne.preprocessing import compute_bridged_electrodes, interpolate_bridged_electrodes
from pyprep import NoisyChannels

from .. import logger
from ..config import load_config
from ..utils._checks import check_type, check_value
from ..utils._cli import query_yes_no
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files
from ..viz import plot_bridged_electrodes

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io import BaseRaw


def bridges_and_autobads(
    participant: str,
    group: str,
    task: str,
    run: int,
    ransac: bool = False,
    *,
    timeout: float = 10,
    overwrite: bool = False,
) -> None:
    """Find bridges and find auto bads.
    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    ransac : bool
        If True, uses RANSAC to auto-detect bad channels (slow).
    %(timeout)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    check_type(overwrite, (bool,), "overwrite")
    check_type(ransac, (bool,), "ransac")
    # prepare folders
    _, derivatives_folder, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)
    os.makedirs(derivatives_folder / "plots", exist_ok=True)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / "plots" / f"{fname_stem}_step1b_bridges.svg",
        derivatives_folder / f"{fname_stem}_step1b_with-bads_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        if not overwrite:
            raise NotOverwriteError

        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step1_raw.fif", preload=True
        )
        _plot_gel_bridges(derivatives_folder, fname_stem, raw, overwrite)
        _interpolate_gel_bridges(raw)
        plt.close("all")
        _auto_bad_channels(raw, ransac=ransac)
        # save interpolated raw
        fname = derivatives_folder / f"{fname_stem}_step1b_with-bads_raw.fif"
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
            "All the destination files for participant %s, group %s, task %s, run %i "
            "already exists. Please use 'overwrite=True' to force overwriting.",
            participant,
            group,
            task,
            run,
        )
    except FileNotFoundError:
        logger.error(
            "Overwrite was set on False"
            )
    finally:
        for lock in locks:
            lock.release()
        del locks


@fill_doc
def annotate_bad_channels_and_segments(
    participant: str,
    group: str,
    task: str,
    run: int,
    ransac: bool = False,
    *,
    timeout: float = 10,
    overwrite: bool = False,
) -> None:
    """Annotate bad channels and segments.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    ransac : bool
        If True, uses RANSAC to auto-detect bad channels (slow).
    %(timeout)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    check_type(overwrite, (bool,), "overwrite")
    check_type(ransac, (bool,), "ransac")
    # prepare folders
    _, derivatives_folder, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)
    os.makedirs(derivatives_folder / "plots", exist_ok=True)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step2_with-bads_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step1b_with-bads_raw.fif", preload=True
        )
        raw.plot(theme="light", highpass=1.0, lowpass=40.0, block=True)

        # save interpolated raw
        fname = derivatives_folder / f"{fname_stem}_step2_with-bads_raw.fif"
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


def _plot_gel_bridges(
    derivatives_folder: Path, fname_stem: str, raw: BaseRaw, overwrite: bool
) -> None:
    fname = derivatives_folder / "plots" / f"{fname_stem}_step1b_bridges.svg"
    if not fname.exists() or overwrite:
        fig, _ = plot_bridged_electrodes(raw)
        fig.suptitle(fname_stem, fontsize=16, y=1.0)
        fig.savefig(fname, transparent=True)
        plt.pause(0.1)


def _interpolate_gel_bridges(raw: BaseRaw):
    raw.set_montage("standard_1020")  # we need a montage for the interpolation
    bridged_idx, _ = compute_bridged_electrodes(raw)
    try:
        interpolate_bridged_electrodes(raw, bridged_idx)
    except RuntimeError:
        bads_idx = sorted(set(chain(*bridged_idx)))
        raw.info["bads"] = [raw.ch_names[k] for k in bads_idx]
        assert "M1" not in raw.info["bads"]
        assert "M2" not in raw.info["bads"]


def _auto_bad_channels(raw: BaseRaw, *, ransac: bool = False):
    # operates on a copy of the raw data, and filters if needed.
    ns = NoisyChannels(raw, do_detrend=False)
    ns.find_bad_by_SNR()
    ns.find_bad_by_correlation()
    ns.find_bad_by_hfnoise()
    ns.find_bad_by_nan_flat()
    if ransac:
        ns.find_bad_by_ransac()  # requires electrode position

    # raw.info["bads"] should be empty at this point, but just in case, let's merge both
    # list together.
    logger.info("Bad channel suggested by PyPREP: %s", ns.get_bads())
    raw.info["bads"].extend([ch for ch in ns.get_bads() if ch not in ("M1", "M2")])
    raw.info["bads"] = list(set(raw.info["bads"]))


def view_annotated_raw(
    participant: str,
    group: str,
    task: str,
    run: int,
    step_to_load: str,
    overwrite: bool,
    *,
    timeout: float = 10,
) -> None:
    """Plot annotated raw.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    step_to_load : str
    overwrite : bool
        If True, overwrites existing derivatives.
    %(timeout)s
    """
    check_type(step_to_load, (str,), "step_to_load")
    check_value(step_to_load, ("step2", "step6"), "step_to_load")
    step_to_load = step_to_load if step_to_load == "step2" else "step6_preprocessed"
    check_type(overwrite, (bool,), "overwrite")
    # prepare folders
    _, derivatives_folder, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (derivatives_folder / f"{fname_stem}_{step_to_load}_bis_raw.fif",)
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_{step_to_load}_raw.fif", preload=True
        )
        raw.plot(theme="light", highpass=1.0, lowpass=40.0, block=True)
        if query_yes_no("Do you want to save this dataset?"):
            fname = derivatives_folder / f"{fname_stem}_{step_to_load}_bis_raw.fif"
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
