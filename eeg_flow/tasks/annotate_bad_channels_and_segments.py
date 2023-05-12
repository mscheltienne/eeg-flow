# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

import os
from itertools import chain
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from mne.io import read_raw_fif, write_info
from mne.preprocessing import compute_bridged_electrodes, interpolate_bridged_electrodes
from pyprep import NoisyChannels

from ..config import load_config
from ..utils._checks import check_type
from ..utils._cli import query_yes_no
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_derivative_folder
from ..utils.concurrency import lock_files
from ..viz import plot_bridged_electrodes

if TYPE_CHECKING:
    from pathlib import Path

    from mne.io import BaseRaw


@fill_doc
def annotate_bad_channels_and_segments(
    participant: str,
    group: str,
    task: str,
    run: int,
    overwrite: bool = False,
    ransac: bool = False,
    *,
    timeout: float = 10,
) -> None:
    """Annotate bad channels and segments.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    overwrite : bool
        If True, overwrites existing derivatives.
    ransac : bool
        If True, uses RANSAC to auto-detect bad channels (slow).
    %(timeout)s
    """
    check_type(overwrite, (bool,), "overwrite")
    check_type(ransac, (bool,), "ransac")
    # prepare folders
    _, derivatives_folder, _ = load_config()
    derivatives_folder = get_folder(derivatives_folder, participant, group, task, run)
    fname_stem = get_fname(participant, group, task, run)
    os.makedirs(derivatives_folder / "plots", exist_ok=True)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / f"{fname_stem}_step2_info.fif",
        derivatives_folder / f"{fname_stem}_step2_oddball_with_bads_annot.fif",
        derivatives_folder / "plots" / f"{fname_stem}_step2_bridges.svg",
        derivatives_folder / f"{fname_stem}_step2_raw.fif",
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        raw = read_raw_fif(
            derivatives_folder / f"{fname_stem}_step1_raw.fif", preload=True
        )
        _plot_gel_bridges(derivatives_folder, fname_stem, raw, overwrite)
        _interpolate_gel_bridges(raw)
        if not query_yes_no("Do you want to continue with this dataset?"):
            raise RuntimeError("Execution aborted by the user.")
        _auto_bad_channels(raw, ransac=ransac)
        raw.plot(theme="light", highpass=1.0, lowpass=40.0, block=True)

        # save info with bad channels
        fname = derivatives_folder / f"{fname_stem}_step2_info.fif"
        if not fname.exists() or overwrite:
            write_info(fname, raw.info)
        else:
            raise RuntimeError(f"Info file {fname.name} does already exist.")

        # save oddball + bad segments annotations
        fname = derivatives_folder / f"{fname_stem}_step2_oddball_with_bads_annot.fif"
        annotations.save(fname, overwrite=overwrite)

        # save interpolated raw
        fname = derivatives_folder / f"{fname_stem}_step2_raw.fif"
        raw.save(fname, overwrite=overwrite)
    finally:
        for lock in locks:
            lock.release()
        del locks


def _plot_gel_bridges(
    derivatives_folder: Path, fname_stem: str, raw: BaseRaw, overwrite: bool
) -> None:
    fname = derivatives_folder / "plots" / f"{fname_stem}_step2_bridges.svg"
    if not fname.exists() or overwrite:
        fig, _ = plot_bridged_electrodes(raw)
        fig.suptitle(fname_stem, fontsize=16, y=1.0)
        fig.savefig(fname, transparent=True)
        plt.show()


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
    raw.info["bads"].extend([ch for ch in ns.get_bads() if ch not in ("M1", "M2")])
    raw.info["bads"] = list(set(raw.info["bads"]))
