from itertools import chain

from mne.io import read_raw_fif, write_info
from mne.preprocessing import compute_bridged_electrodes, interpolate_bridged_electrodes
from pyprep import NoisyChannels

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.annotations import merge_bad_annotations
from ..utils.bids import get_fname, get_folder
from ..utils.concurrency import lock_files
from ..viz import plot_bridged_electrodes


@fill_doc
def annotate_bad_channels_and_segments(
    participant: int,
    group: int,
    task: str,
    run: int,
    overwrite: bool = False,
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
    %(timeout)s
    """
    # prepare folders
    _, derivatives_folder, experimenter = load_config()
    derivatives_folder = get_folder(derivatives_folder, participant, group)
    fname_stem = get_fname(participant, group, task, run)

    # lock the output derivative files
    derivatives = (
        derivatives_folder / (fname_stem + "_2_info.fif"),
        derivatives_folder / (fname_stem + "_oddball_with_bads_2_annot.fif"),
    )
    locks = lock_files(*derivatives, timeout=timeout)
    try:
        _annotate_bad_channels_and_segments(participant, group, task, run, overwrite)
    finally:
        for lock in locks:
            lock.release()
        del locks


@fill_doc
def _annotate_bad_channels_and_segments(
    participant: int,
    group: int,
    task: str,
    run: int,
    overwrite: bool = False,
) -> None:
    """Convert the XDF recording to a raw FIFF file.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    overwrite : bool
        If True, overwrites existing derivatives.
    """
    # prepare folders
    _, derivatives_folder, experimenter = load_config()
    derivatives_folder = get_folder(derivatives_folder, participant, group)
    fname_stem = get_fname(participant, group, task, run)

    # load raw
    raw = read_raw_fif(derivatives_folder / (fname_stem + "_1_raw.fif"), preload=True)

    # fix bridge electrodes
    plot_bridged_electrodes(raw)
    raw.set_montage("standard_1020")  # we need a montage for the interpolation
    bridged_idx, _ = compute_bridged_electrodes(raw)
    try:
        raw = interpolate_bridged_electrodes(raw, bridged_idx)
    except RuntimeError:
        bads_idx = sorted(set(chain(*bridged_idx)))
        raw.info["bads"] = [raw.ch_names[k] for k in bads_idx]
        assert "M1" not in raw.info["bads"]
        assert "M2" not in raw.info["bads"]

    # find bad channels
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
    ns.find_bad_by_ransac()  # requires electrode position

    raw.info["bads"].extend([ch for ch in ns.get_bads() if ch not in ("M1", "M2")])
    raw.info["bads"] = list(set(raw.info["bads"]))

    # visual inspection and annotate bad segments
    raw.plot(theme="light", block=True)
    annotations = merge_bad_annotations(raw)

    # save derivatives
    fname = derivatives_folder / (fname_stem + "_2_info.fif")
    if not overwrite:
        assert not fname.exists()  # write_info always overwrites
    write_info(fname, raw.info)
    derivatives_folder / (fname_stem + "_oddball_with_bads_2_annot.fif"),
    annotations.save(fname, overwrite=False)
