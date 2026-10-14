from __future__ import annotations  # c.f. PEP 563, PEP 649

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import mne
from mne import Epochs, find_events, write_evokeds
from mne.epochs import make_metadata as make_metadata_mne
from mne.io import read_raw_fif
from mne_lsl.utils.logs import warn
from mne.preprocessing import compute_current_source_density

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files
from ..utils.logs import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray, DTypeLike
    ScalarIntType: tuple[DTypeLike, ...] = (
        np.int8, np.int16, np.int32, np.int64
    )

_TOO_QUICK_THRESHOLD: float = 0.2


@fill_doc
def create_epochs_evoked(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
) -> None:
    """Create epochs and evoked responses from the preprocessed raw recording.

    Produces stim-locked epochs/evoked for all cohorts and tasks.
    For the active cohort UT task only, also produces:
    - response-locked epochs/evoked with baseline -200 to -100ms (load paper)
    - response-locked epochs/evoked with CSD, baseline -200 to 0ms (flow paper)

    Stim-locked epochs include all three event types (standard, target, novel)
    in a single file. Epochs immediately following a response are dropped.
    Incorrect trials are dropped.

    Response-locked epochs are filtered to hits only (correct target responses
    with RT >= 200ms). CSD is applied to EEG channels only.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    %(timeout)s
    """
    _, derivatives_folder_root, _ = load_config()
    derivatives_folder = get_derivative_folder(
        derivatives_folder_root, participant, group, task, run
    )
    fname_stem = get_fname(participant, group, task, run)
    is_active  = participant.startswith("P")

    derivatives = [
        derivatives_folder / f"{fname_stem}_step8_stimlocked-epo.fif",
        derivatives_folder / f"{fname_stem}_step8_stimlocked-ave.fif",
    ]

    if is_active:
        derivatives += [
            derivatives_folder
            / f"{fname_stem}_step8_responselocked_bc-200-100-epo.fif",
            derivatives_folder
            / f"{fname_stem}_step8_responselocked_bc-200-100-ave.fif",
            derivatives_folder
            / f"{fname_stem}_step8_responselocked_CSD_bc-200-0-epo.fif",
            derivatives_folder
            / f"{fname_stem}_step8_responselocked_CSD_bc-200-0-ave.fif",
        ]

    try:
        locks = lock_files(*derivatives, timeout=timeout)
    except RuntimeError as error:
        logger.warning(
            "Could not lock files for %s %s %s run %i — "
            "skipping (files may already be in use): %s",
            participant, group, task, run, error,
        )
        return

    try:
        if all(derivative.exists() for derivative in derivatives):
            raise FileExistsError
        _create_epochs_evoked(
            participant, group, task, run,
            derivatives_folder, fname_stem, is_active,
        )
    except FileNotFoundError:
        logger.error(
            "The requested file for participant %s, group %s, task %s, "
            "run %i does not exist and will be skipped.",
            participant, group, task, run,
        )
    except FileExistsError:
        logger.error(
            "The destination file for participant %s, group %s, task %s, "
            "run %i already exists.",
            participant, group, task, run,
        )
    except Exception as error:
        logger.error(
            "The file for participant %s, group %s, task %s, run %i "
            "could not be processed.",
            participant, group, task, run,
        )
        logger.exception(error)
    finally:
        for lock in locks:
            lock.release()
        del locks


def _create_epochs_evoked(
    participant: str,
    group: str,
    task: str,
    run: int,
    derivatives_folder: Path,
    fname_stem: str,
    is_active: bool,
) -> None:
    """Create epochs and evoked — internal implementation."""

    # --- load step7 ---
    raw = read_raw_fif(
        derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif",
        preload=True,
    )
    logger.info("Loaded: %s", f"{fname_stem}_step7_preprocessed_raw.fif")

    # --- find events ---
    events        = find_events(raw, stim_channel="TRIGGER", verbose=False)
    event_id_stim = {"standard": 1, "target": 2, "novel": 3}

    # include response in event_id so make_metadata_mne creates response column
    # row_events still produces one row per stimulus event only
    has_responses = np.any(events[:, 2] == 64)
    if has_responses:
        event_id_meta = {"standard": 1, "target": 2, "novel": 3, "response": 64}
    else:
        event_id_meta = event_id_stim

    # --- build metadata ---
    metadata, events_with_meta, _ = make_metadata_mne(
        events=events,
        event_id=event_id_meta,
        tmin=0.0,
        tmax=0.999,
        sfreq=raw.info["sfreq"],
        row_events=["standard", "target", "novel"],
    )

    # --- classify trial types ---
    if is_active and "response" in metadata.columns:
        conditions = [
            (metadata["event_name"].eq("target"))
            & (pd.notna(metadata["response"]))
            & (metadata["response"] < _TOO_QUICK_THRESHOLD),
            (metadata["event_name"].eq("target"))
            & (pd.notna(metadata["response"]))
            & (metadata["response"] >= _TOO_QUICK_THRESHOLD),
            (metadata["event_name"].eq("target"))
            & (pd.isna(metadata["response"])),
            (metadata["event_name"].eq("standard"))
            & (pd.notna(metadata["response"])),
            (metadata["event_name"].eq("standard"))
            & (pd.isna(metadata["response"])),
            (metadata["event_name"].eq("novel"))
            & (pd.notna(metadata["response"])),
            (metadata["event_name"].eq("novel"))
            & (pd.isna(metadata["response"])),
        ]
        choices = [
            "FalseAlarms_tooquick",
            "Hits",
            "Misses",
            "FalseAlarms_standard",
            "CorrectRejections_standard",
            "FalseAlarms_novel",
            "CorrectRejections_novel",
        ]
        metadata["response_type"]    = np.select(conditions, choices, default=0)
        metadata["response_correct"] = (
            (metadata["response_type"] == "Hits")
            | (metadata["response_type"] == "CorrectRejections_standard")
            | (metadata["response_type"] == "CorrectRejections_novel")
        )
    else:
        # passive or no responses — skip behavioral classification
        metadata["response_type"]    = "n/a"
        metadata["response_correct"] = True

    # ----------------------------------------------------------------
    # STIM-LOCKED — all cohorts, all tasks
    # window: -200 to 800ms, baseline: -200 to 0ms
    # all event types in one file
    # ----------------------------------------------------------------

    epochs_stim = Epochs(
        raw=raw,
        events=events_with_meta,
        event_id=event_id_stim,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        metadata=metadata,
        reject=None,
        preload=True,
        picks="all",
        verbose=False,
    )

    # drop epoch immediately following a response
    if "response" in metadata.columns:
        idx_after_resp = np.where(
            ~np.isnan(metadata["response"].values)
        )[0] + 1
        idx_after_resp = idx_after_resp[idx_after_resp < len(epochs_stim)]
        epochs_stim.drop(idx_after_resp, reason="epoch after response")

        # drop incorrect trials
        incorrect_idx = np.where(
            metadata["response_correct"].values == False  # noqa: E712
        )[0]
        incorrect_idx = incorrect_idx[incorrect_idx < len(epochs_stim)]
        epochs_stim.drop(incorrect_idx, reason="incorrect_trial")

    logger.info(
        "Stim-locked: %i epochs kept, %i dropped",
        len(epochs_stim),
        sum(1 for d in epochs_stim.drop_log if len(d) > 0),
    )

    # save stim-locked epochs
    fname = derivatives_folder / f"{fname_stem}_step8_stimlocked-epo.fif"
    epochs_stim.save(fname, overwrite=False)
    logger.info("Saved: %s", fname.name)

    # save stim-locked evoked — one per event type, all in one file
    evokeds_stim = [
        epochs_stim[cond].average(picks="all")
        for cond in event_id_stim
        if cond in epochs_stim.event_id
    ]
    fname = derivatives_folder / f"{fname_stem}_step8_stimlocked-ave.fif"
    write_evokeds(fname, evokeds_stim, overwrite=False)
    logger.info("Saved: %s", fname.name)

    # ----------------------------------------------------------------
    # RESPONSE-LOCKED — active cohort, UT task only
    # window: -800 to 800ms
    # hits only (correct target responses with RT >= 200ms)
    # ----------------------------------------------------------------

    if is_active and "response" in metadata.columns:

        # filter response events to hits only
        hit_mask         = metadata["response_type"].values == "Hits"
        hit_stim_samples = events_with_meta[hit_mask, 0]
        hit_rt_values    = metadata["response"].values[hit_mask]
        sfreq            = raw.info["sfreq"]
        hit_resp_samples = hit_stim_samples + \
                           (hit_rt_values * sfreq).astype(int)

        all_resp_events  = events[events[:, 2] == 64]
        #  one-to-one matching, handles trigger bounce
        used = set()
        events_resp_hits = []
        for s in hit_resp_samples:
            diffs = np.abs(all_resp_events[:, 0] - s).astype(float)
            for idx in used:
                diffs[idx] = np.inf
            best_idx = np.argmin(diffs)
            if diffs[best_idx] < 5:
                events_resp_hits.append(all_resp_events[best_idx])
                used.add(best_idx)
            else:
                warn(f"No response event found within 5 samples of hit at sample {s}")

        events_resp_hits = np.array(events_resp_hits)
        assert len(events_resp_hits) == len(hit_rt_values), (
            f"Hit response event count mismatch after one-to-one matching: "
            f"{len(events_resp_hits)} events vs {len(hit_rt_values)} hits"
        )

        metadata_response = pd.DataFrame({
            "event_name":    ["response"] * len(events_resp_hits),
            "response_time": hit_rt_values,
        })

        # epoch hits with no baseline — apply each BC separately below
        epochs_resp = Epochs(
            raw=raw,
            events=events_resp_hits,
            event_id={"response": 64},
            tmin=-0.2,
            tmax=0.8,
            baseline=None,
            metadata=metadata_response,
            reject=None,
            preload=True,
            picks="all",
            verbose=False,
        )
        logger.info(
            "Response-locked hits: %i epochs kept, %i dropped",
            len(epochs_resp),
            sum(1 for d in epochs_resp.drop_log if len(d) > 0),
        )

        # --- load paper: bc -200 to -100ms, all channels ---
        epochs_resp_load = epochs_resp.copy().apply_baseline((-0.2, -0.1))

        fname = derivatives_folder / \
            f"{fname_stem}_step8_responselocked_bc-200-100-epo.fif"
        epochs_resp_load.save(fname, overwrite=False)
        logger.info("Saved: %s", fname.name)

        evoked_resp_load = epochs_resp_load.average(picks="all")
        fname = derivatives_folder / \
            f"{fname_stem}_step8_responselocked_bc-200-100-ave.fif"
        write_evokeds(fname, [evoked_resp_load], overwrite=False)
        logger.info("Saved: %s", fname.name)

        # --- flow paper: CSD, bc -200 to 0ms, EEG only ---
        epochs_resp_flow = epochs_resp.copy().apply_baseline((-0.2, 0.0))
        epochs_resp_eeg  = epochs_resp_flow.copy().pick("eeg")
        epochs_resp_CSD  = compute_current_source_density(
            epochs_resp_eeg.interpolate_bads(),
            stiffness=3,
            n_legendre_terms=15,
            verbose=False,
        )

        fname = derivatives_folder / \
            f"{fname_stem}_step8_responselocked_CSD_bc-200-0-epo.fif"
        epochs_resp_CSD.save(fname, overwrite=False)
        logger.info("Saved: %s", fname.name)

        evoked_resp_CSD = epochs_resp_CSD.average(picks="all")
        fname = derivatives_folder / \
            f"{fname_stem}_step8_responselocked_CSD_bc-200-0-ave.fif"
        write_evokeds(fname, [evoked_resp_CSD], overwrite=False)
        logger.info("Saved: %s", fname.name)