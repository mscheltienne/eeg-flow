from __future__ import annotations  # c.f. PEP 563, PEP 649

from pathlib import Path

import numpy as np
import pandas as pd
from mne import Epochs, find_events, write_evokeds
from mne.epochs import make_metadata as make_metadata_mne
from mne.io import read_raw_fif
from mne.preprocessing import compute_current_source_density

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_derivative_folder, get_fname
from ..utils.concurrency import lock_files
from ..utils.logs import logger

_TOO_QUICK_THRESHOLD: float = 0.2
_EVENT_ID_STIM: dict[str, int] = {"standard": 1, "target": 2, "novel": 3}
_RESPONSE_CODE: int = 64
_PICKS: list[str] = ["eeg", "eog"]


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

    Stim-locked (all cohorts, all tasks), window -200 to 800 ms, baseline
    -200 to 0 ms, EEG + EOG channels, all event types in one file. Dropped
    from the epochs file:

    - epochs overlapping a BAD annotation (manual bad segments);
    - epochs whose stimulus immediately follows a stimulus that received a
      response (active only).

    Incorrect trials are KEPT in the epochs file and labelled in the metadata
    (``response_type``, ``response_correct``). Two evoked files are written:

    - ``_step8_stimlocked_correct-ave.fif``: correct trials only — targets
      hit (RT >= 200 ms), standards and novels correctly rejected (primary);
    - ``_step8_stimlocked_alltrials-ave.fif``: all trials.

    For the passive cohort every trial is "correct", so both are identical.

    All exclusion flags are computed once on the full stimulus sequence and
    stored as metadata columns before epoching; every drop then reads the
    epochs' current metadata, so earlier drops can never shift later ones.
    Each epoch carries ``trial_idx``, its position in the full sequence.

    Response-locked (active only), window -200 to 800 ms, hits only:

    - baseline -200 to -100 ms, EEG + EOG (load paper);
    - CSD on EEG, baseline -200 to 0 ms (flow paper).

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
    is_active = participant.startswith("P")

    derivatives = [
        derivatives_folder / f"{fname_stem}_step8_stimlocked-epo.fif",
        derivatives_folder / f"{fname_stem}_step8_stimlocked_correct-ave.fif",
        derivatives_folder / f"{fname_stem}_step8_stimlocked_alltrials-ave.fif",
    ]
    if is_active:
        derivatives += [
            derivatives_folder / f"{fname_stem}_step8_responselocked_bc-200-100-epo.fif",
            derivatives_folder / f"{fname_stem}_step8_responselocked_bc-200-100-ave.fif",
            derivatives_folder / f"{fname_stem}_step8_responselocked_CSD_bc-200-0-epo.fif",
            derivatives_folder / f"{fname_stem}_step8_responselocked_CSD_bc-200-0-ave.fif",
        ]

    try:
        locks = lock_files(*derivatives, timeout=timeout)
    except RuntimeError as error:
        logger.warning(
            "Could not lock files for %s %s %s run %i — skipping: %s",
            participant, group, task, run, error,
        )
        return

    try:
        existing = [d.name for d in derivatives if d.exists()]
        if existing:
            # refuse partial overwrites: old and new files must never mix
            raise FileExistsError(", ".join(existing))
        _create_epochs_evoked(derivatives_folder, fname_stem, is_active)
    except FileNotFoundError:
        logger.error(
            "The requested file for participant %s, group %s, task %s, "
            "run %i does not exist and will be skipped.",
            participant, group, task, run,
        )
    except FileExistsError as error:
        logger.error(
            "Destination files for participant %s, group %s, task %s, run %i "
            "already exist (archive them first): %s",
            participant, group, task, run, error,
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


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _build_metadata(
    raw, events: np.ndarray, is_active: bool
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per stimulus, with behavioral classification and drop flags."""
    has_responses = bool(np.any(events[:, 2] == _RESPONSE_CODE))
    event_id_meta = dict(_EVENT_ID_STIM)
    if has_responses:
        event_id_meta["response"] = _RESPONSE_CODE

    metadata, events_stim, _ = make_metadata_mne(
        events=events,
        event_id=event_id_meta,
        tmin=0.0,
        tmax=0.999,
        sfreq=raw.info["sfreq"],
        row_events=list(_EVENT_ID_STIM),
    )
    metadata = metadata.reset_index(drop=True)
    metadata["trial_idx"] = np.arange(len(metadata))

    if is_active and has_responses:
        ev = metadata["event_name"]
        rt = metadata["response"]
        conditions = [
            (ev == "target") & rt.notna() & (rt < _TOO_QUICK_THRESHOLD),
            (ev == "target") & rt.notna() & (rt >= _TOO_QUICK_THRESHOLD),
            (ev == "target") & rt.isna(),
            (ev == "standard") & rt.notna(),
            (ev == "standard") & rt.isna(),
            (ev == "novel") & rt.notna(),
            (ev == "novel") & rt.isna(),
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
        metadata["response_type"] = np.select(
            conditions, choices, default="Unclassified"
        )
        assert not (metadata["response_type"] == "Unclassified").any()
        metadata["response_correct"] = metadata["response_type"].isin(
            ["Hits", "CorrectRejections_standard", "CorrectRejections_novel"]
        )
        # flag on the FULL stimulus sequence: previous stimulus got a response
        has_resp = rt.notna().to_numpy()
        metadata["after_response"] = np.r_[False, has_resp[:-1]]
    else:
        # passive (or active run without any response): nothing to exclude
        if "response" not in metadata.columns:
            metadata["response"] = np.nan
        metadata["response_type"] = "n/a"
        metadata["response_correct"] = True
        metadata["after_response"] = False

    return metadata, events_stim


def _drop_flagged(epochs: Epochs, flags: np.ndarray, reason: str) -> int:
    """Drop epochs where ``flags`` is True.

    ``flags`` must be computed from ``epochs.metadata`` immediately before the
    call, so it is aligned with the current epochs by construction.
    """
    flags = np.asarray(flags, dtype=bool)
    assert flags.size == len(epochs)
    idx = np.flatnonzero(flags)
    if idx.size:
        epochs.drop(idx, reason=reason)
    return int(idx.size)


def _check_stim_epochs(epochs: Epochs) -> None:
    """Invariants that must hold before anything is saved."""
    md = epochs.metadata
    # every epoch maps back to its own row of the full sequence
    assert np.array_equal(md["trial_idx"].to_numpy(), epochs.selection)
    # labels agree with event codes
    codes = md["event_name"].map(_EVENT_ID_STIM).to_numpy()
    assert np.array_equal(codes, epochs.events[:, 2])
    # after-response exclusion actually applied
    assert not md["after_response"].astype(bool).any()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def _create_epochs_evoked(
    derivatives_folder: Path,
    fname_stem: str,
    is_active: bool,
) -> None:
    """Create epochs and evoked — internal implementation."""
    raw = read_raw_fif(
        derivatives_folder / f"{fname_stem}_step7_preprocessed_raw.fif",
        preload=True,
    )
    logger.info("Loaded: %s_step7_preprocessed_raw.fif", fname_stem)

    events = find_events(raw, stim_channel="TRIGGER", verbose=False)
    metadata, events_stim = _build_metadata(raw, events, is_active)

    # ------------------------------------------------------------------
    # STIM-LOCKED
    # ------------------------------------------------------------------
    epochs_stim = Epochs(
        raw=raw,
        events=events_stim,
        event_id=_EVENT_ID_STIM,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        metadata=metadata,
        reject=None,
        reject_by_annotation=True,  # manual BAD segments
        preload=True,
        picks=_PICKS,
        verbose=False,
    )
    n_total = len(metadata)
    n_bad_segment = n_total - len(epochs_stim)

    n_after = _drop_flagged(
        epochs_stim,
        epochs_stim.metadata["after_response"].to_numpy(bool),
        "after_response",
    )
    _check_stim_epochs(epochs_stim)

    is_correct = epochs_stim.metadata["response_correct"].to_numpy(bool)
    epochs_correct = epochs_stim[is_correct]
    logger.info(
        "Stim-locked: %i stimuli | dropped %i bad segment, %i after response "
        "| kept %i (%i correct, %i incorrect)",
        n_total, n_bad_segment, n_after, len(epochs_stim),
        int(is_correct.sum()), int((~is_correct).sum()),
    )
    for cond in _EVENT_ID_STIM:
        sel = (epochs_stim.metadata["event_name"] == cond).to_numpy()
        logger.info(
            "  %-8s: %i all, %i correct", cond, int(sel.sum()),
            int((sel & is_correct).sum()),
        )

    # epochs: all trials, labelled
    fname = derivatives_folder / f"{fname_stem}_step8_stimlocked-epo.fif"
    epochs_stim.save(fname, overwrite=False)
    logger.info("Saved: %s", fname.name)

    # evoked: correct trials only (primary) and all trials
    for trials, epo in (("correct", epochs_correct), ("alltrials", epochs_stim)):
        evokeds = [
            epo[cond].average(picks="all")
            for cond in _EVENT_ID_STIM
            if cond in epo.event_id and len(epo[cond]) > 0
        ]
        fname = derivatives_folder / f"{fname_stem}_step8_stimlocked_{trials}-ave.fif"
        write_evokeds(fname, evokeds, overwrite=False)
        logger.info("Saved: %s", fname.name)

    # ------------------------------------------------------------------
    # RESPONSE-LOCKED — active only, hits only
    # ------------------------------------------------------------------
    if not is_active:
        return
    hit_mask = (metadata["response_type"] == "Hits").to_numpy()
    if not hit_mask.any():
        logger.warning("No hits — response-locked files not created.")
        return

    sfreq = raw.info["sfreq"]
    hit_rt = metadata["response"].to_numpy(float)[hit_mask]
    hit_resp_samples = events_stim[hit_mask, 0] + np.round(hit_rt * sfreq).astype(int)

    # one-to-one matching of each hit to its response trigger
    resp_events = events[events[:, 2] == _RESPONSE_CODE]
    used = np.zeros(len(resp_events), dtype=bool)
    matched = []
    for s in hit_resp_samples:
        diffs = np.abs(resp_events[:, 0] - s).astype(float)
        diffs[used] = np.inf
        best = int(np.argmin(diffs))
        if diffs[best] < 5:
            matched.append(resp_events[best])
            used[best] = True
        else:
            logger.warning("No response trigger within 5 samples of sample %i", s)
    events_resp = np.array(matched)
    assert len(events_resp) == len(hit_rt), (
        f"Hit/response mismatch: {len(events_resp)} triggers vs {len(hit_rt)} hits"
    )

    metadata_resp = pd.DataFrame(
        {
            "event_name": "response",
            "response_time": hit_rt,
            "trial_idx": metadata["trial_idx"].to_numpy()[hit_mask],
        }
    )
    epochs_resp = Epochs(
        raw=raw,
        events=events_resp,
        event_id={"response": _RESPONSE_CODE},
        tmin=-0.2,
        tmax=0.8,
        baseline=None,  # applied per paper below
        metadata=metadata_resp,
        reject=None,
        reject_by_annotation=True,
        preload=True,
        picks=_PICKS,
        verbose=False,
    )
    logger.info(
        "Response-locked: %i hits | dropped %i bad segment | kept %i",
        len(metadata_resp), len(metadata_resp) - len(epochs_resp), len(epochs_resp),
    )

    # load paper: baseline -200 to -100 ms, EEG + EOG
    epochs_load = epochs_resp.copy().apply_baseline((-0.2, -0.1))
    fname = derivatives_folder / f"{fname_stem}_step8_responselocked_bc-200-100-epo.fif"
    epochs_load.save(fname, overwrite=False)
    logger.info("Saved: %s", fname.name)
    fname = derivatives_folder / f"{fname_stem}_step8_responselocked_bc-200-100-ave.fif"
    write_evokeds(fname, [epochs_load.average(picks="all")], overwrite=False)
    logger.info("Saved: %s", fname.name)

    # flow paper: CSD on EEG, baseline -200 to 0 ms
    epochs_flow = epochs_resp.copy().apply_baseline((-0.2, 0.0)).pick("eeg")
    epochs_csd = compute_current_source_density(
        epochs_flow.interpolate_bads(),
        stiffness=3,
        n_legendre_terms=15,
        verbose=False,
    )
    fname = derivatives_folder / f"{fname_stem}_step8_responselocked_CSD_bc-200-0-epo.fif"
    epochs_csd.save(fname, overwrite=False)
    logger.info("Saved: %s", fname.name)
    fname = derivatives_folder / f"{fname_stem}_step8_responselocked_CSD_bc-200-0-ave.fif"
    write_evokeds(fname, [epochs_csd.average(picks="all")], overwrite=False)
    logger.info("Saved: %s", fname.name)
