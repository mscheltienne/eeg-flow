from __future__ import annotations  # c.f. PEP 563, PEP 649

from importlib.resources import files
from typing import TYPE_CHECKING

import psychtoolbox as ptb
from bsl.lsl import StreamInfo, StreamOutlet
from bsl.triggers import MockTrigger, ParallelPortTrigger
from psychopy.core import wait
from psychopy.sound.backend_ptb import SoundPTB as Sound

from ..utils._checks import check_type, check_value, ensure_path
from ..utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

    from typing import List, Tuple


_TRIAL_LIST_MAPPING = {
    "100": "trialList_1_100_vol-eval.txt",
    "600": "trialList_2_600_game-eval.txt",
    "0a": "trialList_3_360_oddball-%s-a.txt",
    "0b": "trialList_3_360_oddball-%s-b.txt",
    "a": "trialList_4_1000_oddball-game-a.txt",
    "b": "trialList_4_1000_oddball-game-b.txt",
}
_DURATION_STIM = 0.2
_DURATION_ITI = 1
_TRIGGERS = {
    "standard": 1,
    "target": 2,
    "novel": 3,
}


def oddball(condition: str, passive: bool = True, mock: bool = False) -> None:
    """Run the oddball paradigm.

    Parameters
    ----------
    condition : "100" | "600" | "0a" | "0b" | "a" | "b"
        Oddball condition to run.
    passive : bool
        Only used for the condition "0a" and "0b". If True, the oddball is passive with
        the participant is asked to count the flickering on a fixation cross while if
        False, the participant is asked to respond to stimuli physically with a button
        press.
    mock : bool
        If True, uses a MockTrigger instead of a ParallelPortTrigger.
    """
    check_type(condition, (str,), "condition")
    check_value(condition, _TRIAL_LIST_MAPPING, "condition")
    check_type(passive, (bool,), "passive")
    check_type(mock, (bool,), "mock")
    # load trials and sounds
    fname = _TRIAL_LIST_MAPPING[condition]
    if condition in ("0a", "0b"):
        type_oddball = "passive" if passive else "active"
        fname = fname % type_oddball
    fname = files("eeg_flow.oddball") / "trialList" / fname
    fname = ensure_path(fname, must_exist=True)
    trials = _parse_trial_list(fname)
    sounds = _load_sounds(trials)
    # prepare triggers
    trigger = MockTrigger() if mock else ParallelPortTrigger("/dev/parport0")
    sinfo = StreamInfo("Oddball_task", "Markers", 1, 0, "string", "myuidw43536")
    trigger_lsl = StreamOutlet(sinfo)

    # main loop
    for k, trial in trials:
        # retrieve trigger value and sound
        if trial in _TRIGGERS:
            assert trial in ("standard", "target"), f"Error with trial ({k}, {trial})."
            value = _TRIGGERS[trial]
        else:
            assert trial.startswith("wav"), f"Error with trial ({k}, {trial})."
            value = _TRIGGERS["novel"]
        sound = sounds[trial]
        # schedule sound
        now = ptb.GetSecs()
        sound.play(when=now + _DURATION_STIM)
        wait(_DURATION_STIM, hogCPUperiod=_DURATION_STIM)
        trigger.signal(value)
        trigger_lsl.push_sample([str(value)])
        wait(_DURATION_ITI - _DURATION_STIM)


def _parse_trial_list(fname: Path) -> List[Tuple[int, str]]:
    """Parse the trialList file."""
    logger.info("Loading trial list %s", fname.name)
    with open(fname) as f:
        lines = f.readlines()
    lines = [line.rstrip("\n").split(", ") for line in lines if len(line) != 0]
    novel_sounds = [sound.split("-")[0] for sound in _list_novel_sounds()]
    lines_checked = list()
    expected_idx = 1
    for line in lines:
        try:
            idx = int(line[0])
        except ValueError:
            raise ValueError(
                f"The trial idx {line[0]} could not be interpreted as an integer."
            )
        if expected_idx != idx:
            raise ValueError(
                f"The trial idx {idx} does not match the expected idx {expected_idx}."
            )
        trial = line[1]
        if not trial.startswith("wav"):
            check_value(trial, ("standard", "target", "cross"), "trial")
        else:
            check_value(trial, novel_sounds, "trial")
        if trial != "cross":
            expected_idx += 1
        lines_checked.append((idx, trial))
    return lines_checked


def _list_novel_sounds() -> List[str]:
    """List the available sounds."""
    novel_sounds = list()
    for file in (files("eeg_flow.oddball") / "sounds").iterdir():
        if file.suffix != ".wav":
            logger.warning("Non-wav file %s found in the sounds directory.", file)
            continue
        if file.name.startswith("wav"):
            novel_sounds.append(file.name)
    return novel_sounds


def _load_sounds(trials) -> Dict[str, SoundPTB]:
    """Create psychopy sound objects."""
    sounds = dict()
    fname_standard = files("eeg_flow.oddball") / "sounds" / "low_tone-48000.wav"
    fname_standard = ensure_path(fname_standard, must_exist=True)
    sounds["standard"] = Sound(
        fname_standard, secs=_DURATION_STIM, hamming=True, name="stim", sampleRate=48000
    )
    fname_target = files("eeg_flow.oddball") / "sounds" / "high_tone-48000.wav"
    fname_target = ensure_path(fname_target, must_exist=True)
    sounds["target"] = Sound(
        fname_target, secs=_DURATION_STIM, hamming=True, name="stim", sampleRate=48000
    )

    novels = [trial[1] for trial in trials if trial[1].startswith("wav")]
    for novel in novels:
        fname = files("eeg_flow.oddball") / "sounds" / f"{novel}-48000.wav"
        sounds[novel] = Sound(
            fname, secs=_DURATION_STIM, hamming=True, name="stim", sampleRate=48000
        )
    return sounds
