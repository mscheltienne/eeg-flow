from __future__ import annotations  # c.f. PEP 563, PEP 649

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np
import psychtoolbox as ptb
from bsl.lsl import StreamInfo, StreamOutlet
from bsl.triggers import MockTrigger, ParallelPortTrigger
from psychopy.core import wait
from psychopy.sound.backend_ptb import SoundPTB
from psychopy.visual import ShapeStim, Window

from ..utils._checks import check_type, check_value, ensure_path
from ..utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple


_TRIAL_LIST_MAPPING = {
    "100": "trialList_1_100_vol-eval.txt",
    "600": "trialList_2_600_game-eval.txt",
    "0a": "trialList_3_360_oddball-%s-a.txt",
    "0b": "trialList_3_360_oddball-%s-b.txt",
    "a": "trialList_4_1000_oddball-game-a.txt",
    "b": "trialList_4_1000_oddball-game-b.txt",
}
_DURATION_STIM: float = 0.2  # seconds
_DURATION_ITI: float = 1.0  # seconds
_DUATION_FLICKERING: float = 0.05  # seconds
assert 0 < _DURATION_ITI - _DURATION_STIM - 0.3
_TRIGGERS: Dict[str, int] = {
    "standard": 1,
    "target": 2,
    "novel": 3,
}
_CROSS_WIDTH: int = 20  # pixels
_CROSS_LENGTH: int = 100  # pixels
_CROSS_COLOR: str = "black"
_CROSS_FLICKERING_COLOR: str = "white"


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
        fname = fname % ("passive" if passive else "active")
    fname = files("eeg_flow.oddball") / "trialList" / fname
    fname = ensure_path(fname, must_exist=True)
    trials = _parse_trial_list(fname)
    sounds = _load_sounds(trials)
    # prepare triggers
    trigger = MockTrigger() if mock else ParallelPortTrigger("/dev/parport0")
    sinfo = StreamInfo("Oddball_task", "Markers", 1, 0, "string", "myuidw43536")
    trigger_lsl = StreamOutlet(sinfo)
    # prepare fixation cross window
    win = Window(
        units="norm",
        winType="pyglet",
        fullscr=True,
        allowGUI=False,
    )
    crosses = _load_cross(win)
    rng = np.random.default_rng()
    # display fixation cross
    crosses["full"][0].setAutoDraw(True)
    win.flip()
    # main loop
    for i, (k, trial) in enumerate(trials):
        # retrieve trigger value and sound
        if trial in _TRIGGERS:
            assert trial in ("standard", "target"), f"Error with trial ({k}, {trial})."
            value = _TRIGGERS[trial]
        else:
            assert trial.startswith("wav"), f"Error with trial ({k}, {trial})."
            value = _TRIGGERS["novel"]
        sound = sounds[trial]
        # schedule sound, wait, and deliver triggers simultanouesly with the sound
        now = ptb.GetSecs()
        sound.play(when=now + _DURATION_STIM)
        wait(_DURATION_STIM, hogCPUperiod=_DURATION_STIM)
        trigger.signal(value)
        trigger_lsl.push_sample([str(value)])
        # look ahead for the next trial and handle it now if it's a cross for the
        # passive oddball task
        if i == len(trials) - 1:  # end of the trial list
            continue
        elif trials[i + 1][1] == "cross":
            assert passive, f"Trial 'cross' ({k}) found in a non-passive trial-list."
            logger.info("Flickering the fixation cross for trial %i.", k)
            # pick a random time at which the flickering will occur
            delay = rng.uniform(0.3, _DURATION_ITI - _DURATION_STIM - 0.3)
            arm = rng.choice(("top", "left", "right", "bottom"))
            wait(delay)
            crosses["full"][0].setAutoDraw(False)
            for shape in crosses[arm]:
                shape.setAutoDraw(True)
            win.flip()
            wait(_DUATION_FLICKERING)
            crosses["full"][0].setAutoDraw(True)
            for shape in crosses[arm]:
                shape.setAutoDraw(False)
            win.flip()
            wait(_DURATION_ITI - _DURATION_STIM - _DUATION_FLICKERING - delay)
        else:
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
    sounds["standard"] = SoundPTB(
        fname_standard, secs=_DURATION_STIM, hamming=True, name="stim", sampleRate=48000
    )
    fname_target = files("eeg_flow.oddball") / "sounds" / "high_tone-48000.wav"
    fname_target = ensure_path(fname_target, must_exist=True)
    sounds["target"] = SoundPTB(
        fname_target, secs=_DURATION_STIM, hamming=True, name="stim", sampleRate=48000
    )

    novels = [trial[1] for trial in trials if trial[1].startswith("wav")]
    for novel in novels:
        fname = files("eeg_flow.oddball") / "sounds" / f"{novel}-48000.wav"
        sounds[novel] = SoundPTB(
            fname, secs=_DURATION_STIM, hamming=True, name="stim", sampleRate=48000
        )
    return sounds


def _load_cross(win: Window) -> Dict[str, Tuple[ShapeStim, Optional[ShapeStim]]]:
    """Load the shapes used for fixation cross.

    The point position is defined as:

            0  11



     2      1  10     9

     3      4  7      8



            5  6
    """
    crosses = dict()
    # convert the number of pixels into the normalized unit per axis (x, y)
    width = _CROSS_WIDTH * 2 / win.size
    length = _CROSS_LENGTH * 2 / win.size
    points = np.array(
        [
            [-width[0] / 2, width[1] / 2 + length[1]],  # top-left
            [-width[0] / 2, width[1] / 2],  # center-left-up
            [-width[0] / 2 - length[0], width[1] / 2],  # left-up
            [-width[0] / 2 - length[0], -width[1] / 2],  #  left-bottom
            [-width[0] / 2, -width[1] / 2],  # center-left-bottom
            [-width[0] / 2, -width[1] / 2 - length[1]],  # bottom-left
            [width[0] / 2, -width[1] / 2 - length[1]],  # bottom-right
            [width[0] / 2, -width[1] / 2],  # center-right-bottom
            [width[0] + length[0], -width[1] / 2],  # right-bottom
            [width[0] + length[0], width[1] / 2],  # right-up
            [width[0] / 2, width[1] / 2],  # center-right-up
            [width[0] / 2, width[1] / 2 + length[1]],  # top-right
        ]
    )

    # full-cross
    crosses["full"] = (
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_COLOR,
            vertices=points,
        ),
        None,
    )

    # left-flickering
    crosses["left"] = (
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_COLOR,
            vertices=points[[0, 5, 6, 7, 8, 9, 10, 11]],
        ),
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_FLICKERING_COLOR,
            vertices=points[[1, 2, 3, 4]],
        ),
    )

    # bottom-flickering
    crosses["bottom"] = (
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_COLOR,
            vertices=points[[0, 1, 2, 3, 8, 9, 10, 11]],
        ),
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_FLICKERING_COLOR,
            vertices=points[[4, 5, 6, 7]],
        ),
    )

    # right-flickering
    crosses["right"] = (
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_COLOR,
            vertices=points[[0, 1, 2, 3, 4, 5, 6, 11]],
        ),
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_FLICKERING_COLOR,
            vertices=points[[7, 8, 9, 10]],
        ),
    )

    # top-flickering
    crosses["top"] = (
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_COLOR,
            vertices=points[[2, 3, 4, 5, 6, 7, 8, 9]],
        ),
        ShapeStim(
            win,
            units="norm",
            lineColor=None,
            fillColor=_CROSS_FLICKERING_COLOR,
            vertices=points[[0, 1, 10, 11]],
        ),
    )

    return crosses
