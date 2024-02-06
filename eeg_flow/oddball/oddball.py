from __future__ import annotations  # c.f. PEP 563, PEP 649

from importlib.resources import files
from typing import TYPE_CHECKING

import numpy as np

from ..utils._checks import check_type, check_value, ensure_int, ensure_path
from ..utils._imports import import_optional_dependency
from ..utils.logs import logger
from ._utils import parse_trial_list

if TYPE_CHECKING:
    from typing import Union

    from psychopy.sound.backend_ptb import SoundPTB
    from psychopy.visual import ShapeStim, Window


_TRIAL_LIST_MAPPING = {
    "100": "trialList_1_100_vol-eval-%s.txt",
    "600": "trialList_2_600_game-eval.txt",
    "0a": "trialList_3_360_oddball-%s-a.txt",
    "0b": "trialList_3_360_oddball-%s-b.txt",
    "a": "trialList_4_1000_oddball-game-a.txt",
    "b": "trialList_4_1000_oddball-game-b.txt",
}
_DURATION_STIM: float = 0.2  # seconds
_DURATION_ITI: float = 1.0  # seconds
_DURATION_FLICKERING: float = 0.05  # seconds
_TRIGGERS: dict[str, int] = {
    "standard": 1,
    "target": 2,
    "novel": 3,
}
_SCREEN_SIZE = (1920, 1080)
_BACKGROUND_COLOR = (0, 0, 0)  # (r, g, b) between -1 and 1
_CROSS_WIDTH: int = 3  # pixels
_CROSS_LENGTH: int = 8  # pixels
_CROSS_COLOR: str = "white"
_CROSS_FLICKERING_COLOR: str = "orange"

# check the variables
check_type(_DURATION_STIM, ("numeric",), "_DURATION_STIM")
check_type(_DURATION_ITI, ("numeric",), "_DURATION_ITI")
check_type(_DURATION_FLICKERING, ("numeric",), "_DURATION_FLICKERING")
assert 0.3 < _DURATION_ITI - _DURATION_STIM - 0.2
assert all(elt in _TRIGGERS for elt in ("standard", "target", "novel"))
_CROSS_WIDTH = ensure_int(_CROSS_WIDTH, "_CROSS_WIDTH")
_CROSS_LENGTH = ensure_int(_CROSS_LENGTH, "_CROSS_LENGTH")


def oddball(condition: str, active: bool = True, mock: bool = False) -> None:
    """Run the oddball paradigm.

    Parameters
    ----------
    condition : "100" | "600" | "0a" | "0b" | "a" | "b"
        Oddball condition to run.
    active : bool
        Only used for the condition "100", "0a" and "0b". If False, the oddball is
        passive and the participant is asked to count the flickering on a fixation cross
        while if True, the participant is asked to respond to stimuli physically with a
        button press.
    mock : bool
        If True, uses a MockTrigger instead of a ParallelPortTrigger.
    """
    import_optional_dependency("byte_triggers")
    import_optional_dependency("psychopy")
    import_optional_dependency("psychtoolbox")

    import psychtoolbox as ptb
    from byte_triggers import LSLTrigger, MockTrigger, ParallelPortTrigger
    from psychopy.core import wait
    from psychopy.visual import Window

    check_type(condition, (str,), "condition")
    check_value(condition, _TRIAL_LIST_MAPPING, "condition")
    check_type(active, (bool,), "active")
    check_type(mock, (bool,), "mock")
    # load trials and sounds
    fname = _TRIAL_LIST_MAPPING[condition]
    if condition in ("100", "0a", "0b"):
        fname = fname % ("active" if active else "passive")
    fname = files("eeg_flow.oddball") / "trialList" / fname
    fname = ensure_path(fname, must_exist=True)
    trials = parse_trial_list(fname)
    sounds = _load_sounds(trials)
    # prepare triggers
    trigger = MockTrigger() if mock else ParallelPortTrigger("/dev/parport0")
    trigger_lsl = LSLTrigger("Oddball_task")
    # prepare fixation cross window
    win = Window(
        size=_SCREEN_SIZE,
        color=_BACKGROUND_COLOR,
        units="norm",
        winType="pyglet",
        fullscr=True,
        allowGUI=False,
        screen=1,
    )
    crosses = _load_cross(win, active)
    if not active:
        rng = np.random.default_rng()
    # display fixation cross
    crosses["full"].setAutoDraw(True)
    win.flip()
    input(">>> Press ENTER to start.")
    # main loop
    for i, (k, trial) in enumerate(trials):
        if trial == "cross":  # already handled in the last iteration
            continue
        logger.info("Trial %i / %i: %s", k, trials[-1][0], trial)
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
        trigger_lsl.signal(value)
        # look ahead for the next trial and handle it now if it's a cross for the
        # passive oddball task
        if i == len(trials) - 1:  # end of the trial list
            continue
        elif trials[i + 1][1] == "cross":
            assert not active, f"Trial 'cross' ({k}) found in an active trial-list."
            logger.info("Flickering the fixation cross for trial %i.", k)
            # pick a random time at which the flickering will occur
            delay = rng.uniform(0.3, _DURATION_ITI - _DURATION_STIM - 0.2)
            arm = rng.choice(("top", "left", "right", "bottom"))
            wait(delay)
            crosses["full"].setAutoDraw(False)
            for shape in crosses[arm]:
                shape.setAutoDraw(True)
            win.flip()
            wait(_DURATION_FLICKERING)
            crosses["full"].setAutoDraw(True)
            for shape in crosses[arm]:
                shape.setAutoDraw(False)
            win.flip()
            wait(_DURATION_ITI - _DURATION_STIM - _DURATION_FLICKERING - delay)
        else:
            wait(_DURATION_ITI - _DURATION_STIM)
    wait(1)  # wait before closing
    input(">>> Press ENTER to continue and close the window.")
    win.close()


def _load_sounds(trials) -> dict[str, SoundPTB]:
    """Create psychopy sound objects."""
    from psychopy.sound.backend_ptb import SoundPTB

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


def _load_cross(
    win: Window, active: bool
) -> dict[str, Union[ShapeStim, tuple[ShapeStim, ShapeStim]]]:
    """Load the shapes used for fixation cross.

    The point position is defined as:

            0  11



     2      1  10     9

     3      4  7      8



            5  6
    """
    from psychopy.visual import ShapeStim

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
            [width[0] / 2 + length[0], -width[1] / 2],  # right-bottom
            [width[0] / 2 + length[0], width[1] / 2],  # right-up
            [width[0] / 2, width[1] / 2],  # center-right-up
            [width[0] / 2, width[1] / 2 + length[1]],  # top-right
        ]
    )

    # full-cross
    crosses["full"] = ShapeStim(
        win,
        units="norm",
        lineColor=None,
        fillColor=_CROSS_COLOR,
        vertices=points,
    )

    if active:
        return crosses  # exit early

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
