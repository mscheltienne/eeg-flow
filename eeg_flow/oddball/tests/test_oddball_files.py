from importlib.resources import files

from eeg_flow.oddball.oddball import _list_novel_sounds, _parse_trial_list


def test_oddball_files():
    """Test the presence of the oddball files."""
    trials = list()
    for fname in (
        "trialList_1_100_vol-eval.txt",
        "trialList_2_600_game-eval.txt",
        "trialList_3_360_oddball-passive-a.txt",
        "trialList_3_360_oddball-active-a.txt",
        "trialList_3_360_oddball-passive-b.txt",
        "trialList_3_360_oddball-active-b.txt",
        "trialList_4_1000_oddball-game-a.txt",
        "trialList_4_1000_oddball-game-b.txt",
    ):
        trials.extend(
            _parse_trial_list(files("eeg_flow.oddball") / "trialList" / fname)
        )
    trials = list(set([trial[1] for trial in trials if trial[1].startswith("wav")]))
    novel_sounds = [sound.split("-")[0] for sound in _list_novel_sounds()]
    trials = sorted(trials)
    novel_sounds = sorted(novel_sounds)
    assert sorted(trials) == sorted(novel_sounds)
