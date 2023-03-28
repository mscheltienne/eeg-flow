import numpy as np
from mne import create_info
from mne.io import RawArray

from eeg_flow.config import load_triggers
from eeg_flow.utils.annotations import annotations_from_events


# create fake data
rng = np.random.default_rng()
fs = 100  # Hz
data = np.zeros(fs * 10)
onsets = dict()
triggers = load_triggers()
for k, (key, value) in enumerate(triggers.items()):
    idx = rng.integers(low=0, high=data.size, size=k+1)
    data[idx] = value
    onsets[key] = idx / fs
info = create_info(["TRIGGER"], fs, "stim")
raw = RawArray(data.reshape(1, -1), info)


def test_annotations_from_events():
    """Test annotations_from_events."""
    annots = annotations_from_events(raw)
    assert len(triggers) * (len(triggers) + 1) / 2 == len(annots)
    assert sorted(np.unique(annots.description)) == sorted(triggers.keys())

    # compare onsets
    annots_onsets = {key: [] for key in onsets.keys()}
    for annot in annots:
        assert annot["duration"] == 0.1
        annots_onsets[annot["description"]].append(annot["onset"])
    for key, value in onsets.items():
        assert np.allclose(sorted(value), annots_onsets[key])
