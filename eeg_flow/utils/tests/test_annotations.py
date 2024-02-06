from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from mne import create_info
from mne.io import RawArray

from eeg_flow.config import load_triggers
from eeg_flow.utils.annotations import annotations_from_events

if TYPE_CHECKING:
    from mne.io import BaseRaw


@pytest.fixture(scope="module")
def triggers():
    """Load triggers."""
    return load_triggers()


@pytest.fixture(scope="module")
def data(triggers: dict[str, int]) -> tuple[BaseRaw, dict[str, float]]:
    """Create fake raw object."""
    rng = np.random.default_rng(101)
    fs = 100  # Hz
    data = np.zeros(fs * 10)
    onsets = dict()
    for k, (key, value) in enumerate(triggers.items()):
        idx = rng.integers(low=0, high=int(data.size - 0.2 * fs), size=k + 1)
        data[idx] = value
        onsets[key] = idx / fs
    info = create_info(["TRIGGER"], fs, "stim")
    return RawArray(data.reshape(1, -1), info), onsets


def test_annotations_from_events(data, triggers):
    """Test annotations_from_events."""
    raw, onsets = data  # unpack
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
