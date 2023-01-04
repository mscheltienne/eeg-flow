"""Test bridged.py"""

from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from mne.io import read_raw_fif

from eeg_flow.viz import plot_bridged_electrodes

raw = read_raw_fif(
    Path(__file__).parent / "data" / "test-bridged.fif", preload=True
)


def test_plot_bridged_electrodes():
    """Test plotting from a raw object."""
    f, ax = plot_bridged_electrodes(raw)
    plt.close(f)

    with pytest.raises(TypeError, match="must be an instance of BaseRaw"):
        plot_bridged_electrodes(101)

    raw_ = raw.copy().filter(1, None)
    with pytest.raises(RuntimeError, match="should not be highpass-filtered"):
        plot_bridged_electrodes(raw_)
    raw_ = raw.copy().filter(None, 10)
    with pytest.raises(RuntimeError, match="should not be lowpass-filtered"):
        plot_bridged_electrodes(raw_)
