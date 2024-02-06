"""Test bridged.py"""

from importlib.resources import files

import pytest
from matplotlib import pyplot as plt
from mne.io import read_raw_fif

from eeg_flow.viz import plot_bridged_electrodes


@pytest.fixture(scope="module")
def raw():
    """Load a raw object."""
    fname = files("eeg_flow.viz.tests") / "data" / "test-bridged.fif"
    return read_raw_fif(fname, preload=True)


def test_plot_bridged_electrodes(raw):
    """Test plotting from a raw object."""
    f, ax = plot_bridged_electrodes(raw)

    with pytest.raises(TypeError, match="must be an instance of BaseRaw"):
        plot_bridged_electrodes(101)
    raw_ = raw.copy().filter(1, None)
    with pytest.raises(RuntimeError, match="should not be highpass-filtered"):
        plot_bridged_electrodes(raw_)
    raw_ = raw.copy().filter(None, 10)
    with pytest.raises(RuntimeError, match="should not be lowpass-filtered"):
        plot_bridged_electrodes(raw_)

    plt.close("all")
