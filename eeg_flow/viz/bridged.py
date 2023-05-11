# postponed evaluation of annotations, c.f. PEP 563 and PEP 649 alternatively, the type
# hints can be defined as strings which will be evaluated with eval() prior to type
# checking.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from mne.io import BaseRaw, Info
from mne.preprocessing import (
    compute_bridged_electrodes as compute_bridged_electrodes_mne,
)
from mne.utils import check_version
from mne.viz import plot_bridged_electrodes as plot_bridged_electrodes_mne

from ..utils._checks import check_type

if TYPE_CHECKING:
    from typing import List, Tuple

    from matplotlib.pyplot import Axes, Figure
    from numpy.typing import NDArray


def plot_bridged_electrodes(
    raw: BaseRaw,
) -> Tuple[Figure, NDArray[Axes]]:
    """Compute and plot bridged electrodes.

    Parameters
    ----------
    raw : Raw
        MNE Raw instance before filtering. The raw instance is copied, the EEG channels
        are picked and filtered between 0.5 and 30 Hz.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Array of Axes
        Matplotlib subplots.
    """
    check_type(raw, (BaseRaw,), "raw")
    if 0.5 < raw.info["highpass"]:
        raise RuntimeError(
            "The raw instance should not be highpass-filtered above 0.5 Hz."
        )
    if raw.info["lowpass"] < 30:
        raise RuntimeError(
            "The raw instance should not be lowpass-filtered below 30 Hz."
        )
    # retrieve bridge electrodes, operates on a copy
    bridged_idx, ed_matrix = compute_bridged_electrodes_mne(raw)
    # plot
    fig, ax = plot_bridged_electrodes_array(
        bridged_idx, ed_matrix, raw.info.copy().set_montage("standard_1020")
    )
    return fig, ax


# TODO: Why is 'ed_matrix' of shape (n_epochs, n_channels, n_channels) and not
# (n_channels, n_channels)?
def plot_bridged_electrodes_array(
    bridged_idx: List[Tuple[int, int]],
    ed_matrix: NDArray[float],
    info: Info,
) -> Tuple[Figure, NDArray[Axes]]:
    """Pot bridged electrodes.

    Parameters
    ----------
    bridged_idx : list of tuple
        The indices of channels marked as bridged with each bridged pair stored as a
        tuple.
    ed_matrix : ndarray of shape (n_epochs, n_channels, n_channels)
        The electrical distance matrix for each pair of EEG electrodes.
    info : Info
        MNE Info including the montage (location of each electrodes).

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Array of Axes
        Matplotlib subplots.
    """
    check_type(bridged_idx, (list,), "bridged_idx")
    for bridge in bridged_idx:
        check_type(bridge, (tuple,), "bridge")
        assert len(bridge) == 2
    check_type(ed_matrix, (np.ndarray,), "ed_matrix")
    assert ed_matrix.ndim == 3
    check_type(info, (Info,), "info")
    assert info.get_montage() is not None

    # create figure
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # plot electrical distances on the first row
    ed_plot = np.zeros(ed_matrix.shape[1:]) * np.nan
    triu_idx = np.triu_indices(ed_plot.shape[0], 1)
    for idx0, idx1 in np.array(triu_idx).T:
        ed_plot[idx0, idx1] = np.nanmedian(ed_matrix[:, idx0, idx1])

    im1 = ax[0, 0].imshow(ed_plot, aspect="auto")
    cax1 = fig.colorbar(im1, ax=ax[0, 0])
    cax1.set_label(r"Electrical Distance ($\mu$$V^2$)")
    ax[0, 0].set_xlabel("Channel Index")
    ax[0, 0].set_ylabel("Channel Index")

    im2 = ax[0, 1].imshow(ed_plot, aspect="auto", vmax=5)
    cax2 = fig.colorbar(im2, ax=ax[0, 1])
    cax2.set_label(r"Electrical Distance ($\mu$$V^2$)")
    ax[0, 1].set_xlabel("Channel Index")
    ax[0, 1].set_ylabel("Channel Index")

    # plot distribution
    ax[1, 0].hist(ed_matrix[~np.isnan(ed_matrix)], bins=np.linspace(0, 500, 51))
    ax[1, 0].set_xlabel(r"Electrical Distance ($\mu$$V^2$)")
    ax[1, 0].set_ylabel("Count (channel pairs for all epochs)")
    ax[1, 0].set_title("Electrical Distance Matrix Distribution")

    # plot topographic map
    args = dict(vlim=(None, 5)) if check_version("mne", "1.3.0") else dict(vmax=5)
    args["axes"] = ax[1, 1]
    plot_bridged_electrodes_mne(
        info,
        bridged_idx,
        ed_matrix,
        title="Bridged Electrodes",
        topomap_args=args,
    )

    fig.tight_layout()
    return fig, ax
