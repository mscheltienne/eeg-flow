# ########################################
# Modified on Mon May 08 01:01:00 2023
# @anguyen

from datetime import datetime
from matplotlib import pyplot as plt  # viz
import numpy as np

from mne import read_evokeds

from ..config import load_config
from ..utils._docs import fill_doc
from ..utils.bids import get_fname, get_folder


@fill_doc
def prep_evoked(
    participant: str,
    group: str,
    task: str,
    run: int,
    *,
    timeout: float = 10,
) -> None:
    """Load the necessary files for the plots.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    """
    # prepare folders
    _, DERIVATIVES_FOLDER_ROOT, _ = load_config()
    FNAME_STEM = get_fname(participant, group, task, run)
    DERIVATIVES_SUBFOLDER = get_folder(
        DERIVATIVES_FOLDER_ROOT, participant, group, task, run
    )

    # load previous steps
    fname_ev_standard = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step7_d1-standard_evoked-ave.fif"
    )
    fname_ev_target = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step7_d2-target_evoked-ave.fif"
    )
    fname_ev_novel = DERIVATIVES_SUBFOLDER / (
        FNAME_STEM + "_step7_d3-novel_evoked-ave.fif"
    )
    all_evokeds = dict()

    all_evokeds["standard"] = read_evokeds(fname_ev_standard, condition=0)
    all_evokeds["target"] = read_evokeds(fname_ev_target, condition=0)
    all_evokeds["novel"] = read_evokeds(fname_ev_novel, condition=0)

    return DERIVATIVES_SUBFOLDER, FNAME_STEM, all_evokeds


@fill_doc
def find_ylim(all_evokeds):
    """Find the appropriate ylim for all given the min and max of each stim.

    Parameters
    ----------
    %(participant)s
    %(group)s
    %(task)s
    %(run)s
    """
    event_id = dict(standard=1, target=2, novel=3)

    max_peak = False
    min_peak = False

    for event in event_id:
        all_evokeds[event].drop_channels(
            all_evokeds[event].info["bads"]
        )  # remove bads from this report)
        event_max = np.amax(all_evokeds[event].data)
        event_min = np.amin(all_evokeds[event].data)

        if not max_peak or max_peak < event_max:
            max_peak = event_max
        if not min_peak or min_peak > event_min:
            min_peak = event_min

    min_peak *= 10**6
    max_peak *= 10**6

    print(min_peak, max_peak)

    ylim_values = {}

    for event in event_id:
        event_max = np.amax(all_evokeds[event].data)
        event_min = np.amin(all_evokeds[event].data)
        ylim_values[event] = [
            event_max * 1.2 * 10**6,
            event_min * 1.2 * 10**6,
        ]
    return ylim_values, min_peak, max_peak


@fill_doc
def plot_fixed_scale(
    DERIVATIVES_SUBFOLDER,
    FNAME_STEM,
    all_evokeds,
    new_min_peak=None,
    new_max_peak=None,
):
    """Plot ERPs with the same scale for all three stims.

    Parameters
    ----------
    %(DERIVATIVES_SUBFOLDER)s
    %(FNAME_STEM)s
    %(all_evokeds)s
    %(new_min_peak)s
    %(new_max_peak)s
    """
    ylim_values, min_peak, max_peak = find_ylim(all_evokeds)

    # %%
    # # use this if limits need to be set manually
    if (new_min_peak is not None) and (new_max_peak is not None):
        max_peak = new_max_peak
        min_peak = new_min_peak

    f, ax = plt.subplots(3, 1, figsize=(8, 8))
    f.suptitle(
        f"{FNAME_STEM} | All evoked | All channels | Fixed scale | ({str(min_peak)}, {str(max_peak)})"
    )

    ylim = dict(eeg=[max_peak * 1.2, min_peak * 1.2])

    for k, (condition, evo) in enumerate(all_evokeds.items()):
        print(k, condition, evo)
        evo.plot(axes=ax[k], ylim=ylim, zorder="std")
        ax[k].set_title(condition.capitalize())
    f.tight_layout()

    timestampStr = datetime.now().strftime("%Y_%m_%d_%H_%M")
    FNAME_PLOT = (
        DERIVATIVES_SUBFOLDER
        / "plots"
        / (f"{FNAME_STEM}_step7_allElec_fixedScale-{timestampStr}.svg")
    )
    f.savefig(FNAME_PLOT)

    # fig_all_evoked_fixed = f
    # fig_all_evoked_fixed
    return


@fill_doc
def plot_adapt_scale(
    DERIVATIVES_SUBFOLDER,
    FNAME_STEM,
    all_evokeds,
    min_peak=None,
    max_peak=None,
):
    """Plot ERPs with the appropriate scale for each three stims.

    Parameters
    ----------
    %(DERIVATIVES_SUBFOLDER)s
    %(FNAME_STEM)s
    %(all_evokeds)s
    %(new_min_peak)s
    %(new_max_peak)s
    """
    ylim_values, _, _ = find_ylim(all_evokeds)
    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    fig.suptitle(f"{FNAME_STEM} | All evoked | All channels | Adaptive scale")

    ylim_standard = dict(eeg=ylim_values["standard"])
    ylim_target = dict(eeg=ylim_values["target"])
    ylim_novel = dict(eeg=ylim_values["novel"])

    all_evokeds["standard"].plot(
        picks="eeg",
        axes=ax1,
        spatial_colors=True,
        zorder="std",
        ylim=ylim_standard,
        titles="standard",
    )
    all_evokeds["target"].plot(
        picks="eeg",
        axes=ax2,
        spatial_colors=True,
        zorder="std",
        ylim=ylim_target,
        titles="target",
    )
    all_evokeds["novel"].plot(
        picks="eeg",
        axes=ax3,
        spatial_colors=True,
        zorder="std",
        ylim=ylim_novel,
        titles="novel",
    )
    fig.tight_layout()

    timestampStr = datetime.now().strftime("%Y_%m_%d_%H_%M")
    FNAME_PLOT2 = (
        DERIVATIVES_SUBFOLDER
        / "plots"
        / (f"{FNAME_STEM}_step7_allElec_AdaptiveScale-{timestampStr}.svg")
    )
    fig.savefig(FNAME_PLOT2)
    fig
