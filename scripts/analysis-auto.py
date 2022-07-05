from itertools import chain

import mne
from autoreject import get_rejection_threshold
from mne.io.constants import FIFF
from mne.preprocessing import compute_bridged_electrodes
from mne_icalabel import label_components

from eeg_flow.bad_channels import PREP_bads_suggestion
from eeg_flow.config import load_triggers
from eeg_flow.io import create_raw, find_streams, load_xdf

#%% Load from .xdf
fname = ".xdf"

streams = load_xdf(fname)
eeg_stream = find_streams(streams, "eego")[0][1]
raw = create_raw(eeg_stream)

#%% Load from .fif
fname = "-raw.fif"
raw = mne.io.read_raw_fif(fname, preload=True)

#%% Set AUX channels' names and types
mne.rename_channels(
    raw.info, {"AUX7": "ECG", "AUX8": "hEOG", "EOG": "vEOG", "AUX4": "EDA"}
)
# AUX5 to be defined
raw.set_channel_types(mapping={"ECG": "ecg", "vEOG": "eog", "hEOG": "eog"})

#%% Create and prepare raw for mastoids
raw_mastoids = raw.copy()
raw_mastoids.pick_channels(["M1", "M2"])
# Apply final filter
raw_mastoids.filter(
    l_freq=0.5,
    h_freq=100.0,
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
# Trick MNE in thinking that a custom-ref has been applied
with raw_mastoids.info._unlock():
    raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

#%% Select channels from raw on which the ICA is applied
raw.drop_channels(["M1", "M2"])
raw.pick_types(stim=True, eeg=True, eog=True, ecg=True)

#%% Prepare raw on which the ICA is fitted
raw_ica_fit = raw.copy()
raw_ica_fit.filter(
    l_freq=1.0,
    h_freq=100.0,
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
# Look for bridged electrodes
bridged_idx, _ = compute_bridged_electrodes(raw_ica_fit)
raw_ica_fit.info["bads"].extend(
    [
        ch
        for k, ch in enumerate(raw_ica_fit.ch_names)
        if k in set(chain(*bridged_idx))
    ]
)
# Search for bads with PREP pipeline (excludes bridged electrodes)
bads = PREP_bads_suggestion(raw_ica_fit)
raw_ica_fit.info["bads"].extend(bads)
# Add reference and montage
raw_ica_fit.add_reference_channels(ref_channels="CPz")
raw_ica_fit.set_montage("standard_1020")
# Apply CAR
raw_ica_fit.set_eeg_reference("average", projection=False)

#%% Fit ICA and detect bad components
picks = mne.pick_types(raw_ica_fit.info, eeg=True, exclude="bads")
ica = mne.preprocessing.ICA(
    n_components=picks.size - 1,
    method="picard",
    max_iter="auto",
    fit_params=dict(ortho=False, extended=True),
)
ica.fit(raw_ica_fit, picks=picks)

# label components
component_dict = label_components(raw_ica_fit, ica, method="iclabel")
ica.exclude = [
    k for k, name in enumerate(component_dict["labels"]) if name != "brain"
]

#%% Prepare original raw on which the ICA is applied
raw.filter(
    l_freq=0.5,
    h_freq=100.0,
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
raw.info["bads"] = raw_ica_fit.info["bads"]
raw.add_reference_channels(ref_channels="CPz")
raw.set_montage("standard_1020")
raw.set_eeg_reference("average", projection=False)

#%% Apply ICA to original raw
ica.apply(raw)

#%% Change the reference from CAR to average of both mastoids
raw.set_eeg_reference(["CPz"], projection=False)
raw.add_channels([raw_mastoids])
raw.set_montage("standard_1020")  # add montage for M1 and M2
raw.set_eeg_reference(["M1", "M2"])

#%% Create stimuli-locked epochs
events = mne.find_events(raw, stim_channel="TRIGGER")
events_id = load_triggers().by_name
epochs = mne.Epochs(
    raw,
    events,
    events_id,
    tmin=-0.2,
    tmax=0.5,
    baseline=(None, 0),
    reject=None,
    picks="eeg",
    preload=True,
)
# get_rejection_threshold excludes bad channels
reject = get_rejection_threshold(epochs, decim=1, ch_types="eeg")
epochs.drop_bad(reject=reject)
epochs.plot_drop_log()

#%% Create stimuli-locked evoked response
evoked_standard = epochs["standard"].average()
evoked_target = epochs["target"].average()
evoked_novel = epochs["novel"].average()

#%% Plot
evoked_standard.plot_joint(picks="eeg", title="Standard")
evoked_target.plot_joint(picks="eeg", title="Target")
evoked_novel.plot_joint(picks="eeg", title="Novel")
