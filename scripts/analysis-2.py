import numpy as np
from mne import pick_types, rename_channels
from mne.io import read_raw_fif
from mne.io.constants import FIFF
from mne.preprocessing import (
    ICA,
    compute_bridged_electrodes,
    interpolate_bridged_electrodes,
)
from mne.viz.ica import _prepare_data_ica_properties
from mne_icalabel import label_components
from pyprep import NoisyChannels

from eeg_flow.io import create_raw, find_streams, load_xdf
from eeg_flow.viz import plot_bridged_electrodes

#%% Load recording from .xdf
fname = ".xdf"

streams = load_xdf(fname)
eeg_stream = find_streams(streams, "eego")[0][1]
raw = create_raw(eeg_stream)

#%% Load from .fif
fname = "-raw.fif"
raw = read_raw_fif(fname, preload=True)

#%% Fix channel name/types
# We start by finalizing the loading, if needed:
# - fix the channel names
# - fix the channel types

rename_channels(
    raw.info, {"AUX7": "ECG", "AUX8": "hEOG", "EOG": "vEOG", "AUX4": "EDA"}
)
raw.set_channel_types(mapping={"ECG": "ecg", "vEOG": "eog", "hEOG": "eog"})

#%% Bridges
# At the moment, raw is referenced to CPz and includes all channels (except
# CPz). The first step is to handle the bridged electrodes. This is the first
# step because noise sources, e.g. muscle, blink, .. do not impact bridged
# electrodes. They will anyway record the same signal, the same noise.

# Let's start with a visual inspection of a the bridged electrodes. If it seems
# fixable, i.e. if you don't see huge patches of bridged electrodes, we can
# continue and interpolate the missing data.
plot_bridged_electrodes(raw)

# The interpolation will:
# - look for a group of bridged electrodes, e.g. 3 electrodes bridged together.
# - create a virtual channel at the average location of the group
# - assign to the virtual channel the average data from the 3 bridged channels
#   (note that it assigns the average, because even if they are bridged, the
#    electrical distance is not exactly 0 and some small variation still exist
#    between those 3 channels)
# - mark the 3 bridged channels as bads, in raw.info["bads"]
# - interpolate with raw.interpolate_bads()
# - remove the virtual channels and remove the formerly bridged electrodes from
#   the list of bad channels
# The asset of this method, is that instead of completely disregarding the
# signal on the bridged channels and interpolating only from the neighbor
# channels, it does retain some information from the bridged location.

raw.set_montage("standard_1020")  # we need a montage for the interpolation
bridged_idx, _ = compute_bridged_electrodes(raw)
raw = interpolate_bridged_electrodes(raw, bridged_idx)
raw.set_montage(None)  # we need to remove the montage since CPz is missing

# At this point, raw is referenced to CPz, includes all channels (except CPz),
# does not have any bad channels marked and is free of bridges.

#%% Prepare the future reference
# Next in preprocessing-land, we are going to prepare our future reference: the
# mastoids. The mastoids are often contaminated by muscles and heartbeat
# artifacts, which are well captured by an ICA. Thus, we are going to run an
# ICA to target specifically those artifacts on the mastoids.

#%%% Prepare the raw to fit the ICA
# ICA looks for independent components (IC). To get good IC, we need to:
# - remove low frequencies, which include slow-drifts impacting all electrodes
#   simultaneously, and thus creating a dependency between the channels.
# - remove the bad channels, which will be interpolated from the cleaned
#   channels later on.

raw_ica_fit = raw.copy()
raw_ica_fit.filter(
    l_freq=1.0,
    h_freq=40.0,
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)

# Note that I am not filtering between (1, 100) Hz and I am not changing the
# reference to a common average (CAR). Thus, ICLabel can not be used to suggest
# bad components.
# This is deliberate as we want to remove noise from the mastoid channels only.

ns = NoisyChannels(raw_ica_fit, do_detrend=False)
ns.find_bad_by_SNR()
ns.find_bad_by_correlation()
ns.find_bad_by_hfnoise()
ns.find_bad_by_nan_flat()
ns.find_bad_by_ransac()
raw_ica_fit.info["bads"] = [
    ch for ch in ns.get_bads() if ch not in ("M1", "M2")
]
# Note that I do not include the mastoids in the bad channels even if they are
# detected as such. The goal is to prepare the mastoids that will be used as a
# reference, thus we *need* to fit components on those channels.

# Before we continue, visually inspect the bad channels. You can also annotate
# segments to reject.
raw_ica_fit.plot(theme="light")
# Since the annotations are not going to change drastically, let's move them
# directly to 'raw'.
raw.set_annotations(raw_ica_fit.annotations)

#%%% Fit an ICA
ica = ICA(
    n_components=0.99,
    method="picard",
    max_iter="auto",
    fit_params=dict(ortho=False, extended=True),
)
picks = pick_types(raw_ica_fit.info, eeg=True, exclude="bads")
ica.fit(raw_ica_fit, picks=picks)

#%%% Annotate bad ICs
# At this stage, let's only focus on the mastoids. Look for:
# - heartbeat in the IC-time series
# - muscle/noise on the mastoids on the topographic map

ica.info.set_montage("standard_1020")  # required for topographic maps
ica.plot_sources(raw_ica_fit)
ica.plot_components(inst=raw_ica_fit)

#%%% Filter the mastoids, apply ICA, and keep the mastoids
# Final step in the preparation of our future reference, we need to filter
# those channels to the desired frequencies, and apply the ICA.

raw_mastoids = raw.copy()
raw_mastoids.filter(
    l_freq=0.5,
    h_freq=40.0,
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
ica.apply(raw_mastoids)
raw_mastoids.pick_channels(["M1", "M2"])

# Trick MNE in thinking that a custom-ref has been applied
with raw_mastoids.info._unlock():
    raw_mastoids.info["custom_ref_applied"] = FIFF.FIFFV_MNE_CUSTOM_REF_ON

# At this stage, the reference have been denoised. We have in 'raw_mastoids'
# the 2 mastoids M1 and M2 referenced to CPz. Now, let's clean the rest.

#%% Clean the other channels
# The first step is to prepare the raw object for an ICA, and for suggestions
# from ICLabel. The steps are very similar to the previous ones.
raw.drop_channels(["M1", "M2"])
raw.pick_types(stim=True, eeg=True)

# filter
raw_ica_fit = raw.copy()
raw_ica_fit.filter(
    l_freq=1.0,
    h_freq=100.0,  # Note the higher frequency
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)

# look for bad channels
ns = NoisyChannels(raw_ica_fit, do_detrend=False)
ns.find_bad_by_SNR()
ns.find_bad_by_correlation()
ns.find_bad_by_hfnoise()
ns.find_bad_by_nan_flat()
ns.find_bad_by_ransac()
bads = ns.get_bads()  # we will need this later for interpolation
raw_ica_fit.info["bads"] = bads
# it's unlikely to be different from the previous run on the (1, 40) Hz raw
# including the mastoids, but just in case ;)

# change the reference to a common average reference (CAR)
raw_ica_fit.add_reference_channels(ref_channels="CPz")
raw_ica_fit.set_montage("standard_1020")
raw_ica_fit.set_eeg_reference("average", projection=False)
# Note that the CAR is excluding the bad channels.

# fit an ICA
ica = ICA(
    n_components=0.99,  # can be set to None
    method="picard",
    max_iter="auto",
    fit_params=dict(ortho=False, extended=True),
)
picks = pick_types(raw_ica_fit.info, eeg=True, exclude="bads")
ica.fit(raw_ica_fit, picks=picks)

#%% Label components
# Let's start by getting suggestion from the ICLabel model
component_dict = label_components(raw_ica_fit, ica, method="iclabel")

# let's remove eye-blink and heart beat
labels = component_dict["labels"]
exclude = [
    k
    for k, name in enumerate(labels)
    if name in ("eye blink", "heart beat")
]

# let's remove other non-brain components that occur often
_, _, _, data = _prepare_data_ica_properties(
    raw_ica_fit,
    ica,
    reject_by_annotation=True,
    reject="auto",
)
ica_data = np.swapaxes(data, 0, 1)
var = np.var(ica_data, axis=2)  # (n_components, n_epochs)
var = np.var(var.T / np.linalg.norm(var, axis=1), axis=0)
# linear fit to determine the variance thresholds
z = np.polyfit(range(0, ica.n_components_, 1), var, 1)
threshold = [z[0] * x + z[1] for x in range(0, ica.n_components_, 1)]
# add non-brain ICs below-threshold to exclude
for k, label in enumerate(labels):
    if label in ("brain", "eye blink", "heart beat"):
        continue
    if threshold[k] <= var[k]:
        continue
    exclude.append(k)
ica.exclude = exclude

# Visual inspection
ica.plot_sources(raw_ica_fit)
ica.plot_components(inst=raw_ica_fit)

#%% Apply ICA
# At this stage, we have an ICA decomposition with labeled components. Now, we
# can apply it on the initial raw object, filtered between the final
# frequencies.
# But for this operation to be valid, it needs to be referenced as raw_ica_fit.

raw.info["bads"] = bads  # set the *same* bad channels, to get the same ref.
raw.filter(
    l_freq=0.5,
    h_freq=40.0,
    picks="eeg",
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge",
)
raw.add_reference_channels(ref_channels="CPz")
raw.set_eeg_reference("average", projection=False)
ica.apply(raw)

#%% Rereferenced to mastoids
# At this stage, we have:
# - raw_mastoids, the mastoids cleaned and referenced to CPz
# - raw, the other electrodes, cleaned + bads and referenced to CAR

raw.set_eeg_reference(["CPz"], projection=False)  # change reference back
raw.add_channels([raw_mastoids])
raw.set_montage("standard_1020")  # add montage for M1 and M2
raw.set_eeg_reference(["M1", "M2"])

#%% And we are done.
# We have a raw with bad channels in raw.info["bads"], good channel denoised,
# M1, M2 denoised and set as the reference.
