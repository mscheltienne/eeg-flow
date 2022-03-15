from autoreject import get_rejection_threshold
import mne
import numpy as np

from eeg_flow.io import load_xdf, find_streams, create_raw
from eeg_flow.bad_channels import PREP_bads_suggestion


#%% Load from .xdf
fname = '.xdf'

streams = load_xdf(fname)
eeg_stream = find_streams(streams, 'eego')[0][1]
raw = create_raw(eeg_stream)

#%% Load from .fif
fname = '-raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)

#%% Set AUX channels' names and types
mne.rename_channels(
    raw.info, {'AUX7': 'ECG',
               'AUX8': 'hEOG',
               'EOG': 'vEOG',
               'AUX4': 'EDA'})
# AUX5 to be defined
raw.set_channel_types(mapping={'ECG': 'ecg', 'vEOG': 'eog', 'hEOG': 'eog'})

#%% Preprocess
raw.pick_types(stim=True, eeg=True, eog=True, ecg=True)
# bandpass filter
raw.filter(
    l_freq=1.,
    h_freq=40.,
    picks='eeg',
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge")

# bandpass filter + notch on aux
raw.filter(
    l_freq=1.,
    h_freq=40.,
    picks=['eog', 'ecg'],
    method="fir",
    phase="zero-double",
    fir_window="hamming",
    fir_design="firwin",
    pad="edge")

#%% Add ref channel and montage
raw.add_reference_channels(ref_channels='CPz')
raw.set_montage('standard_1020')  # only after adding ref channel
# Retrieve EEG channels without CPz
picks = mne.pick_types(raw.info, eeg=True, exclude=['CPz'])

#%% Search for bads
bads = PREP_bads_suggestion(raw)
raw.info['bads'] = [bad for bad in bads if bad != 'CPz']

#%% Plot
raw.plot()

#%% ICA
ica = mne.preprocessing.ICA(method='picard', max_iter='auto',
                            n_components=0.99)
ica.fit(raw, picks=picks)

#%% Plot
ica.plot_components(inst=raw)
ica.plot_sources(raw)

#%% Apply
ica.apply(raw)

#%% Interpolations
raw.interpolate_bads(reset_bads=True, mode='accurate')

#%% Reference
raw.set_eeg_reference(ref_channels=['M1', 'M2'], projection=False,
                      ch_type='eeg')
raw.drop_channels(['M1', 'M2'])

#%% Epochs
events = mne.find_events(raw, stim_channel='TRIGGER')
events_id = dict(standard=1, target=2, novel=3)
epochs = mne.Epochs(raw, events, events_id, tmin=-0.2, tmax=0.5,
                    baseline=(None, 0), reject=None, picks='eeg',
                    preload=True)
reject = get_rejection_threshold(epochs, decim=1)
epochs.drop_bad(reject=reject)

#%% Plot
epochs.plot_drop_log()

#%% Evoked
evoked_standard = epochs['standard'].average()
evoked_target = epochs['target'].average()
evoked_novel = epochs['novel'].average()

#%% Plot
evoked_standard.plot_joint(picks='eeg', title='Standard')
evoked_target.plot_joint(picks='eeg', title='Target')
evoked_novel.plot_joint(picks='eeg', title='Novel')
