events = mne.find_events(raw, stim_channel='TRIGGER')

if oddball_type == 3:
    events_id = dict(standard=1, target=2, novel=3, response=64)
    row_events = ['standard','target','novel','response']
elif oddball_type == 2:
    events_id = dict(standard=1, target=2, response=64)
    row_events = ['standard','target','response']

# metadata for each epoch shall include events from the range: [0.0, 1.5] s,
# i.e. starting with stimulus onset and expanding beyond the end of the epoch
metadata_tmin, metadata_tmax = 0.0, 0.99

# auto-create metadata
# this also returns a new events array and an event_id dictionary. we'll see
# later why this is important
metadata, events, event_id = mne.epochs.make_metadata(
    events=events, event_id=events_id,
    tmin=metadata_tmin, tmax=metadata_tmax, sfreq=raw.info['sfreq'], row_events=row_events)

# let's look at what we got!
metadata


##########
#check that previous row is target
conditions = [metadata.event_name.shift(periods=1) == "target"]
choices = [True]

metadata['previousIsTarget'] = np.select(conditions, choices, default=False)
metadata['previousIsTarget'].value_counts()

#metadata[10:20]

###############


#check that current row is response
conditions = [metadata['event_name'].eq('response')]
choices = [True]

metadata['currentIsResponse'] = np.select(conditions, choices, default=False)
metadata['currentIsResponse'].value_counts()

#metadata[10:20]

###############################
#check that current row is response and previous is target
conditions = [metadata['previousIsTarget'].eq(True) & metadata['currentIsResponse'].eq(True)]
choices = [True]

metadata['first_correct_response'] = np.select(conditions, choices, default=False)
metadata['first_correct_response'].value_counts()

#metadata
###################
metadata_correct_response_locked = metadata[(metadata.event_name == "response")] 

metadata_correct_response_locked.to_csv(output_data_path + condition + r"\\metadata_correct_response.csv")
metadata_correct_response_locked

#################
mask = (events[:, 2] == 64)
events_response=events[mask, :]
event_id = {'response': 64}

epochs_tmin, epochs_tmax = -0.2, 0.8 

epochs = mne.Epochs(raw=raw, tmin=epochs_tmin, tmax=epochs_tmax,
                    events=events_response, event_id=event_id, metadata=metadata_correct_response_locked,
                    reject=None, preload=True)

#############

#this keeps only the first correct response (and not second response within a trial, nor an incorrect response)
all_evokeds_response = epochs["first_correct_response"].average()
all_evokeds_response