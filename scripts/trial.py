import psychtoolbox as ptb
from bsl.triggers import ParallelPortTrigger
from psychopy.clock import wait
from psychopy.sound.backend_ptb import SoundPTB as Sound

from eeg_flow.config import load_triggers

# Read trial list file
filename_trialList = "trialList/1000-trialListA.txt"
trials = []
with open(filename_trialList) as f:
    for line in f:
        trials.append(line.split()[1])

# Create sounds
sound_standard = Sound(
    "snd/low_tone-48000.wav", blockSize=32, preBuffer=-1, hamming=True
)
sound_target = Sound(
    "snd/high_tone-48000.wav", blockSize=32, preBuffer=-1, hamming=True
)
sounds = dict(standard=sound_standard, target=sound_target)
tdef = load_triggers()
trigger_values = dict(standard=tdef.standard, target=tdef.target)

for trial in trials:
    if trial in ("standard", "target"):
        continue
    sounds[trial] = Sound(
        f"snd/{trial}-48000.wav", blockSize=32, preBuffer=-1, hamming=True
    )
    trigger_values[trial] = tdef.novel

# Create trigger
trigger = ParallelPortTrigger("/dev/parport0")

# Inter-trial delay
delay = 1  # seconds
scheduling_delay = 0.2  # seconds

# Main loop
input("Press any key to start..")

for trial in trials:
    sounds[trial].play(when=ptb.GetSecs() + scheduling_delay)
    wait(scheduling_delay, hogCPUperiod=scheduling_delay + 1)
    trigger.signal(trigger_values[trial])
    wait(delay - scheduling_delay, hogCPUperiod=delay)
