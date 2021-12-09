from pathlib import Path

from bsl.utils import Timer
from psychopy import parallel, core
from eeg_flow.audio import Tone, Sound


filename_trialList = "trialList/1000-trialListA.txt"

# read trial list file
trials = [];
with open(filename_trialList) as f:
    for line in f:
       trials.append(line.split()[1])

sound_standard = Sound('snd/low_tone-48000.wav')
sound_target = Sound('snd/high_tone-48000.wav')
sounds = dict(standard=sound_standard, target=sound_target)
trigger_values = dict(standard=1, target=2)

for trial in trials:
    if trial in ('standard', 'target'):
        continue
    sounds[trial] = Sound(f'snd/{trial}-48000.wav')
    trigger_values[trial] = 3

port = parallel.ParallelPort(address='/dev/parport0')
port.setData(0)

input('Press any key to start..')
events = list(range(1, 1001, 1))
timer = Timer()
while True:
    if timer.sec() >= events[0]:
        ev = events.pop(0)
        sounds[trials[ev-1]].play()
        port.setData(trigger_values[trials[ev-1]])
        core.wait(0.01)
        port.setData(0)
