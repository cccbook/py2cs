### Basic usage
import music as M, numpy as n
T = M.tables.Basic()
H = M.utils.H


# 1) start a ѕynth
b = M.core.Being()

# 2) set its parameters using sequences to be iterated through
b.d_ = [1/2, 1/4, 1/4]  # durations in seconds
b.fv_ = [0, 1,5,15,150,1500,15000]  # vibrato frequency
b.nu_ = [5]  # vibrato depth in semitones (maximum deviation of pitch)
b.f_ = [220, 330]  # frequencies for the notes

# 3) render the wavfile
b.render(30, 'aMusicalSound.wav')  # render 30 notes iterating though the lists above