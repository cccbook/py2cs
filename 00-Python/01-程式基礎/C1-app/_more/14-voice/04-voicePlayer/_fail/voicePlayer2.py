from pydub import AudioSegment
from pydub.playback import play

# song = AudioSegment.from_wav("../data/voice1.mp3")
song = AudioSegment.from_wav("ex1.wav")
play(song)