from musicgen import MusicGen
from utils import save_audio

model = MusicGen.from_pretrained("facebook/musicgen-medium")

audio = model.generate("happy rock")

save_audio("out.wav", audio, model.sampling_rate)