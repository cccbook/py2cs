import eliza
import sounddevice as sd
import wavio as wv
import speech_recognition as sr

def voiceInput(i):
    freq = 44100
    duration = 3
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    wv.write(f"input{i}.wav", recording, freq, sampwidth=2)
    lang = "en-US"; AUDIO_FILE = f'input{i}.wav'
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)
    try:
        result = r.recognize_google(audio, language=lang)
    except:
        result = None
    return result

eliza = eliza.Eliza()
eliza.load('doctor.txt')

print(eliza.initial())
# while True:
for i in range(100):
    # said = input('> ')
    while True:
        print("> ", end="")
        said = voiceInput(i)
        print(said)
        if said is not None:
            break
    response = eliza.respond(said)
    if response is None:
        break
    print(response)
print(eliza.final())