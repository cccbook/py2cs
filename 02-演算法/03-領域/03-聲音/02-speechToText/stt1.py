import speech_recognition as sr

# lang = "en-US"; AUDIO_FILE = '../data/voice1.mp3'
lang = "zh-TW"; AUDIO_FILE = '../data/chinese_tts1.mp3' # 失敗
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file

# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    # result = r.recognize_google(audio)
    result = r.recognize_google(audio, language="zh-TW")
    print("Google Speech Recognition thinks you said :\n-- " + result)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
