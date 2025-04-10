from gtts import gTTS
from io import BytesIO

mp3_fp = BytesIO()
tts = gTTS('Hello , How are you! 你好嗎?')
tts.write_to_fp(mp3_fp)
tts.save('hello.mp3')
# playsound('hello.mp3')