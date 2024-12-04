import pyttsx3
engine = pyttsx3.init()
# text = "Just do it. You can make it."
text = "盡管做吧！你做得到的！"

# 直接說出來
engine.say(text)
engine.runAndWait()

# 存到檔案
engine.save_to_file(text, 'tts1.mp3')
engine.runAndWait()