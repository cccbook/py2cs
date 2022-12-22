import vlc
import time
p = vlc.MediaPlayer("voice1.mp3")
p.play()
time.sleep(10)
p.stop()