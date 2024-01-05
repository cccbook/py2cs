# 參考 -- https://www.geeksforgeeks.org/python-vlc-mediaplayer-getting-media
import vlc
import time

media_player = vlc.MediaPlayer()
media = vlc.Media("test.mpg")
media_player.set_media(media)
media_player.play()
time.sleep(100)
print("Media : ", media_player.get_media())
