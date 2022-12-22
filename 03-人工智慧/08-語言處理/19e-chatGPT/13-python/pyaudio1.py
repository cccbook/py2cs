import pyaudio
import wave

# 打開音頻文件
wf = wave.open("/media/file_example_WAV_1MG.wav", "rb")

# 創建pyaudio對象
p = pyaudio.PyAudio()

# 打開音頻流
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# 讀取音頻數據並播放
data = wf.readframes(1024)
while data != b'':
    stream.write(data)
    data = wf.readframes(1024)

# 關閉音頻流
stream.stop_stream()
stream.close()

# 關閉pyaudio對象
p.terminate()
