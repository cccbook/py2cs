import whisper

# 加載 Whisper 模型
model = whisper.load_model("base")  # 可選 "tiny", "base", "small", "medium", "large"

# 加載音訊文件進行轉錄
result = model.transcribe("mp3/test.mp3")

# 輸出轉錄結果
print(result["text"])
