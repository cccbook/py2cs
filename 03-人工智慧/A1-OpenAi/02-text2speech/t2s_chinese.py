from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "speechChinese.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="訓練深度學習模型已經不是一般人玩得起的技術了，必須有蠻大的資本才行 ..."
)

response.stream_to_file(speech_file_path)
