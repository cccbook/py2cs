import re
import json

# 您的输入字符串
input_string = "这是一些文本，然后是一个嵌套的JSON对象：{'name': 'John', 'info': {'age': 30, 'city': 'New York'}}, 再跟一些其他文本。"

# 定義匹配 JSON 對象的正規表達式
pattern = r'\{.*?\}'

# 使用正規表達式尋找匹配
matches = re.findall(pattern, input_string)

# 如果有多個 JSON 對象匹配，可以遍歷它們
for match in matches:
    # 使用 JSON 解碼解析 JSON 對象
    try:
        print('match=', match)
        json_object = json.loads(match)
        # 現在，json_object 包含解析後的 JSON 對象
        print(json_object)
    except json.JSONDecodeError:
        # 如果 JSON 解碼失敗，您可以處理錯誤或者跳過該對象
        print("JSON 解碼失敗")

# 輸出解析後的 JSON 對象
