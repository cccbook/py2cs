import ai2

ai2.open()
response = ai2.chat2("請問 32179 加上 13426 是多少? 兩者相乘又是多少？")

"""
response = ai.chat2("請問 2025 年的美國總統是誰？")
response = ai.chat2("請問桃園中正機場到台北總統府距離多遠？")
response = ai.chat2("請問 1995 年的美國總統是誰？")
response = ai.chat2("請寫一個 Python 程式劃出折線圖")
response = ai.chat2("我後天早上九點要開系務會議")
"""
# response = ai.chat2("我明天要搭華航下午五點的飛機去泰國")
response = ai2.chat2("我下周三早上 10 點要搭華航下午五點的飛機去泰國")

ai2.close()
