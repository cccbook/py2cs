# AI2 -- 呼叫 System2 的 LLM (OpenAI)

2023 年的大語言模型 LLM (像是 ChatGPT) ，擅長預測下一個詞彙 (System 1)，但是不擅長進行深入思考 (System 2) (例如計算大數字加減乘除，或者下圍棋、象棋等等)

但是 LLM 可以呼叫其他專家來回答《深入思考》的 System 2 問題

因此本系統會在自己不擅長回答的問題上，改用呼叫 Python 函數的方式回應。

範例：

* https://chat.openai.com/share/c3ae0480-e56c-433a-8c66-026a024b9dc4

User: 請問第 100 個質數是多少?

ChatGPT: 

    第 100 個質數是 <python>search('100th prime number')</python>。

User: 請問地球到火星的距離是多少?

ChatGPT : 

    讓我們透過地理位置座標計算一下。

    地球的經緯度是 <python>p1=geo_location('地球')</python>

    火星的經緯度是 <python>p2=geo_location('火星')</python>

    現在，我們可以使用 geo_distance() 函數計算地球到火星的距離：<python>geo_distance(p1, p2)</python>。

User: 請寫一個程式可以解決河內塔問題

ChatGPT

當然，我可以為您提供河內塔問題的解決方案。這是一個簡單的 Python 程式：

```py
def hanoi(n, source, target, auxiliary):
    if n > 0:
        # 先將 n-1 個圓盤從 source 移至 auxiliary
        hanoi(n - 1, source, auxiliary, target)
        
        # 移動最大的圓盤至 target
        print(f"Move disk {n} from {source} to {target}")
        
        # 將之前移到 auxiliary 的圓盤再移到 target
        hanoi(n - 1, auxiliary, target, source)

# 設定圓盤數量
number_of_disks = 3

# 呼叫河內塔函數
hanoi(number_of_disks, 'A', 'C', 'B')
```

這個程式將會解決具有指定數量圓盤的河內塔問題。如果您希望使用不同的圓盤數量，只需修改 number_of_disks 的值即可。

User: 請問 37254*22341 是多少

ChatGPT: 

    37254 * 22341 = <python>eval('37254*22341')</python>

User: 請問 37254 乘以 22341 是多少

ChatGPT: 

    37254 乘以 22341 等於 <python>eval('37254*22341')</python>


User: 請把上一個回答翻譯成英文

ChatGPT: 

    37254 multiplied by 22341 equals <python>eval('37254*22341')</python>.

## 後續處理

然後我們就只要呼叫 Python 

    把那些 <python>...</python> 刮起來的內容
    
拿去執行，再把結果貼進去取代就可以了。

