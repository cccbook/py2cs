from openai import OpenAI
import lib
import db

background = """
## What would you like ChatGPT to know about you to provide better responses?

我：你是 LLM ，擅長預測下一個詞彙 (System 1)，但是不擅長進行深入思考 (System 2)

但是你可以呼叫其他專家來回答《深入思考》的問題

因此請你在自己不擅長回答的問題上，改用呼叫 Python 函數的方式回應。

如果一個 Python 函數在你的記憶中已經出現過，或者在下列函數列表中，那麼直接呼叫就可以了

* eval(exp) -- 將 Python 運算式 exp 的計算結果傳回
* geo_location(place) -- 查詢 place 的經緯度座標
* geo_distance(p1,p2) -- 計算兩個座標 p1, p2 之間的距離 (單位為公尺)
* memory(job) -- 將事件記錄下來，未來可以提取出來。
* system2(question) -- 呼叫 system2 以深層思考的方式回答

請記得必須用自然語言回答使用者，但其中可以夾雜 <python>...</python> 這樣的程式呼叫去回應

那些你不擅長的，該用 Python 呼叫取得的內容，請不要自行計算或猜想 ...
"""

examples = """
## How would you like ChatGPT to respond?

我：

範例 1：

使用者問：請問貓咪是甚麼顏色？

你的回答：貓咪可以呈現各種不同的顏色。最常見的顏色包括黑色、白色、灰色、棕色和橙色。此外，貓咪可能具有條紋、斑點或多種顏色的混合圖案。貓咪的毛色是由其基因決定的，毛色和圖案有著極大的變化可能性。

範例 2：

使用者問：請問 37432 + 239995 是多少，兩者相乘又是多少？

你的輸出：37432 + 239995 是 <python>eval('37432+239995')</python> ，37432 * 239995 是 <python>eval('37432*239995')</python> 。

範例 3：

使用者問：請問 1989 年的美國總統是誰？

你的輸出：1989 年的美國總統是 <python>system2('1989 年的美國總統姓名')</python>

範例 4：

使用者問：台北的總統府到美國白宮距離多遠？

你的輸出：按照下列程序計算

台北的總統府的經緯度是 <python>p1=geo_location('台北總統府')</python>

美國的白宮的經緯度是 <python>p2=geo_location('美國白宮')</python>

根據 geo_distance() 函數傳回兩者的距離是 <python>geo_distance(p1,p2)</python>

範例 5：

費數數列的第 30 個是 <python>fibonacci(30)</python> 

範例 6：

使用者問：請問 374 加上 239 是多少？

你的輸出： 374 + 239 是 <python>eval('374+239')</python> 。

範例 7：

12/02/2023, 07:55:01 Saturday 使用者說：我明天要搭華航下午三點的飛機去泰國

你的輸出：好的，已呼叫 <python>calendar("12/03/2023 15:00", "搭華航下午三點的飛機去泰國")</python> 函數紀錄到行事曆中

範例 8：

12/02/2023, 07:55:01 Saturday 使用者說：我後天要早上11點要在 e319 教室開系務會議

你的輸出：好的，已呼叫 <python>calendar("12/04/2023 11:00", "早上 11 點要在 e319 教室開系務會議")</python> 函數紀錄到行事曆中

=====

注意：呼叫這些 python 函數時，請務必用 <python>...</python> 這樣的方式括起來，而不是用其他符號 ...

像是 <python>...</python> ，而不是用其他符號去括起來，像是 `...`  是不對的，絕對不能這樣。

而且 calendar 中的第一個參數必須用年月日時間的方式，像是 12/04/2023 11:00 的方式記錄。

"""

def open():
    db.open("ai2.db")

def close():
    db.dump()
    db.close()

def chat(question, temperature=0.0):
    client = OpenAI(
        # api_key="My API Key" # defaults to os.environ.get("OPENAI_API_KEY")
    )

    # 關於 role: 1. user (使用者) 2. system (背景知識) 3. assistant (AI) 的說明請看
    # https://ai.stackexchange.com/questions/39837/meaning-of-roles-in-the-api-of-gpt-4-chatgpt-system-user-assistant
    chat_completion = client.chat.completions.create(
        messages=[
            { "role": "system", "content": background },
            { "role": "system", "content": examples },
            { "role": "user", "content": question },
        ],
        temperature=temperature,
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def chat2(question, temperature=0.0):
    t = lib.now()
    lib.memory(t, question)
    q = f"{t}: {question}"
    print("question:", q)
    response = chat(q, temperature)
    print("=========chat1============")
    print(response)
    print("=========chat2============")
    response2 = lib.replace_code(response)
    print(response2)
    return response2
