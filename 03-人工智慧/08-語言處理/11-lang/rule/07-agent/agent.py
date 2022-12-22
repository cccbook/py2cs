import sys
import webbrowser
import re
import urllib.parse

qa_list = [
{ 'Q':'瀏覽器|網路|上網', 'A': 'https://google.com/'},
{ 'Q':'股票|價錢|幾塊', 'A': 'https://tw.stock.yahoo.com/'},
{ 'Q':'地圖|怎麼去', 'A': 'https://www.google.com/maps'},
{ 'Q':'郵件|信', 'A': 'https://mail.google.com/'},
{ 'Q':'天氣|下雨|溫度', 'A': 'https://news.pchome.com.tw/weather/taiwan'},

]

def answer(say):
	url = "https://www.google.com/search?q="+urllib.parse.quote(say)
	for qa in qa_list: # 對於每一個 QA
		qList = qa['Q'].split("|")
		a = qa['A']
		for q in qList:
			if q.strip() == "": # 如果是最後一個「空字串」的話，那就不用比對，直接任選一個回答。
				break
			m = re.search("(.*)"+q.strip()+"([^?.;]*)", say)
			if m: # 比對成功的話
				url = a
				break
	return url

def agent():
	print('你好，請問你想做甚麼？')
	while (True):
		say = input('> ') # 取得使用者輸入的問句。
		if say == 'bye':
			break
		url = answer(say)
		print(url)
		webbrowser.open_new(url)

agent()
