import requests

try:
    resp = requests.get('http://blog.castman.net/web-crawler-tutorial/ch1/connect.html')
except:
    resp = None

if resp and resp.status_code == 200: # 下載成功
    print('status_code=', resp.status_code)
    print(resp.text)
