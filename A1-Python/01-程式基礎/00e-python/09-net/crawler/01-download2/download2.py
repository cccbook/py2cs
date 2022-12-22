import requests
from bs4 import BeautifulSoup

def main():
    try:
        resp = requests.get('http://blog.castman.net/web-crawler-tutorial/ch1/connect.html')
    except:
        resp = None

    if resp and resp.status_code == 200:
        print('status_code=', resp.status_code)
        # print(resp.text)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # print(soup)
        try:
            h1 = soup.find('h1')
        except:
            h1 = None
        if h1:
            print(soup.find('h1'))
            print(soup.find('h1').text)
            print(h1.text)
        try:
            h2 = soup.find('h2')
        except:
            h2 = None
        if h2:
            print(soup.find('h2').text)
        else:
            print('h2 tag not found!')

if __name__ == '__main__':
    main()