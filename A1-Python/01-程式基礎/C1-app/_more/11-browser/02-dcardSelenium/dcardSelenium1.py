from selenium import webdriver
import time

url = 'https://www.dcard.tw/f'
browser = webdriver.Chrome()
browser.get(url)
html = browser.page_source
time.sleep(2)
print('html=', html)
