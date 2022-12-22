from selenium import webdriver
import time

url = 'https://www.dcard.tw/f'
browser = webdriver.Chrome()
browser.get(url)
time.sleep(2)
html = browser.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
print('html=', html)
