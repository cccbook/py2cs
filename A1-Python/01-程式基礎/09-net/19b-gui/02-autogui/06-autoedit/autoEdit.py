import pyautogui
import os
import time

text = ''
with open('message.txt',  encoding='utf8') as f:
    text = f.read()

os.system('start hello.txt')
time.sleep(2)
b = pyautogui.locateOnScreen('chineseSwitch.png')
print('b=', b)
if b is not None:
    pyautogui.click(b)
    pyautogui.hotkey('ctrl', 'W')
# pyautogui.press('shift')
time.sleep(2)
pyautogui.typewrite(list(text), interval=0.001)
