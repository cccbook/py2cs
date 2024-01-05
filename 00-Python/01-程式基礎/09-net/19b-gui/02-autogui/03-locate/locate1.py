import pyautogui
b = pyautogui.locateOnScreen('file.png')
print('b=', b)
pyautogui.click(b)
# 或者直接用 pyautogui.click('file.png')