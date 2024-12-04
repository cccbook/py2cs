from tkinter import Tk, Label, Button

root = Tk()

# 創建元件
label1 = Label(root, text="元件1")
label2 = Label(root, text="元件2")
button = Button(root, text="按鈕")

# 使用pack佈局方式依次放置元件
label1.pack()
label2.pack()
button.pack()

root.mainloop()
