from tkinter import Tk, Label, Button

root = Tk()

# 創建元件
label1 = Label(root, text="元件1")
label2 = Label(root, text="元件2")
button = Button(root, text="按鈕")

# 使用grid佈局方式放置元件，指定行和列
label1.grid(row=0, column=0)
label2.grid(row=0, column=1)
button.grid(row=1, column=0, columnspan=2)  # 指定元件佔據的列數

root.mainloop()
