from tkinter import Tk, Label, Button

root = Tk()

# 創建元件
label1 = Label(root, text="元件1")
label2 = Label(root, text="元件2")
button = Button(root, text="按鈕")

# 使用place佈局方式指定元件的位置
label1.place(x=50, y=50)
label2.place(x=100, y=50)
button.place(x=50, y=100, width=100, height=30)  # 指定元件的尺寸

root.mainloop()
