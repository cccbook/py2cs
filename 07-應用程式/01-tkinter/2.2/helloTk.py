import tkinter as tk

root = tk.Tk()
root.title("我的第一個 tkinter 程式")
root.geometry("500x500")

label = tk.Label(root, text="Hello, World!")
label.pack()

button = tk.Button(root, text="點我")
button.pack()

root.mainloop()