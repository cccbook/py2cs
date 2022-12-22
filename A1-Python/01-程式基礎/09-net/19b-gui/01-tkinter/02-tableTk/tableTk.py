from tkinter import ttk
import tkinter as tk
import sqlite3

root = tk.Tk()

textSql = tk.Text(root, height=3, width=100)
textSql.pack()
textSql.insert(tk.END, "SELECT * FROM stockPrice")

button1 = tk.Button(text="查詢")
button1.pack(pady=10)

tree = ttk.Treeview(root, column=("c1", "c2", "c3", "c4"), show='headings')
tree.column("#1", anchor=tk.CENTER)
tree.heading("#1", text="日期")
tree.column("#2", anchor=tk.CENTER)
tree.heading("#2", text="開盤")
tree.column("#3", anchor=tk.CENTER)
tree.heading("#3", text="收盤價")
tree.column("#4", anchor=tk.CENTER)
tree.heading("#4", text="成交筆數")
tree.pack()

root.mainloop()