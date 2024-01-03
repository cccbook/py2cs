from tkinter import ttk
import tkinter as tk
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime

def twDateToGlobal(twDate):
    year, month, date = twDate.split("/")
    return datetime(int(year), int(month), int(date))

def Chart():
    cmd = "SELECT * FROM stockPrice"
    cursor.execute(cmd)
    rows = cursor.fetchall()
    x = [twDateToGlobal(row[0]) for row in rows] # 日期
    y = [row[2] for row in rows] # 股價
    plt.plot_date(x, y, 'g')
    plt.xticks(rotation=70)
    plt.show()

def View():
    global cursor
    for item in tree.get_children():
        tree.delete(item)
    cmd = sqltext.get(1.0, "end-1c")
    cursor.execute(cmd)
    rows = cursor.fetchall()
    for row in rows:
        # print(row) 
        tree.insert("", tk.END, values=row)        

db = sqlite3.connect("stockPrice.db1")
cursor = db.cursor()

root = tk.Tk()

sqltext = tk.Text(root, height=3, width=100)
sqltext.pack()
sqltext.insert(tk.END, "SELECT * FROM stockPrice")

buttonQuery = tk.Button(text="查詢", command=View)
buttonQuery.pack(pady=10)
buttonChart = tk.Button(text="股價圖", command=Chart)
buttonChart.pack(pady=10)

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

db.close()