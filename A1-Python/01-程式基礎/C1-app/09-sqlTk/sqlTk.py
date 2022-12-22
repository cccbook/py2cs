from tkinter import ttk
import tkinter as tk
import sqlite3

def View():
    for item in tree.get_children():
        tree.delete(item)
    con1 = sqlite3.connect("stockPrice.db1")
    cur1 = con1.cursor()
    cmd = sqltext.get(1.0, "end-1c")
    cur1.execute(cmd)
    rows = cur1.fetchall()
    for row in rows:
        # print(row) 
        tree.insert("", tk.END, values=row)        
    con1.close()

root = tk.Tk()

sqltext = tk.Text(root, height=3, width=100)
sqltext.pack()
sqltext.insert(tk.END, "SELECT * FROM stockPrice WHERE 收盤價 > 410 ORDER BY 收盤價 DESC")

buttonQuery = tk.Button(text="查詢", command=View)
buttonQuery.pack(pady=10)

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