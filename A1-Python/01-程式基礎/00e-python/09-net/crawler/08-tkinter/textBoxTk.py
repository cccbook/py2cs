import tkinter as tk

root = tk.Tk()
T = tk.Text(root, height=2, width=30)
T.pack()
T.insert(tk.END, "SELECT * FROM stockPrice")
tk.mainloop()