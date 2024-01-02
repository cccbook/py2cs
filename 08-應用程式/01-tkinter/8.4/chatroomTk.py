import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox

class ChatroomApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("聊天室")
        self.chat_log = ""
        
        # 創建輸入框
        self.input_box = tk.Entry(self.window, width=50)
        self.input_box.pack(side=tk.LEFT)
        
        # 創建送出按鈕
        self.send_button = tk.Button(self.window, text="送出", command=self.send_message)
        self.send_button.pack(side=tk.LEFT)
        
        # 創建聊天紀錄框
        self.chat_log_box = scrolledtext.ScrolledText(self.window, width=60, height=20)
        self.chat_log_box.pack(side=tk.LEFT)
        
        self.window.mainloop()
    
    def send_message(self):
        message = self.input_box.get()
        if message != "":
            self.chat_log += "我: {}\n".format(message)
            self.chat_log_box.insert(tk.END, self.chat_log)
            
            # 清空輸入框
            self.input_box.delete(0, tk.END)


if __name__ == '__main__':
    app = ChatroomApp()
