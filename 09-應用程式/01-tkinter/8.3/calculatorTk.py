import tkinter as tk

# 計算器類別
class Calculator:
    def __init__(self):
        # 建立主視窗
        self.window = tk.Tk()
        self.window.title("計算器")

        # 建立顯示結果的文字方塊
        self.entry = tk.Entry(self.window, font=("Arial", 16), justify="right")
        self.entry.grid(row=0, column=0, columnspan=4)

        # 建立按鈕
        buttons = [
            "7", "8", "9", "/",
            "4", "5", "6", "*",
            "1", "2", "3", "-",
            "0", ".", "=", "+"
        ]
        r = 1
        c = 0
        for btn in buttons:
            tk.Button(self.window, text=btn, font=("Arial", 14), width=5,
                      command=lambda button=btn: self.button_clicked(button)).grid(row=r, column=c)
            c += 1
            if c > 3:
                c = 0
                r += 1

    # 按下按鈕的事件處理器
    def button_clicked(self, button):
        if button == "=":  # 如果按下等於鍵，計算結果並顯示
            try:
                result = eval(self.entry.get())
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except Exception as e:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "錯誤")
        else:
            self.entry.insert(tk.END, button)  # 否則將按鈕的內容加到顯示文字方塊中

    # 執行計算器
    def run(self):
        self.window.mainloop()

# 建立計算器物件並執行
calculator = Calculator()
calculator.run()