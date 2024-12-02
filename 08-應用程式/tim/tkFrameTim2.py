import tkinter as tk

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Page Switching Example")
        print('SampleApp...')
        # 創建兩個框架
        self.frame_a = FrameA(self)
        self.frame_b = FrameB(self)
        # ccc: 要加上下列兩行
        self.frame_a.grid(row=0, column=0, sticky='news')
        self.frame_b.grid(row=0, column=0, sticky='news')

        # 在開始時，顯示 A 頁面
        self.show_frame(self.frame_a)

    def show_frame(self, frame):
        print('show_frame...')
        """切換顯示框架"""
        frame.tkraise()

class FrameA(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # A 頁面的元素
        label_a = tk.Label(self, text="Page A")
        label_a.pack(pady=10)

        button_a_to_b = tk.Button(self, text="Go to Page B", command=lambda: master.show_frame(master.frame_b))
        button_a_to_b.pack()

class FrameB(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # B 頁面的元素
        label_b = tk.Label(self, text="Page B")
        label_b.pack(pady=10)

        button_b_to_a = tk.Button(self, text="Go to Page A", command=lambda: master.show_frame(master.frame_a))
        button_b_to_a.pack()

if __name__ == "__main__":
    app = SampleApp()
    app.geometry("300x200")  # 設置主窗口大小
    # app.geometry("800x600")  # 設置主窗口大小
    app.mainloop()