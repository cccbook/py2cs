import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

class PhotoAlbumApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("圖片相簿")
        
        self.current_index = 0
        self.images = []
        
        self.label = tk.Label(self.window, text="歡迎使用圖片相簿應用！")
        self.label.pack()
        
        self.prev_button = tk.Button(self.window, text="上一張", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT)
        
        self.next_button = tk.Button(self.window, text="下一張", command=self.show_next_image)
        self.next_button.pack(side=tk.RIGHT)
        
        self.window.mainloop()
        
    def browse_images(self):
        # 選擇圖片檔案
        file_paths = filedialog.askopenfilenames(
            initialdir="/", title="選擇圖片", filetypes=(("圖片檔案", "*.jpg *.jpeg *.png"), ("所有檔案", "*.*"))
        )
        
        self.images = []
        self.current_index = 0
        
        for file_path in file_paths:
            # 將圖片加載到列表中
            image = Image.open(file_path)
            image = image.resize((500, 500))  # 調整圖片大小
            self.images.append(image)
        
        self.show_image()
        
    def show_image(self):
        # 取得當前索引對應的圖片
        image = self.images[self.current_index]
        photo = ImageTk.PhotoImage(image)
        
        self.label.configure(text="")  # 清除標籤中的文字
        self.label.configure(image=photo)
        self.label.image = photo  # 保存對圖片物件的參考
        
    def show_previous_image(self):
        # 顯示上一張圖片
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()
            
    def show_next_image(self):
        # 顯示下一張圖片
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

app = PhotoAlbumApp()
