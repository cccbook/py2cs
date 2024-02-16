import tkinter as tk

def main():
    root = tk.Tk()
    root.title("簡易文字編輯器")
    
    # 創建菜單
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)
    
    # 創建 "文件" 選單
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="文件", menu=file_menu)
    file_menu.add_command(label="新建文件")
    file_menu.add_command(label="打開文件")
    file_menu.add_command(label="保存文件")
    file_menu.add_separator()
    file_menu.add_command(label="退出", command=root.quit)
    
    # 創建 "編輯" 選單
    edit_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="編輯", menu=edit_menu)
    edit_menu.add_command(label="剪切")
    edit_menu.add_command(label="複製")
    edit_menu.add_command(label="貼上")
    
    # 創建文字區域
    text_area = tk.Text(root)
    text_area.pack(fill=tk.BOTH, expand=True)
    
    root.mainloop()

if __name__ == "__main__":
    main()
