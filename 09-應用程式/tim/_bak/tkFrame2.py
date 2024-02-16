import tkinter as tk
from tkinter import ttk

SUNKABLE_BUTTON = 'SunkableButton.TButton'

class tkinterApp(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # initializing frames to an empty array
        self.frames = {} 
  
        top_container = tk.LabelFrame(self, width=400, height=100, text="item")
        top_container.grid(row = 0, column=0, columnspan=2)
        
        mid_container_l = tk.LabelFrame(self, width=200, height=100, text="item_a")
        mid_container_l.grid(row = 1, column=0)

        mid_container_r = tk.LabelFrame(self, width=200, height=100, text="item_b")
        mid_container_r.grid(row = 1, column=1)

        but_container = tk.LabelFrame(self, width=400, height=400, text="item")
        but_container.grid(row = 2, column=0, columnspan=2, sticky=tk.W)
        
        for F in (One,Two):
            frame = F(self)
            self.frames[F] = frame

        btn_1 = tk.Button(but_container, text="btn_1", command=lambda : self.show_frame(One))
        btn_1.pack(side=tk.LEFT)

        btn_2 = tk.Button(but_container, text="btn_2", command=lambda : self.show_frame(Two))
        btn_2.pack(side=tk.LEFT)

        #self.show_frame(One)

    def show_frame(self, cont):
        frame = self.frames[cont]
        print(frame)
        frame.tkraise(aboveThis=None)

class One(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.grid(row = 3, column=0, columnspan=2, sticky=tk.W)

        but_container_in = tk.Frame(self, width=400, height=200, bg="yellow")
        but_container_in.pack(fill="both", expand=True)

"""
class One(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        but_container_in = tk.Frame(parent, width=400, height=200, bg="yellow")
        but_container_in.grid(row = 3, column=0, columnspan=2, sticky=tk.W)
"""

class Two(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.grid(row = 3, column=0, columnspan=2, sticky=tk.W)

        but_container_in_l = tk.Frame(self, width=400, height=200, bg="green")
        but_container_in_l.pack(fill="both", expand=True)

"""
class Two(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        but_container_in_l = tk.Frame(parent, width=400, height=200, bg="green")
        but_container_in_l.grid(row = 3, column=0, columnspan=2, sticky=tk.W)

        #but_container_in_r = tk.Frame(parent, width=200, height=200, bg="red")
        #but_container_in_r.grid(row = 3, column=1, columnspan=2, sticky=tk.W)
"""

# Driver Code
app = tkinterApp()
app.mainloop()