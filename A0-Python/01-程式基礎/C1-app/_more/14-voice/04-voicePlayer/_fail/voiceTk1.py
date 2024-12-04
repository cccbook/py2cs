from tkinter import *
import tksnack

root = Tk()
tkSnack.initializeSnack(root)

snd = tkSnack.Sound()
snd.read('ex1.wav')
snd.play(blocking=1)