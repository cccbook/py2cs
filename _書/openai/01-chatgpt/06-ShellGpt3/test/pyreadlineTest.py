#import readline
#import pyreadline as readline
from pyreadline import Readline

readline = Readline()

# 设置历史记录文件的路径
histfile = ".python_history"
try:
    readline.read_history_file(histfile)
    # 设置历史记录最大数量为1000
    readline.set_history_length(1000)
except FileNotFoundError:
    pass

while True:
    # 读取用户输入
    line = readline.readline(">>> ")
    line = line.strip()
    if line == "exit":
        break
    if line == "":
        continue

    # 将用户输入添加到历史记录中
    # readline.add_history(line)

    # 处理用户输入
    # ...

    # 将历史记录写入文件
    readline.write_history_file(histfile)
