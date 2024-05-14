from sys import platform

# 设置历史记录文件的路径
histfile = ".python_history"

def start():
    print('platform=', platform)
    if platform == "win32":
        from pyreadline3 import Readline
        global readline
        readline = Readline()
    else:
        pass

    try:
        global histfile
        readline.read_history_file(histfile)
        # 设置历史记录最大数量为1000
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

def input(prompt):
    if platform == "win32":
        return readline.readline(prompt)
    else:
        return input(prompt)

def end():
    if platform == "win32":
        global histfile
        readline.write_history_file(histfile)
    else:
        pass
