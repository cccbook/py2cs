# https://github.com/Chrisbelefantis/A-Star-Algorithm/blob/master/Astar-Algorithm.py
from tkinter import *
from functools import partial
from time import sleep
import heapq

def center_gui(root):

    # Gets the requested values of the height and widht.
    windowWidth = root.winfo_reqwidth()
    windowHeight = root.winfo_reqheight()

    # Gets both half the screen width/height and window width/height
    positionRight = int(root.winfo_screenwidth()/2 - windowWidth/2)
    positionDown = int(root.winfo_screenheight()/2 - windowHeight/2)

    # Positions the window in the center of the page.
    root.geometry("+{}+{}".format(positionRight, positionDown))


def pop_up_window(app):
    """
    Displays the start up window with the instructions.
    When the pop up is displayed the buttons at the backround
    are disabled
    """

    def start_button_action():

        app.enable_buttons()
        win.destroy()

    win = Toplevel()
    win.wm_title("Welcome")

    Label(win, text="Step 1: Select starting point",
          font=("Calibri", 13), pady=5, padx=10).pack()
    Label(win, text="Step 2: Select end point", font=(
        "Calibri", 13), pady=5, padx=10).pack()
    Label(win, text="Step 3: Select Obstacles", font=(
        "Calibri", 13), pady=5, padx=10).pack()
    Label(win, text="Click and hover.Then click again to stop", padx=25).pack()
    Label(win, text="Step 4: Press Enter to start",
          font=("Calibri", 13), pady=5, padx=10).pack()
    Label(win, text="Step 5: Press R to restart",
          font=("Calibri", 13), pady=5, padx=10).pack()
    Button(win, text="Start", command=start_button_action,
           ).pack()

    win.update_idletasks()
    center_gui(win)


class App:

    def __init__(self, master): # 初始化 25*25 棋盤

        self.master = master
        master.wm_title("A* Algorithm")
        self.buttons = []
        self.start = () # []
        self.goal = () # []
        self.obstacles = []
        self.mode = 0

        for i in range(25):
            self.buttons.append([])
            for j in range(25):

                # Initiliaze buttons
                button = Button(master, width=2, height=1,
                                command=partial(self.button_operation, i, j), state="disabled")

                self.buttons[i].append(button)

                # This event is used for the obstacle setting
                self.buttons[i][j].bind('<Enter>', partial(
                    self.add_obstacle, i, j))

                self.buttons[i][j].grid(row=i, column=j)

        master.update_idletasks()
        center_gui(master)

        pop_up_window(self)

    def enable_buttons(self):

        for i in range(25):
            for j in range(25):
                self.buttons[i][j].configure(state="normal")

    def disable_buttons(self):

        for i in range(25):
            for j in range(25):
                self.buttons[i][j].configure(state="disable")

    # Every time a button is clicked this function is triggered
    # This function is responsible for controling the flow of the program

    def button_operation(self, row, column): # 使用者按鈕互動
        """
        According to the value of 'mode' this fuction
        sets the value of start and end. Also by changing
        the value of mode it controls when we can set obstacles and
        when we can start the algorithm
        """

        # Set start mode
        if self.mode == 0:
            self.start = (row, column)
            self.mode = 1
            self.buttons[row][column].configure(bg='green')

        # Set end mode
        elif self.mode == 1:
            self.goal = (row, column)
            self.mode = 2
            self.buttons[row][column].configure(bg='red')

        elif self.mode == 2:
            # Set to set obstacles mode => By hovering over buttons
            self.mode = 3

        else:
            # When the mode = 2 the user cant set obstacles by hovering and the algorithm can start
            self.mode = 2

    def add_obstacle(self, row, column, event): # 加入障礙 (黑色)

        # Checks if we are in the obstacle setting mode
        if self.mode == 3:
            obstacle_node = (row,column)

            self.obstacles.append(obstacle_node) # self.obstacles.append(obstacle_node[:])
            self.buttons[row][column].configure(bg='black')

    def heuristic(self, node1, node2): # h(node) 函數 = |x2-x1| + |y2-y1| (曼哈頓距離)
        result = abs(node1[0] - node2[0]) + abs(node1[1]-node2[1])
        return result

    def find_neighbors(self, current, obstacles): # 找出上下左右的鄰居

        neighbors = []

        # With current[:] I create a new list and I dont use the pointer to the original list otherwise the end result whould have same lists

        right_neighbor = (current[0], current[1]+1)

        if 0 <= right_neighbor[1] < 25 and right_neighbor not in self.obstacles:
            neighbors.append(right_neighbor)

        left_neighbor = (current[0], current[1]-1)

        if 0 <= left_neighbor[1] < 25 and left_neighbor not in self.obstacles:
            neighbors.append(left_neighbor)

        up_neighbor = (current[0]+1, current[1])

        if 0 <= up_neighbor[0] < 25 and up_neighbor not in self.obstacles:

            neighbors.append(up_neighbor)

        down_neighbor = (current[0]-1, current[1])

        if 0 <= down_neighbor[0] < 25 and down_neighbor not in self.obstacles:

            neighbors.append(down_neighbor)

        down_right_neighbor = (current[0]+1, current[1]+1)

        if 0 <= down_right_neighbor[0] < 25 and 0 <= down_right_neighbor[1] < 25 and down_right_neighbor not in self.obstacles:
            neighbors.append(down_right_neighbor)

        up_right_neighbor = (current[0]-1, current[1]+1)

        if 0 <= up_right_neighbor[0] < 25 and 0 <= up_right_neighbor[1] < 25 and up_right_neighbor not in self.obstacles:

            neighbors.append(up_right_neighbor)

        up_left_neighbor = (current[0]-1, current[1]-1)

        if 0 <= up_left_neighbor[0] < 25 and 0 <= up_left_neighbor[1] < 25 and up_left_neighbor not in self.obstacles:

            neighbors.append(up_left_neighbor)

        down_left_neighbor = (current[0]+1, current[1]-1)

        if 0 <= down_left_neighbor[0] < 25 and 0 <= down_left_neighbor[1] < 25 and down_left_neighbor not in self.obstacles:
            neighbors.append(down_left_neighbor)

        return neighbors
        
    def reconstruct_path(self, cameFrom, current): # 畫出路徑 (紅色)
        total_path = []

        while current != self.start:

            self.buttons[current[0]][current[1]].configure(bg='red')

            total_path.append(current[:])
            current = cameFrom[current[0]][current[1]] # 一直往 cameFrom 追就可以了

    def a_star_algorithm(self, start, goal):

        open_set = {start}
        g_score = []
        f_score = []
        came_from = []

        # Initialiazation of g_score and came_from # 共 25*25 格
        for i in range(25):
            f_score.append([])
            g_score.append([])
            came_from.append([])
            for j in range(25):
                temp = float('inf')
                came_from[i].append(())
                g_score[i].append(temp)  # set it to infinity
                f_score[i].append(temp)  # set it to infinity

        g_score[start[0]][start[1]] = 0 # 起始點的 g(start) 為 0
        f_start = self.heuristic(start, goal)
        f_score[start[0]][start[1]] = f_start # 起始點的 f(s) = 0 + h(s)
        f_open_set = []
        heapq.heappush(f_open_set, (f_start, start))

        while len(open_set) > 0:
            self.master.update_idletasks()
            sleep(0.02)

            f_current = heapq.heappop(f_open_set) # 取得下一個 f 最低的點
            current = f_current[1]

            current_row = current[0]    # 取出 row
            current_column = current[1] # column

            if current == goal: # 已經到達目標了
                return self.reconstruct_path(came_from, current) # 印出路徑 (紅色)

            open_set.remove(current)    # 移除該點

            neighbors = self.find_neighbors(current, []) # 找出該點的鄰居

            for node in neighbors:      # 對於每個鄰居

                node_row = node[0]      # 取出 row
                node_column = node[1]   # 取出 column

                # The weight of every edge is 1 # 每個邊權重為 1
                tentative_gScore = g_score[current_row][current_column] + 1 # 臨時 gScore

                if tentative_gScore < g_score[node_row][node_column]: # 如果從 current 到 node 的新走法 gScore 更低

                    came_from[node_row][node_column] = (current_row, current_column) # 更新 node 的 came_from 為 current

                    g_score[node_row][node_column] = tentative_gScore # 設定 gScore

                    f_node = g_score[node_row][node_column] + self.heuristic(node, self.goal) # 設定 fScore
                    f_score[node_row][node_column] = f_node

                    if node not in open_set: # 若 node 不在 openSet (藍點) 中

                        self.buttons[node[0]][node[1]].configure(bg='blue') # 塗藍色
                        heapq.heappush(f_open_set, (f_node, node))
                        open_set.add(node) # 將 node 加入 openSet 中

        print("fail!")

    def find_path(self, event):

        # Checks if we are in the correct mode to start the algorithm
        if self.mode == 2:
            self.a_star_algorithm(self.start, self.goal) # 主要程式
            self.disable_buttons()

    def reset(self, event):

        if self.mode == 2:
            self.start = []
            self.goal = []
            self.obstacles = []
            self.mode = 0

            for i in range(25):
                for j in range(25):

                    self.buttons[i][j].configure(bg='SystemButtonFace')

            self.enable_buttons()


if __name__ == '__main__':
    root = Tk()
    app = App(root)

    # Starts the algorithm when we press enter
    root.bind('<Return>', app.find_path)
    # Resets when we press 'R'
    root.bind('r', app.reset)
    root.mainloop()
