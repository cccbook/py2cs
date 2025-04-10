import random
import matplotlib.pyplot as plt

class SimplePR:
    def __init__(self, canvas_size, num_cells, num_nets):
        self.canvas_size = canvas_size  # 設計面板的大小 (width, height)
        self.num_cells = num_cells      # 元件數量
        self.num_nets = num_nets        # 連線數量
        self.cells = {}                 # 儲存元件的位置
        self.nets = []                  # 儲存連接的網路

    def placement(self):
        """隨機佈局元件到面板上，避免重疊"""
        for cell_id in range(self.num_cells):
            while True:
                x = random.randint(0, self.canvas_size[0] - 1)
                y = random.randint(0, self.canvas_size[1] - 1)
                if (x, y) not in self.cells.values():  # 確保無重疊
                    self.cells[cell_id] = (x, y)
                    break

    def generate_nets(self):
        """隨機生成網路 (連接的元件對)"""
        for _ in range(self.num_nets):
            src = random.randint(0, self.num_cells - 1)
            dest = random.randint(0, self.num_cells - 1)
            if src != dest:  # 避免自連
                self.nets.append((src, dest))

    def routing(self):
        """簡單的曼哈頓距離布線"""
        routes = []
        for src, dest in self.nets:
            src_pos = self.cells[src]
            dest_pos = self.cells[dest]
            route = self.manhattan_route(src_pos, dest_pos)
            routes.append(route)
        return routes

    def manhattan_route(self, src, dest):
        """計算曼哈頓距離路徑 (水平後垂直，或垂直後水平)"""
        x1, y1 = src
        x2, y2 = dest
        path = []
        # 水平移動
        step = 1 if x2 > x1 else -1
        for x in range(x1, x2 + step, step):
            path.append((x, y1))
        # 垂直移動
        step = 1 if y2 > y1 else -1
        for y in range(y1, y2 + step, step):
            path.append((x2, y))
        return path

    def visualize(self, routes):
        """視覺化元件與連線"""
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        # 畫元件
        for cell_id, (x, y) in self.cells.items():
            plt.scatter(x, y, color='blue', label=f'Cell {cell_id}' if cell_id == 0 else "")
            plt.text(x + 0.3, y, f'{cell_id}', fontsize=8, color='black')
        # 畫路徑
        for route in routes:
            xs, ys = zip(*route)
            plt.plot(xs, ys, color='red')
        plt.title('Placement & Routing')
        plt.xlim(-1, self.canvas_size[0])
        plt.ylim(-1, self.canvas_size[1])
        plt.legend()
        plt.show()


# 使用簡易 P&R
canvas_size = (20, 20)  # 設計面板大小
num_cells = 10          # 元件數量
num_nets = 5            # 網路數量

pr = SimplePR(canvas_size, num_cells, num_nets)
pr.placement()
pr.generate_nets()
routes = pr.routing()
pr.visualize(routes)
