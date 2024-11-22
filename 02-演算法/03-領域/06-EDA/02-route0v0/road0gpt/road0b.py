import re
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class VerilogPR:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size
        self.modules = {}
        self.wires = set()
        self.assignments = []
        self.placements = {}
        self.routes = defaultdict(list)

    def parse_verilog(self, filepath):
        """解析 Verilog 檔案"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # 提取 wire 聲明
            if line.startswith("wire"):
                wires = re.findall(r'(\w+)', line)
                self.wires.update(wires[1:])
            # 提取 assign 語句
            elif line.startswith("assign"):
                assign_match = re.match(r'assign\s+(\w+)\s*=\s*(.+);', line)
                if assign_match:
                    self.assignments.append(assign_match.groups())

    def placement(self):
        """隨機分配元件到佈局中"""
        x_range, y_range = self.canvas_size
        used_positions = set()
        for wire in self.wires:
            while True:
                x = random.randint(0, x_range - 1)
                y = random.randint(0, y_range - 1)
                if (x, y) not in used_positions:
                    self.placements[wire] = (x, y)
                    used_positions.add((x, y))
                    break

    def routing(self):
        """曼哈頓布線"""
        for src, expr in self.assignments:
            components = re.findall(r'\w+', expr)
            for comp in components:
                if comp in self.placements and src in self.placements:
                    src_pos = self.placements[src]
                    dest_pos = self.placements[comp]
                    self.routes[src].append(self.manhattan_route(src_pos, dest_pos))

    def manhattan_route(self, src, dest):
        """計算曼哈頓距離的布線路徑"""
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

    def visualize(self):
        """視覺化佈局與布線"""
        plt.figure(figsize=(8, 8))
        plt.grid(True)
        # 畫元件
        for wire, (x, y) in self.placements.items():
            plt.scatter(x, y, color='blue')
            plt.text(x + 0.3, y, wire, fontsize=8, color='black')
        # 畫布線
        for routes in self.routes.values():
            for route in routes:
                xs, ys = zip(*route)
                plt.plot(xs, ys, color='red')
        plt.title('Placement & Routing')
        plt.xlim(-1, self.canvas_size[0])
        plt.ylim(-1, self.canvas_size[1])
        plt.show()


# 測試程式
canvas_size = (20, 20)  # 佈局大小
pr = VerilogPR(canvas_size)
pr.parse_verilog('adder_synth.v')  # 替換為您的 Verilog 檔案路徑
pr.placement()
pr.routing()
pr.visualize()
