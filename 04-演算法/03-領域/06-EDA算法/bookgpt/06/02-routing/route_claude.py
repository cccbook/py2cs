# https://claude.ai/chat/9a58d131-2165-40be-8eb3-1023024e9fbc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

class ChipRouter:
    def __init__(self, chip_width=100, chip_height=100, metal_layers=4):
        """
        初始化晶片路由器
        
        參數:
        - chip_width: 晶片寬度
        - chip_height: 晶片高度
        - metal_layers: 金屬層數量
        """
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.metal_layers = metal_layers
        
        # 建立網格佈局
        self.grid = np.zeros((metal_layers, chip_height, chip_width), dtype=int)
        
        # 模擬網表：定義需要連接的元件
        self.netlist = [
            {'source': (10, 10), 'target': (80, 80), 'layer': 1},
            {'source': (20, 30), 'target': (70, 60), 'layer': 2},
            {'source': (50, 50), 'target': (90, 20), 'layer': 3}
        ]
    
    def global_routing(self):
        """
        全域布線階段
        劃分佈線區域，分析網路流量
        """
        print("--- 全域布線階段 ---")
        global_routing_map = np.zeros_like(self.grid[0])
        
        for net in self.netlist:
            sx, sy = net['source']
            tx, ty = net['target']
            layer = net['layer']
            
            # 區域劃分與流量分析
            route_region = np.zeros_like(global_routing_map)
            route_region[min(sy,ty):max(sy,ty), min(sx,tx):max(sx,tx)] = layer
            
            # 流量熱點分析
            traffic_density = np.sum(route_region > 0)
            print(f"網路 {net}: 佈線區域大小={traffic_density}, 層級={layer}")
            
            global_routing_map += route_region
        
        return global_routing_map
    
    def maze_routing(self, start, end, layer):
        """
        詳細布線：Maze Routing找最短路徑
        
        參數:
        - start: 起點座標
        - end: 終點座標
        - layer: 佈線金屬層
        """
        G = nx.grid_2d_graph(self.chip_height, self.chip_width)
        
        # 移除已被佔用的路徑
        for i in range(self.chip_height):
            for j in range(self.chip_width):
                if self.grid[layer, i, j] != 0:
                    G.remove_node((i, j))
        
        try:
            path = nx.shortest_path(G, source=start, target=end)
            return path
        except nx.NetworkXNoPath:
            print(f"無法在第 {layer} 層找到路徑")
            return None
    
    def detailed_routing(self, global_routing_map):
        """
        詳細布線階段
        根據全域布線結果進行精確佈線
        """
        print("\n--- 詳細布線階段 ---")
        routed_nets = []
        
        for net in self.netlist:
            sx, sy = net['source']
            tx, ty = net['target']
            layer = net['layer']
            
            path = self.maze_routing((sy, sx), (ty, tx), layer-1)
            
            if path:
                # 標記路徑
                for (y, x) in path:
                    if 0 <= y < self.chip_height and 0 <= x < self.chip_width:
                        self.grid[layer-1, y, x] = 1
                
                routed_nets.append({
                    'net': net,
                    'path': path,
                    'layer': layer
                })
                print(f"網路 {net}: 已成功路由，路徑長度={len(path)}")
            else:
                print(f"網路 {net}: 路由失敗")
        
        return routed_nets
    
    def visualize_routing(self, routed_nets):
        """
        可視化路由結果
        """
        plt.figure(figsize=(10, 8))
        
        for route in routed_nets:
            layer = route['layer']
            path = route['path']
            
            xs = [x for (y, x) in path]
            ys = [y for (y, x) in path]
            
            plt.plot(xs, ys, label=f'Layer {layer}', linewidth=2)
        
        plt.title('晶片佈線路徑')
        plt.xlabel('X 座標')
        plt.ylabel('Y 座標')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def route(self):
        """
        執行完整的佈線流程
        """
        global_routing_map = self.global_routing()
        routed_nets = self.detailed_routing(global_routing_map)
        self.visualize_routing(routed_nets)

# 執行佈線
router = ChipRouter()
router.route()