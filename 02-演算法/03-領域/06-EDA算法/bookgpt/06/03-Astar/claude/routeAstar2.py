import numpy as np
import heapq
import matplotlib.pyplot as plt

class AdvancedAStarRouter:
    def __init__(self, chip_width=100, chip_height=100, metal_layers=4):
        """
        初始化高級A*路由器
        
        參數:
        - chip_width: 晶片寬度
        - chip_height: 晶片高度
        - metal_layers: 金屬層數
        """
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.metal_layers = metal_layers
        
        # 三維網格：層、行、列
        self.routing_grid = np.zeros((metal_layers, chip_height, chip_width), dtype=int)
        
        # 設計規則和成本參數
        self.design_rules = {
            'via_cost': 10,           # 通孔代價
            'layer_change_cost': 5,   # 層間變換代價
            'congestion_penalty': 2   # 擁塞懲罰係數
        }
    
    def _generate_obstacles(self, density=0.1):
        """
        生成佈線障礙物
        
        參數:
        - density: 障礙物密度
        """
        for layer in range(self.metal_layers):
            obstacles = np.random.random(self.routing_grid[layer].shape) < density
            self.routing_grid[layer][obstacles] = 1
    
    def _calculate_heuristic(self, current, goal):
        """
        計算啟發式估值（加權曼哈頓距離）
        
        參數:
        - current: 當前位置 (layer, x, y)
        - goal: 目標位置 (layer, x, y)
        
        返回:
        - 估計代價
        """
        layer_diff = abs(current[0] - goal[0])
        x_diff = abs(current[1] - goal[1])
        y_diff = abs(current[2] - goal[2])
        
        # 考慮層間變換的額外代價
        return x_diff + y_diff + layer_diff * self.design_rules['layer_change_cost']
    
    def _is_valid_position(self, position):
        """
        檢查位置是否有效
        
        參數:
        - position: 位置坐標 (layer, x, y)
        
        返回:
        - 是否可用
        """
        layer, x, y = position
        return (
            0 <= layer < self.metal_layers and
            0 <= x < self.chip_height and
            0 <= y < self.chip_width and
            self.routing_grid[layer, x, y] == 0
        )
    
    def _get_neighbors(self, current):
        """
        獲取鄰近節點
        
        參數:
        - current: 當前位置 (layer, x, y)
        
        返回:
        - 可用鄰近節點列表
        """
        layer, x, y = current
        neighbors = [
            # 同層移動
            (layer, x+1, y), (layer, x-1, y),
            (layer, x, y+1), (layer, x, y-1),
            
            # 層間移動
            (layer+1, x, y) if layer+1 < self.metal_layers else None,
            (layer-1, x, y) if layer > 0 else None
        ]
        
        return [n for n in neighbors if n and self._is_valid_position(n)]
    
    def route(self, start, goal):
        """
        A*路由算法主體
        
        參數:
        - start: 起始點 (layer, x, y)
        - goal: 目標點 (layer, x, y)
        
        返回:
        - 最優路徑
        """
        open_set = []
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._calculate_heuristic(start, goal)}
        
        heapq.heappush(open_set, (f_score[start], start))
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current):
                # 計算移動代價
                tentative_g_score = g_score[current] + 1
                
                # 層間和通孔額外代價
                if neighbor[0] != current[0]:
                    tentative_g_score += (
                        self.design_rules['via_cost'] + 
                        self.design_rules['layer_change_cost']
                    )
                
                # 更新最佳路徑
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._calculate_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _reconstruct_path(self, came_from, current):
        """
        重建路徑
        
        參數:
        - came_from: 路徑追蹤字典
        - current: 當前節點
        
        返回:
        - 完整路徑
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def visualize_routing(self, path):
        """
        可視化路由結果
        
        參數:
        - path: 路由路徑
        """
        plt.figure(figsize=(15, 5))
        
        for layer in range(self.metal_layers):
            plt.subplot(1, self.metal_layers, layer + 1)
            plt.title(f'Layer {layer + 1}')
            plt.imshow(self.routing_grid[layer], cmap='binary')
            
            # 繪製該層路徑
            layer_path = [p for p in path if p[0] == layer]
            if layer_path:
                xs = [p[2] for p in layer_path]
                ys = [p[1] for p in layer_path]
                plt.plot(xs, ys, 'r-', linewidth=2)
        
        plt.tight_layout()
        plt.show()

# 使用範例
def main():
    # 初始化路由器
    router = AdvancedAStarRouter(
        chip_width=100, 
        chip_height=100, 
        metal_layers=4
    )
    
    # 生成障礙物
    router._generate_obstacles(density=0.1)
    
    # 設置起點和終點
    start = (0, 10, 10)    # 第1層 (10, 10)
    goal = (3, 80, 80)     # 第4層 (80, 80)
    
    # 執行路由
    path = router.route(start, goal)
    
    if path:
        print(f"找到路徑，長度: {len(path)}")
        router.visualize_routing(path)
    else:
        print("未找到可行路徑")

# 執行主程序
if __name__ == "__main__":
    main()