import heapq
import numpy as np
import matplotlib.pyplot as plt

class AStarRouter:
    def __init__(self, grid_size=50, metal_layers=3):
        """
        初始化A*佈線路由器
        
        參數:
        - grid_size: 網格大小
        - metal_layers: 金屬層數量
        """
        self.grid_size = grid_size
        self.metal_layers = metal_layers
        
        # 創建三維網格（層、行、列）
        self.grid = np.zeros((metal_layers, grid_size, grid_size), dtype=int)
        
        # 定義可移動方向（包括層間移動）
        self.directions = [
            (0, 1, 0),   # 右
            (0, -1, 0),  # 左
            (0, 0, 1),   # 下
            (0, 0, -1),  # 上
            (0, 0, 0),   # 同層不移動
            (1, 0, 0),   # 上一層
            (-1, 0, 0)   # 下一層
        ]
        
        # 設置障礙物（模擬已佔用區域）
        self._create_obstacles()
    
    def _create_obstacles(self):
        """
        創建模擬的佈線障礙物
        """
        # 在不同層隨機放置障礙
        for layer in range(self.metal_layers):
            num_obstacles = self.grid_size // 5
            for _ in range(num_obstacles):
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                self.grid[layer, x, y] = 1
    
    def _manhattan_distance(self, start, goal):
        """
        計算曼哈頓距離啟發式函數
        
        參數:
        - start: 起點座標 (layer, x, y)
        - goal: 終點座標 (layer, x, y)
        
        返回:
        - 估計代價
        """
        layer_diff = abs(start[0] - goal[0])
        x_diff = abs(start[1] - goal[1])
        y_diff = abs(start[2] - goal[2])
        
        # 層間移動代價較高
        return x_diff + y_diff + layer_diff * 10
    
    def _is_valid_move(self, current, next_pos):
        """
        檢查移動是否合法
        
        參數:
        - current: 當前座標 (layer, x, y)
        - next_pos: 下一個座標 (layer, x, y)
        
        返回:
        - 是否可移動
        """
        l1, x1, y1 = current
        l2, x2, y2 = next_pos
        
        # 檢查邊界
        if (l2 < 0 or l2 >= self.metal_layers or
            x2 < 0 or x2 >= self.grid_size or
            y2 < 0 or y2 >= self.grid_size):
            return False
        
        # 檢查是否為障礙物
        if self.grid[l2, x2, y2] == 1:
            return False
        
        return True
    
    def a_star_routing(self, start, goal):
        """
        A*路由算法實現
        
        參數:
        - start: 起點座標 (layer, x, y)
        - goal: 終點座標 (layer, x, y)
        
        返回:
        - 最短路徑或 None
        """
        # 優先隊列
        open_list = []
        # 已訪問節點
        closed_set = set()
        
        # 起始節點的代價計算
        heapq.heappush(open_list, (0, start, [start]))
        
        while open_list:
            # 取出代價最小的節點
            current_cost, current, path = heapq.heappop(open_list)
            
            # 已到達目標
            if current == goal:
                return path
            
            # 避免重複訪問
            if current in closed_set:
                continue
            closed_set.add(current)
            
            # 嘗試所有可能方向
            for direction in self.directions:
                next_layer = current[0] + direction[0]
                next_x = current[1] + direction[1]
                next_y = current[2] + direction[2]
                next_pos = (next_layer, next_x, next_y)
                
                # 檢查移動是否合法
                if self._is_valid_move(current, next_pos):
                    # 計算移動代價
                    move_cost = 1
                    if direction[0] != 0:  # 層間移動代價較高
                        move_cost = 10
                    
                    # 計算總代價
                    new_cost = current_cost + move_cost
                    estimated_total_cost = (
                        new_cost + 
                        self._manhattan_distance(next_pos, goal)
                    )
                    
                    # 加入優先隊列
                    new_path = path + [next_pos]
                    heapq.heappush(
                        open_list, 
                        (estimated_total_cost, next_pos, new_path)
                    )
        
        return None
    
    def visualize_routing(self, path):
        """
        可視化路由路徑
        
        參數:
        - path: 路由路徑
        """
        if not path:
            print("未找到有效路徑")
            return
        
        plt.figure(figsize=(15, 5))
        
        # 繪製每一層的路徑
        for layer in range(self.metal_layers):
            plt.subplot(1, self.metal_layers, layer + 1)
            plt.title(f'Layer {layer + 1}')
            plt.imshow(self.grid[layer], cmap='binary')
            
            # 繪製該層的路徑
            layer_path = [p for p in path if p[0] == layer]
            if layer_path:
                xs = [p[2] for p in layer_path]
                ys = [p[1] for p in layer_path]
                plt.plot(xs, ys, 'r-', linewidth=2)
        
        plt.tight_layout()
        plt.show()
    
    def route(self, start, goal):
        """
        執行完整路由
        
        參數:
        - start: 起點座標 (layer, x, y)
        - goal: 終點座標 (layer, x, y)
        """
        print(f"路由: {start} -> {goal}")
        path = self.a_star_routing(start, goal)
        
        if path:
            print(f"找到路徑，長度：{len(path)}")
            self.visualize_routing(path)
        else:
            print("無法找到有效路徑")

# 執行範例
router = AStarRouter(grid_size=50, metal_layers=3)

# 嘗試不同的起點和終點
start = (0, 10, 10)   # 第1層的(10, 10)
goal = (2, 40, 40)    # 第3層的(40, 40)

router.route(start, goal)