# https://claude.ai/chat/9a58d131-2165-40be-8eb3-1023024e9fbc
import numpy as np
import random
import math

class ChipLayout:
    def __init__(self, modules, chip_width=100, chip_height=100):
        """
        初始化晶片佈局類別
        
        參數:
        - modules: 模組列表，每個模組是一個字典，包含名稱、寬度、高度
        - chip_width: 晶片寬度
        - chip_height: 晶片高度
        """
        self.modules = modules
        self.chip_width = chip_width
        self.chip_height = chip_height
        self.current_layout = self._generate_initial_layout()
    
    def _generate_initial_layout(self):
        """
        隨機生成初始佈局
        """
        layout = []
        for module in self.modules:
            # 隨機放置模組
            x = random.uniform(0, self.chip_width - module['width'])
            y = random.uniform(0, self.chip_height - module['height'])
            layout.append({
                'name': module['name'],
                'x': x,
                'y': y,
                'width': module['width'],
                'height': module['height']
            })
        return layout
    
    def calculate_cost(self, layout):
        """
        計算佈局代價
        包括面積利用率、模組間距、佈局總面積
        """
        total_area = 0
        overlap_penalty = 0
        distance_penalty = 0
        
        # 計算佈局的總面積
        for module in layout:
            total_area += module['width'] * module['height']
        
        # 檢查模組重疊
        for i in range(len(layout)):
            for j in range(i+1, len(layout)):
                m1, m2 = layout[i], layout[j]
                if (m1['x'] < m2['x'] + m2['width'] and 
                    m1['x'] + m1['width'] > m2['x'] and
                    m1['y'] < m2['y'] + m2['height'] and 
                    m1['y'] + m1['height'] > m2['y']):
                    overlap_penalty += 1000  # 大的懲罰值
                
                # 計算模組間距離
                distance = math.sqrt(
                    ((m1['x'] - m2['x'])**2 + (m1['y'] - m2['y'])**2)
                )
                distance_penalty += distance
        
        return total_area + overlap_penalty + distance_penalty
    
    def simulated_annealing(self, max_iterations=1000, initial_temp=100, cooling_rate=0.95):
        """
        模擬退火演算法實現
        """
        current_layout = self.current_layout
        current_cost = self.calculate_cost(current_layout)
        best_layout = current_layout.copy()
        best_cost = current_cost
        
        temperature = initial_temp
        
        for iteration in range(max_iterations):
            # 生成鄰近解
            new_layout = current_layout.copy()
            
            # 隨機移動一個模組
            module_to_move = random.choice(new_layout)
            module_to_move['x'] += random.uniform(-5, 5)
            module_to_move['y'] += random.uniform(-5, 5)
            
            # 計算新佈局代價
            new_cost = self.calculate_cost(new_layout)
            
            # 是否接受新解
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_layout = new_layout
                current_cost = new_cost
                
                # 更新最佳解
                if current_cost < best_cost:
                    best_layout = current_layout.copy()
                    best_cost = current_cost
            
            # 降溫
            temperature *= cooling_rate
        
        return best_layout, best_cost

# 測試範例
modules = [
    {'name': 'CPU', 'width': 20, 'height': 20},
    {'name': 'GPU', 'width': 15, 'height': 15},
    {'name': 'Memory', 'width': 10, 'height': 25},
    {'name': 'Network', 'width': 12, 'height': 12}
]

chip_layout = ChipLayout(modules)
best_layout, best_cost = chip_layout.simulated_annealing()

print("最佳佈局:")
for module in best_layout:
    print(f"{module['name']}: x={module['x']:.2f}, y={module['y']:.2f}")
print(f"佈局代價: {best_cost}")