# https://claude.ai/chat/3a4d2284-c8be-4aae-8520-b4ef59f17028
# https://claude.ai/chat/53f00345-8345-483a-bf00-da7a7aaa4921

import matplotlib.pyplot as plt
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class Component:
    name: str
    cell_type: str
    x: int
    y: int
    input_signals: List[str] = None  # 新增 input_signals 欄位

@dataclass
class Net:
    name: str
    route_points: List[Tuple[int, int]]

class DEFVisualizer:
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.nets: List[Net] = []
        self.die_area: Tuple[int, int, int, int] = (0, 0, 0, 0)
        
    def parse_def(self, def_content: str):
        """解析 DEF 文件內容，包括處理微小座標偏移"""
        # 解析 DIEAREA（保持不變）
        die_match = re.search(r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', def_content)
        if die_match:
            self.die_area = tuple(map(int, die_match.groups()))
        
        # 解析 COMPONENTS（保持不變）
        components_section = re.search(r'COMPONENTS.*?END COMPONENTS', def_content, re.DOTALL)
        if components_section:
            comp_pattern = r'-\s+(\w+)\s+(\w+).*?PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)(?:.*?INPUTS\s*\((.*?)\))?'
            for match in re.finditer(comp_pattern, components_section.group(0), re.DOTALL):
                name, cell_type, x, y = match.groups()[:4]
                input_signals = match.group(5).split() if match.group(5) else None
                
                self.components[name] = Component(
                    name=name, 
                    cell_type=cell_type, 
                    x=int(x), 
                    y=int(y),
                    input_signals=input_signals
                )
        
        # 解析 NETS（修改為支持微小座標偏移）
        nets_section = re.search(r'NETS.*?END NETS', def_content, re.DOTALL)
        if nets_section:
            current_net = None
            for line in nets_section.group(0).split('\n'):
                if line.strip().startswith('-'):
                    # 使用第一個 '-' 後的名稱作為網路名稱
                    net_name = line.strip().split()[1]
                    current_net = Net(net_name, [])
                    self.nets.append(current_net)
                elif 'ROUTED' in line and current_net:
                    # 提取所有路由點，並處理微小偏移
                    points = re.finditer(r'\(\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*\)', line)
                    
                    route_points = []
                    for point in points:
                        # 支持整數和浮點數座標
                        x = float(point.group(1))
                        y = float(point.group(2))
                        route_points.append((x, y))
                    
                    # 更新當前網路的路由點
                    current_net.route_points = route_points
                    
                    # 如果存在微小偏移，可以在這裡添加額外處理
                    # 例如：檢測並存儲偏移量
                    if len(route_points) > 1:
                        # 計算第一個點的原始整數座標和帶偏移的浮點座標
                        original_x, original_y = int(route_points[0][0]), int(route_points[0][1])
                        offset_x = route_points[0][0] - original_x
                        offset_y = route_points[0][1] - original_y
                        
                        # 如果有微小偏移，可以將其存儲在 Net 物件中（需要擴展 Net 類）
                        current_net.offset = (offset_x, offset_y)

    def visualize(self, output_file: str = 'placement_routing.png'):
        """生成視覺化圖形"""
        plt.figure(figsize=(16, 10))  # 增加圖形大小
        
        # 設置圖形範圍
        plt.xlim(self.die_area[0], self.die_area[2])
        plt.ylim(self.die_area[1], self.die_area[3])
        
        # 繪製晶片邊界
        plt.plot([self.die_area[0], self.die_area[2]], [self.die_area[1], self.die_area[1]], 'k-', linewidth=2)
        plt.plot([self.die_area[0], self.die_area[2]], [self.die_area[3], self.die_area[3]], 'k-', linewidth=2)
        plt.plot([self.die_area[0], self.die_area[0]], [self.die_area[1], self.die_area[3]], 'k-', linewidth=2)
        plt.plot([self.die_area[2], self.die_area[2]], [self.die_area[1], self.die_area[3]], 'k-', linewidth=2)
        
        # 定義顏色映射
        color_map = {
            'AND': 'blue',
            'OR': 'green', 
            'XOR': 'red', 
            'NOT': 'purple'
        }
        
        # 繪製元件
        for comp in self.components.values():
            # 根據 cell_type 選擇顏色
            facecolor = color_map.get(comp.cell_type, 'gray')
            
            # 繪製元件為彩色方框
            rect = plt.Rectangle((comp.x-2, comp.y-2), 4, 4, 
                                facecolor=facecolor, 
                                alpha=0.5, 
                                edgecolor='black')
            plt.gca().add_patch(rect)
            
            # 添加元件名稱標籤
            plt.text(comp.x, comp.y+5, comp.name, ha='center', va='bottom', fontsize=8)
            
            # 添加輸入訊號標籤
            if comp.input_signals:
                input_text = f' {comp.cell_type} '.join(comp.input_signals)
                plt.text(comp.x, comp.y-3, input_text, 
                        ha='center', va='top', 
                        fontsize=7, 
                        color='darkred',
                        bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7))
        
        # 繪製布線，使用不同顏色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.nets)))
        
        # 存儲所有元件的邊界
        component_bounds = {}
        for comp in self.components.values():
            component_bounds[comp.name] = {
                'left': comp.x - 2,
                'right': comp.x + 2,
                'bottom': comp.y - 2,
                'top': comp.y + 2
            }
        
        for net, color in zip(self.nets, colors):
            if len(net.route_points) > 1:
                x_coords = [p[0] for p in net.route_points]
                y_coords = [p[1] for p in net.route_points]
                
                # 篩選出不在元件內部的路由點
                filtered_route = []
                for x, y in zip(x_coords, y_coords):
                    is_inside_component = False
                    for bounds in component_bounds.values():
                        if (bounds['left'] <= x <= bounds['right'] and 
                            bounds['bottom'] <= y <= bounds['top']):
                            is_inside_component = True
                            break
                    
                    if not is_inside_component:
                        filtered_route.append((x, y))
                
                # 如果有足夠的路由點，則繪製
                if len(filtered_route) > 1:
                    filtered_x = [p[0] for p in filtered_route]
                    filtered_y = [p[1] for p in filtered_route]
                    plt.plot(filtered_x, filtered_y, '-', color=color, linewidth=1, alpha=0.7)
                    plt.text(filtered_x[0], filtered_y[0], net.name, fontsize=6, color=color)
        
        plt.title('Placement and Routing Visualization with Detailed Inputs')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()  # 調整佈局
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def generate_visualization(def_file_path: str, output_file: str = 'placement_routing.png'):
    """主要函數：讀取DEF文件並生成視覺化圖形"""
    with open(def_file_path, 'r') as f:
        def_content = f.read()
    
    visualizer = DEFVisualizer()
    visualizer.parse_def(def_content)
    visualizer.visualize(output_file)
    
# 使用示例
if __name__ == "__main__":
    generate_visualization('output.def', 'placement_routing.png')
