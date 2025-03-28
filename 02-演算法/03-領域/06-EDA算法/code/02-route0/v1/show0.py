# https://claude.ai/chat/3a4d2284-c8be-4aae-8520-b4ef59f17028
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
        """解析 DEF 文件內容"""
        # 解析 DIEAREA
        die_match = re.search(r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', def_content)
        if die_match:
            self.die_area = tuple(map(int, die_match.groups()))
        
        # 解析 COMPONENTS 段落
        components_section = re.search(r'COMPONENTS.*?END COMPONENTS', def_content, re.DOTALL)
        if components_section:
            comp_pattern = r'-\s+(\w+)\s+(\w+).*?PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)'
            for match in re.finditer(comp_pattern, components_section.group(0)):
                name, cell_type, x, y = match.groups()
                self.components[name] = Component(name, cell_type, int(x), int(y))
        
        # 解析 NETS 段落
        nets_section = re.search(r'NETS.*?END NETS', def_content, re.DOTALL)
        if nets_section:
            current_net = None
            for line in nets_section.group(0).split('\n'):
                if line.strip().startswith('-'):
                    # 新的網路開始
                    net_name = line.strip().split()[1]
                    current_net = Net(net_name, [])
                    self.nets.append(current_net)
                elif 'ROUTED' in line and current_net:
                    # 解析路徑點
                    points = re.finditer(r'\(\s*(\d+)\s+(\d+)\s*\)', line)
                    for point in points:
                        x, y = map(int, point.groups())
                        current_net.route_points.append((x, y))

    def visualize(self, output_file: str = 'placement_routing.png'):
        """生成視覺化圖形"""
        plt.figure(figsize=(12, 8))
        
        # 設置圖形範圍
        plt.xlim(self.die_area[0], self.die_area[2])
        plt.ylim(self.die_area[1], self.die_area[3])
        
        # 繪製晶片邊界
        plt.plot([self.die_area[0], self.die_area[2]], [self.die_area[1], self.die_area[1]], 'k-', linewidth=2)
        plt.plot([self.die_area[0], self.die_area[2]], [self.die_area[3], self.die_area[3]], 'k-', linewidth=2)
        plt.plot([self.die_area[0], self.die_area[0]], [self.die_area[1], self.die_area[3]], 'k-', linewidth=2)
        plt.plot([self.die_area[2], self.die_area[2]], [self.die_area[1], self.die_area[3]], 'k-', linewidth=2)
        
        # 繪製元件
        for comp in self.components.values():
            # 繪製元件為藍色方框
            rect = plt.Rectangle((comp.x-2, comp.y-2), 4, 4, facecolor='blue', alpha=0.5)
            plt.gca().add_patch(rect)
            # 添加元件名稱標籤
            plt.text(comp.x, comp.y+5, comp.name, ha='center', va='bottom', fontsize=8)
        
        # 繪製布線，使用不同顏色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.nets)))
        for net, color in zip(self.nets, colors):
            if len(net.route_points) > 1:
                # 將路徑點轉換為x和y座標列表
                x_coords = [p[0] for p in net.route_points]
                y_coords = [p[1] for p in net.route_points]
                # 繪製布線路徑
                plt.plot(x_coords, y_coords, '-', color=color, linewidth=1, alpha=0.7)
                # 在布線起點添加網路名稱
                plt.text(x_coords[0], y_coords[0], net.name, fontsize=6, color=color)
        
        plt.title('Placement and Routing Visualization')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

def generate_visualization(def_file_path: str, output_file: str = 'placement_routing.png'):
    """主要函數：讀取DEF文件並生成視覺化圖形"""
    # 讀取DEF文件
    with open(def_file_path, 'r') as f:
        def_content = f.read()
    
    # 創建視覺化器並處理
    visualizer = DEFVisualizer()
    visualizer.parse_def(def_content)
    visualizer.visualize(output_file)
    
# 使用示例
if __name__ == "__main__":
    generate_visualization('output.def', 'placement_routing.png')