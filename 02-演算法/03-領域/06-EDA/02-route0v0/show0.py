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
        """解析 DEF 文件內容"""
        # 解析 DIEAREA
        die_match = re.search(r'DIEAREA\s*\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', def_content)
        if die_match:
            self.die_area = tuple(map(int, die_match.groups()))
        
        # 解析 COMPONENTS 段落
        components_section = re.search(r'COMPONENTS.*?END COMPONENTS', def_content, re.DOTALL)
        if components_section:
            # 擴展正則表達式，同時匹配 INPUTS 資訊
            comp_pattern = r'-\s+(\w+)\s+(\w+).*?PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)(?:.*?INPUTS\s*\((.*?)\))?'
            for match in re.finditer(comp_pattern, components_section.group(0), re.DOTALL):
                name, cell_type, x, y = match.groups()[:4]
                # 如果有輸入訊號，則解析；否則為 None
                input_signals = match.group(5).split() if match.group(5) else None
                
                self.components[name] = Component(
                    name=name, 
                    cell_type=cell_type, 
                    x=int(x), 
                    y=int(y),
                    input_signals=input_signals
                )
        
        # 解析 NETS 段落（保持不變）
        nets_section = re.search(r'NETS.*?END NETS', def_content, re.DOTALL)
        if nets_section:
            current_net = None
            for line in nets_section.group(0).split('\n'):
                if line.strip().startswith('-'):
                    net_name = line.strip().split()[1]
                    current_net = Net(net_name, [])
                    self.nets.append(current_net)
                elif 'ROUTED' in line and current_net:
                    points = re.finditer(r'\(\s*(\d+)\s+(\d+)\s*\)', line)
                    for point in points:
                        x, y = map(int, point.groups())
                        current_net.route_points.append((x, y))

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
                input_text = f' {comp.cell_type} '.join(comp.input_signals) # ccc 修改
                # plt.text(comp.x, comp.y-7, input_text, 
                plt.text(comp.x, comp.y-3, input_text, 
                         ha='center', va='top', 
                         fontsize=7, 
                         color='darkred',
                         bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.7))
        
        # 繪製布線，使用不同顏色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.nets)))
        for net, color in zip(self.nets, colors):
            if len(net.route_points) > 1:
                x_coords = [p[0] for p in net.route_points]
                y_coords = [p[1] for p in net.route_points]
                plt.plot(x_coords, y_coords, '-', color=color, linewidth=1, alpha=0.7)
                plt.text(x_coords[0], y_coords[0], net.name, fontsize=6, color=color)
        
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