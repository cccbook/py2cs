# https://claude.ai/chat/3a4d2284-c8be-4aae-8520-b4ef59f17028

import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import random
import math

@dataclass
class Pin:
    name: str           # Pin 的名稱
    is_input: bool     # 是否為輸入端
    is_output: bool    # 是否為輸出端
    net: str = None    # 對應的網路（預設為 None）
    x: float = 0       # Pin 在 2D 空間的 X 座標
    y: float = 0       # Pin 在 2D 空間的 Y 座標

@dataclass
class Cell:
    name: str           # Cell 的名稱
    cell_type: str     # Cell 的類型（如：邏輯門、電容器等）
    pins: List[Pin]    # 此 Cell 包含的 Pin 列表
    width: float = 1   # Cell 寬度（預設為 1）
    height: float = 1  # Cell 高度（預設為 1）
    x: float = 0       # Cell 在 2D 空間的 X 座標
    y: float = 0       # Cell 在 2D 空間的 Y 座標

@dataclass
class Net:
    name: str               # Net 的名稱
    source_pin: Pin = None  # Net 的來源 Pin（預設為 None）
    sink_pins: List[Pin] = None  # Net 的接收 Pins 列表（預設為 None）


class SimpleRouter:
    def __init__(self):
        # 儲存 Cell 的字典，鍵為 Cell 的名稱，值為 Cell 物件
        self.cells: Dict[str, Cell] = {}
        # 儲存 Net 的字典，鍵為 Net 的名稱，值為 Net 物件
        self.nets: Dict[str, Net] = {}
        # 定義芯片的寬度（假設值為 100）
        self.die_width = 100
        # 定義芯片的高度（假設值為 100）
        self.die_height = 100
        # 定義網格大小（假設值為 1）
        self.grid_size = 1

    def parse_yosys(self, netlist_str: str):
        """解析 Yosys 輸出的網表"""
        
        # 解析模組定義
        # 使用正則表達式來尋找模組的名稱和端口
        module_match = re.search(r'module\s+(\w+)\((.*?)\);', netlist_str, re.DOTALL)
        if not module_match:
            raise ValueError("無法找到模組定義")  # 如果找不到模組定義，拋出錯誤
        
        module_name = module_match.group(1)  # 取得模組名稱
        ports = module_match.group(2).split(',')  # 取得端口並以逗號分割
        
        # 解析端口定義
        input_ports = set()  # 儲存所有輸入端口的集合
        output_ports = set()  # 儲存所有輸出端口的集合
        for line in netlist_str.split('\n'):
            if 'input' in line:  # 如果是輸入端口
                match = re.search(r'input\s+(?:\[\d+:\d+\])?\s*(\w+)', line)
                if match:
                    input_ports.add(match.group(1))  # 添加輸入端口
            elif 'output' in line:  # 如果是輸出端口
                match = re.search(r'output\s+(?:\[\d+:\d+\])?\s*(\w+)', line)
                if match:
                    output_ports.add(match.group(1))  # 添加輸出端口

        # 解析 assign 語句，提取訊號賦值
        assign_pattern = r'assign\s+(\w+)\s*=\s*(.+?);'  # 匹配 assign 語句的正則表達式
        for match in re.finditer(assign_pattern, netlist_str):
            output_wire = match.group(1)  # 取得輸出端的信號
            expression = match.group(2)  # 取得賦值表達式
            
            # 創建邏輯門單元，這裡的邏輯門可以是任意邏輯操作，根據輸出端信號的名稱來命名 Cell
            cell_name = f"cell_{output_wire}"
            pins = []  # 儲存這個邏輯門的所有 Pin
            
            # 添加輸出 Pin，將其設為輸出端
            pins.append(Pin(output_wire, False, True))
            
            # 查找表達式中的輸入信號，並將其對應為輸入 Pin
            input_signals = re.findall(r'[ab]\[\d\]|\w+(?<!&)(?<!\|)(?<!\^)(?<!\()', expression)
            for signal in input_signals:
                if signal not in ('&', '|', '^', '~'):  # 排除一些邏輯運算符 （只取線路名稱）
                    pins.append(Pin(signal, True, False))  # 添加對應的輸入 Pin
            
            # 創建一個 Cell 物件，並將其加入到 cells 字典中
            cell = Cell(cell_name, "LOGIC", pins)
            self.cells[cell_name] = cell


    def place_cells(self):
        """簡單的單元放置算法 (Simple cell placement algorithm)
        使用網格式放置策略，將電路單元均勻分布在晶片上
        """
        # 計算需要放置的總單元數
        num_cells = len(self.cells)
        
        # 計算網格的邊長
        # 例如：9個單元會得到 3x3 的網格，12個單元會得到 4x4 的網格
        # math.ceil 確保有足夠的網格空間放置所有單元
        grid_size = math.ceil(math.sqrt(num_cells))
        
        # 初始化放置位置為左下角
        current_x = 0  # 當前網格的 x 座標（從0開始）
        current_y = 0  # 當前網格的 y 座標（從0開始）
        
        # 計算每個網格的實際寬度和高度
        # 例如：如果晶片大小是 100x100，網格大小是 4，
        # 則每個網格是 25x25 的大小
        grid_width = self.die_width / grid_size   # 單個網格的寬度
        grid_height = self.die_height / grid_size  # 單個網格的高度
        
        # 遍歷所有需要放置的單元
        for cell in self.cells.values():
            # 計算單元的實際放置座標
            # 將單元放在網格的中心點
            # 例如：對於第一個網格(0,0)，如果網格大小是25x25
            # 則單元會被放置在 (12.5, 12.5) 的位置
            cell.x = current_x * grid_width + grid_width/2   # 單元的 x 座標
            cell.y = current_y * grid_height + grid_height/2 # 單元的 y 座標
            
            # 移動到下一個網格位置
            current_x += 1
            
            # 如果到達當前行的末尾
            # 則移動到下一行的開始位置
            if current_x >= grid_size:  # 若已到達行尾
                current_x = 0           # x 座標重置到行首
                current_y += 1          # y 座標增加，移到下一行

    def route_nets(self):
        """簡單的布線算法 - 使用曼哈頓路徑（Manhattan routing）
        為每個輸出管腳找到對應的輸入管腳，並在它們之間創建布線路徑
        """
        routes = []  # 存儲所有布線路徑的列表
        
        # 遍歷所有電路單元
        for cell in self.cells.values():
            # 遍歷當前單元的所有管腳
            for pin in cell.pins:
                # 只處理輸出管腳
                if pin.is_output:
                    # 找到這個輸出管腳需要連接的所有輸入管腳
                    for target_cell in self.cells.values():
                        for target_pin in target_cell.pins:
                            # 檢查是否為匹配的輸入管腳（名稱相同的輸入管腳）
                            if target_pin.is_input and target_pin.name == pin.name:
                                # 為這對管腳創建曼哈頓布線路徑
                                route = self.create_manhattan_route(
                                    (cell.x, cell.y),        # 起點：輸出管腳所在單元的位置
                                    (target_cell.x, target_cell.y)  # 終點：輸入管腳所在單元的位置
                                )
                                # 將布線信息添加到結果列表
                                routes.append((pin.name, route))
        return routes

    def create_manhattan_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """創建簡單的曼哈頓路徑（Manhattan path）
        
        參數:
            start (Tuple[float, float]): 起點座標 (x1, y1)
            end (Tuple[float, float]): 終點座標 (x2, y2)
        
        返回:
            List[Tuple[float, float]]: 包含三個點的曼哈頓路徑
                - 起點
                - 轉折點
                - 終點
        """
        x1, y1 = start  # 解包起點座標
        x2, y2 = end    # 解包終點座標
        
        # 返回三個點來定義曼哈頓路徑：
        # 1. start: 起點 (x1, y1)
        # 2. (x2, y1): 轉折點，x座標已到達終點，y座標還在起點
        # 3. end: 終點 (x2, y2)
        return [
            start,      # 路徑起點
            (x2, y1),   # 轉折點（先水平移動）
            end         # 路徑終點
        ]

    def generate_def(self) -> str:
        """生成簡化的 DEF (Design Exchange Format) 格式輸出
        
        DEF 是電子設計自動化（EDA）中常用的標準文件格式，
        用於描述 IC 設計中的物理佈局信息
        
        返回:
            str: 完整的 DEF 格式字符串
        """
        # 初始化 DEF 文件內容列表
        def_content = []
        
        # 添加 DEF 文件頭部信息
        def_content.append("VERSION 5.8 ;")              # DEF 文件版本
        def_content.append(f"DESIGN top ;")              # 設計名稱
        def_content.append(f"UNITS DISTANCE MICRONS 1000 ;")  # 定義物理單位
        
        # 定義晶片的物理區域
        # DIEAREA 指定了可放置單元的矩形區域，從(0,0)到(die_width, die_height)
        def_content.append(f"DIEAREA ( 0 0 ) ( {self.die_width} {self.die_height} ) ;")
        
        # 添加單元放置信息（COMPONENTS段）
        def_content.append(f"COMPONENTS {len(self.cells)} ;")  # 聲明組件總數
        for cell in self.cells.values():
            # 每個單元的格式：
            # - <單元名稱> <單元類型>
            #   + PLACED (<x座標> <y座標>) <方向>
            def_content.append(f"- {cell.name} {cell.cell_type}")
            # PLACED 表示單元的放置位置，N 表示標準方向（不旋轉）
            def_content.append(f"  + PLACED ( {int(cell.x)} {int(cell.y)} ) N ;")
        def_content.append("END COMPONENTS")  # 結束組件段
        
        # 添加布線信息（NETS段）
        routes = self.route_nets()  # 獲取所有布線路徑
        def_content.append(f"NETS {len(routes)} ;")  # 聲明網路總數
        for net_name, route in routes:
            # 每條網路的格式：
            # - <網路名稱>
            #   + ROUTED (<點1座標>) (<點2座標>) ... (<點N座標>)
            def_content.append(f"- {net_name}")
            # 將路徑點轉換為 DEF 格式的字符串
            # 例如：(12 34) (56 78) (90 12)
            path_str = " ".join([f"( {int(x)} {int(y)} )" for x, y in route])
            def_content.append(f"  + ROUTED {path_str} ;")
        def_content.append("END NETS")  # 結束網路段
        
        # 添加文件結束標記
        def_content.append("END DESIGN")
        
        # 將所有行用換行符連接，生成完整的 DEF 文件內容
        return "\n".join(def_content)
# place_route() 函數用於放置和佈線數位電路
# 輸入參數 netlist_str 是 Yosys 產生的網表字串
def place_route(netlist_str: str):
    # 建立 SimpleRouter 物件實例，用於處理放置和佈線
    router = SimpleRouter()
    
    # 解析 Yosys 格式的網表
    # 將網表轉換為內部資料結構以供後續處理
    router.parse_yosys(netlist_str)
    
    # 執行單元放置
    # 決定每個邏輯單元在晶片上的物理位置
    router.place_cells()
    
    # 產生 DEF(Design Exchange Format)格式的輸出
    # DEF 是一種描述 IC 實體版圖的標準格式
    router.generate_def()
    
    # 回傳處理完成的 router 物件
    return router

# 開啟並讀取包含 Yosys 網表的輸入檔案
with open('netlist.v', 'r') as f:
    # 將整個網表檔案內容讀入字串變數
    netlist_str = f.read()

# 呼叫 place_route() 函數進行放置和佈線
# router 物件包含處理後的結果
router = place_route(netlist_str)

# 產生最終的 DEF 輸出
# 取得 DEF 格式的內容字串
def_content = router.generate_def()
# 將 DEF 內容寫入輸出檔案
with open('output.def', 'w') as f:
    f.write(def_content)