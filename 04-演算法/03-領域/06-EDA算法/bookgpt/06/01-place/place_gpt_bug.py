import random
import math

# 定義模組
modules = [
    {"name": "A", "width": 3, "height": 2},
    {"name": "B", "width": 2, "height": 2},
    {"name": "C", "width": 4, "height": 3},
    {"name": "D", "width": 2, "height": 1},
]

# 定義晶片尺寸
chip_width = 10
chip_height = 10

# 隨機生成初始佈局
def generate_initial_layout():
    layout = {}
    for module in modules:
        x = random.randint(0, chip_width - module["width"])
        y = random.randint(0, chip_height - module["height"])
        layout[module["name"]] = (x, y)
    return layout

# 計算成本函數
def calculate_cost(layout):
    overlap_penalty = 0
    for i in range(len(modules)):
        for j in range(i + 1, len(modules)):
            m1, m2 = modules[i], modules[j]
            x1, y1 = layout[m1["name"]]
            x2, y2 = layout[m2["name"]]

            # 判斷模組是否重疊
            if not (x1 + m1["width"] <= x2 or x2 + m2["width"] <= x1 or
                    y1 + m1["height"] <= y2 or y2 + m2["height"] <= y1):
                overlap_penalty += 1

    # 示例：此處僅包含重疊懲罰，可擴展為面積、延遲或功耗
    return overlap_penalty

# 模擬退火算法
def simulated_annealing(initial_layout, initial_temp, cooling_rate, max_steps):
    current_layout = initial_layout
    current_cost = calculate_cost(current_layout)
    temperature = initial_temp

    for step in range(max_steps):
        if temperature <= 0:
            break

        # 生成鄰域解
        new_layout = current_layout.copy()
        module = random.choice(modules)
        x = random.randint(0, chip_width - module["width"])
        y = random.randint(0, chip_height - module["height"])
        new_layout[module["name"]] = (x, y)

        # 計算新解的成本
        new_cost = calculate_cost(new_layout)

        # 接受概率
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            current_layout = new_layout
            current_cost = new_cost

        # 降低溫度
        temperature *= cooling_rate

    return current_layout, current_cost

# 主程式
if __name__ == "__main__":
    initial_layout = generate_initial_layout()
    initial_temp = 100.0
    cooling_rate = 0.95
    max_steps = 1000

    final_layout, final_cost = simulated_annealing(initial_layout, initial_temp, cooling_rate, max_steps)

    print("最佳佈局：", final_layout)
    print("最終成本：", final_cost)
