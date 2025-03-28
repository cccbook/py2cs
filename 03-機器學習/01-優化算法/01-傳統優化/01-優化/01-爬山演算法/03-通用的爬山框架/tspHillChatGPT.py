import random
import math
from hillClimbing import hillClimbing

def total_distance(path, distances):
    return sum(distances[path[i]][path[i+1]] for i in range(len(path) - 1)) + distances[path[-1]][path[0]]

def height(path, distances):
    # return 1 / total_distance(path, distances)  # 目標函數：距離越小越好
    return -1*total_distance(path, distances)  # 目標函數：距離越小越好

def neighbor(path):
    a, b = random.sample(range(len(path)), 2)
    new_path = path[:]
    new_path[a], new_path[b] = new_path[b], new_path[a]
    return new_path

# 測試函數
def generate_random_tsp(n):
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    distances = [[math.dist(p1, p2) for p2 in points] for p1 in points]
    return list(range(n)), distances

# 預設旅行推銷員數據，兩兩距離在 1..10 之間
def predefined_tsp10():
    distances = [
        [0, 2, 9, 10, 3, 7, 4, 8, 6, 5],
        [2, 0, 4, 8, 6, 5, 9, 7, 3, 10],
        [9, 4, 0, 6, 5, 3, 7, 10, 2, 8],
        [10, 8, 6, 0, 2, 9, 5, 4, 7, 3],
        [3, 6, 5, 2, 0, 4, 10, 9, 8, 7],
        [7, 5, 3, 9, 4, 0, 6, 2, 10, 8],
        [4, 9, 7, 5, 10, 6, 0, 3, 8, 2],
        [8, 7, 10, 4, 9, 2, 3, 0, 5, 6],
        [6, 3, 2, 7, 8, 10, 8, 5, 0, 9],
        [5, 10, 8, 3, 7, 8, 2, 6, 9, 0]
    ]
    return list(range(10)), distances

# 預設旅行推銷員數據，最短迴圈為 4
def predefined_tsp4():
    distances = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    # return list(range(4)), distances
    return [0, 2, 1, 3], distances

# 生成測試案例
n = 10
# path, distances = generate_random_tsp(n)
# 生成預設測試案例
# path, distances = predefined_tsp10() # 10 個城市
path, distances = predefined_tsp4() # 4 個城市


# 執行爬山演算法
best_path = hillClimbing(path, lambda p: height(p, distances), neighbor)
print("Best path:", best_path)
print("Total distance:", total_distance(best_path, distances))
