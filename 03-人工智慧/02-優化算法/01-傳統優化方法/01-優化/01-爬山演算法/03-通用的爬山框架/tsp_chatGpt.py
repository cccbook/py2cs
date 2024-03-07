import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(path, cities):
    distance = 0
    for i in range(len(path) - 1):
        distance += calculate_distance(cities[path[i]], cities[path[i+1]])
    return distance

def travel_salesman(cities):
    num_cities = len(cities)
    path = [i for i in range(num_cities)]
    np.random.shuffle(path)  # 隨機初始化路徑

    best_path = path.copy()
    best_distance = total_distance(best_path, cities)

    for _ in range(10000):  # 迭代次數，可以根據需要調整
        for i in range(num_cities - 1):
            new_path = path.copy()
            new_path[i], new_path[i+1] = new_path[i+1], new_path[i]
            new_distance = total_distance(new_path, cities)

            if new_distance < best_distance:
                best_path = new_path
                best_distance = new_distance

        np.random.shuffle(path)  # 隨機改變路徑順序

    return best_path

def plot_path(path, cities):
    path_coords = [cities[i] for i in path]
    path_coords.append(path_coords[0])  # 回到起點形成一個迴圈

    print('path_coords=', path_coords)
    print('*path_coords=', *path_coords)
    print('zip(*path_coords)=', zip(*path_coords))
    x, y = zip(*path_coords)
    print('x=', x, 'y=', y)

    plt.plot(x, y, marker='o')
    plt.scatter(x, y, c='red')
    plt.title('Traveling Salesman Problem')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

if __name__ == "__main__":
    # 生成隨機城市座標，可以根據需要修改城市數量
    np.random.seed(42)
    num_cities = 10
    cities = np.random.rand(num_cities, 2)

    # 求解旅行推銷員問題
    best_path = travel_salesman(cities)

    # 繪製路徑圖
    plot_path(best_path, cities)
