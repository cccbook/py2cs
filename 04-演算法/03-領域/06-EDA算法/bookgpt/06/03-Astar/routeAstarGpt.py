import heapq

def a_star(grid, start, goal):
    """
    A* 算法實現

    :param grid: 2D 網格，0 表示可通行，1 表示障礙。
    :param start: 起點 (x, y)。
    :param goal: 終點 (x, y)。
    :return: 最短路徑列表，或 None 如果無法找到路徑。
    """
    def heuristic(a, b):
        # 使用 Manhattan 距離作為啟發式函數
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # 反轉路徑

    open_set = []
    heapq.heappush(open_set, (0, start))  # 優先隊列，儲存 (f_score, 節點)

    came_from = {}  # 用於回溯路徑
    g_score = {start: 0}  # 起點到每個節點的實際成本
    f_score = {start: heuristic(start, goal)}  # f(n) = g(n) + h(n)

    while open_set:
        _, current = heapq.heappop(open_set)

        # 如果找到目標，回溯生成路徑
        if current == goal:
            return reconstruct_path(came_from, current)

        # 探索鄰居節點
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 更新成本和路徑
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # 無法找到路徑

# 測試範例
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]
start = (0, 0)
goal = (4, 4)

path = a_star(grid, start, goal)
if path:
    print("找到最短路徑:", path)
else:
    print("無法找到路徑")
