import numpy as np
from skgeom import Polygon, polygon_split

# 建立一個凸多邊形
vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
polygon = Polygon(vertices)

# 將凸多邊形分割成若干個較小的凸多邊形
sub_polygons = polygon_split(polygon)

# 列印出分割後的凸多邊形的頂點座標
for sub_polygon in sub_polygons:
    print(sub_polygon.vertices)
