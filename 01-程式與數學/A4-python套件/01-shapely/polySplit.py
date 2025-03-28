from shapely.geometry import Polygon, MultiPolygon

# 建立一個凸多邊形
vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
polygon = Polygon(vertices)

# 將凸多邊形分割成若干個較小的凸多邊形
sub_polygons = polygon.convex_hull

# 列印出分割後的凸多邊形的頂點座標
for sub_polygon in sub_polygons:
    print(sub_polygon.exterior.coords)
