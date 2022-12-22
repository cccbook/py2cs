# 光跡追蹤法 ray-tracing

## ccc: 請說明下列程式的原理

```py
# https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9
import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

width = 300
height = 200

max_depth = 3

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = normalize(intersection - nearest_object['center'])
            shifted_point = intersection + 1e-5 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                break

            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)

            # reflection
            color += reflection * illumination
            reflection *= nearest_object['reflection']

            origin = shifted_point
            direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
    print("%d/%d" % (i + 1, height))

plt.imsave('image.png', image)
```

這個程式是在模擬光线追蹤 (ray tracing)，主要目的是計算出每個像素點應該要顯示的顏色。光线追蹤是一種在電腦圖形學領域中常用的技術，它能夠計算出三維空間中的物體在不同的光源下的真實顏色。

程式中定義了許多函數，用來計算不同的參數。例如，normalize 函數可以用來計算向量的正規化，即計算出該向量的長度為 1 的版本。reflected 函數則可以計算出一個向量在另一個向量作為法向量下的反射向量。sphere_intersect 函數則可以用來計算一條光線是否與球體相交，並返回相交點與光源之間的距離。

接下來，程式進入了一個迴圈，逐行逐列地枚舉了圖像中的每個像素點。每當枚舉到一個像素點時，程式會計算出該像素點對應到的物體，並計算出該像素點應該要顯示的顏色。程式會模擬光線從視角（相機）發射出去，並且經過多次的折射和反射，最終在某個物體上折射或反射出來，為了計算出每個像素點應該要顯示的顏色，程式會模擬光線從視角（相機）發射出去，並且經過多次的折射和反射。首先，程式會用 nearest_intersected_object 函數來計算出光線最先撞到哪個物體，並返回該物體以及光線與該物體相交的點。接下來，程式會計算出該相交點的法向量，並用這個法向量來計算出光線在該點的反射向量。

當光線經過多次的折射和反射之後，最終會在某個物體上折射或反射出來。程式會用該物體的顏色來計算出折射或反射出來的光線的顏色。程式中定義了一個光源，它會影響物體表面的顏色。例如，如果光源的顏色較暗，那麼物體表面也會呈現較暗的顏色。另外，物體的反射率也會影響光線的顏色，如果物體的反射率較高，那麼光線經過該物體時會變得更亮。

最後，程式會將每個像素點的顏色存入一個二維數組中，並使用 matplotlib.pyplot 模組來顯示這個二維數組，也就是整張圖像。通過這種方式，程式能夠高效地計算出每個像素點的顏色，並生成一張真實逼真的三維圖像。

總而言之，這個程式主要是模擬光线追蹤 (ray tracing)，它能夠計算出三維空間中的物體在不同光源下的真實顏色，並將這些計算結果呈現為一張圖像。
