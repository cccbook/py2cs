import numpy as np
import math

def rotate_2d(vec, theta):
  # 將角度轉換為弧度
  # theta = np.radians(theta)
  
  # 建立旋轉矩陣
  rot_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
  ])
  
  # 將向量與旋轉矩陣相乘，以實現旋轉
  return rot_matrix @ vec

def scale_2d(vec, sx, sy):
  # 建立縮放矩陣
  scale_matrix = np.array([
    [sx, 0],
    [0, sy]
  ])
  
  # 將向量與縮放矩陣相乘，以實現縮放
  return scale_matrix @ vec


def translate_2d(vec, tx, ty):
  # 建立平移矩陣
  trans_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty],
    [0, 0, 1]
  ])
  
  # 將向量擴展為 3 維向量，並與平移矩陣相乘，以實現平移
  homogeneous_vec = np.append(vec, 1)
  r = trans_matrix @ homogeneous_vec
  return r[0:2]

vec = np.array([1, 0])
rotated_vec = rotate_2d(vec, math.pi/2)
print(rotated_vec)  # [-0. 1.]

scaled_vec = scale_2d(vec, 0.5, 0.5)
print(scaled_vec)  # [0.5 0]

translated_vec = translate_2d(vec, 2, 1)
print(translated_vec)  # [3 1]