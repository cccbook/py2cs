import numpy as np

def translate_3d(vec, tx, ty, tz):
  # 建立平移矩陣
  trans_matrix = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]
  ])
  
  # 將向量擴展為 4 維向量，並與平移矩陣相乘，以實現平移
  homogeneous_vec = np.append(vec, 1)
  r = trans_matrix @ homogeneous_vec
  return r[0:3]

def scale_3d(vec, sx, sy, sz):
  # 建立縮放矩陣
  scale_matrix = np.array([
    [sx, 0, 0],
    [0, sy, 0],
    [0, 0, sz]
  ])
  
  # 將向量與縮放矩陣相乘，以實現縮放
  return scale_matrix @ vec

def rotate_3d(vec, theta_x, theta_y, theta_z):
  # 建立旋轉矩陣
  rot_matrix = np.array([
    [np.cos(theta_x), -np.sin(theta_x), 0],
    [np.sin(theta_x), np.cos(theta_x), 0],
    [0, 0, 1]
  ]) @ np.array([
    [1, 0, 0],
    [0, np.cos(theta_y), -np.sin(theta_y)],
    [0, np.sin(theta_y), np.cos(theta_y)]
  ]) @ np.array([
    [np.cos(theta_z), 0, -np.sin(theta_z)],
    [0, 1, 0],
    [np.sin(theta_z), 0, np.cos(theta_z)]
  ])
  
  # 將向量與旋轉矩陣相乘，以實現旋轉
  return rot_matrix @ vec

vec = np.array([1, 1, 1])
translated_vec = translate_3d(vec, 1, 2, 3)
print(translated_vec)  # [2 3 4]

scaled_vec = scale_3d(vec, 0.5, 0.5, 0.5)
print(scaled_vec)  # [0.5 0.5 0.5]

rotated_vec = rotate_3d(vec, np.pi/2, 0, 0)
print(rotated_vec) # [-1.  1.  1.]
