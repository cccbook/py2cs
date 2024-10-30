import matrix

print(dir(matrix))
# 創建兩個矩陣
A = matrix.Matrix(2, 2)
B = matrix.Matrix(2, 2)

# 設定矩陣的數值
A.data[0] = 1
A.data[1] = 2
