# from -- https://ithelp.ithome.com.tw/articles/10201028
import numpy as np
Matrix_trans = np.array([[0.9,0.1],[0.5,0.5]])

# after n steps
Matrix_trans_step3 = np.linalg.matrix_power(Matrix_trans, 3)
Matrix_trans_step5 = np.linalg.matrix_power(Matrix_trans, 5)
Matrix_trans_step10 = np.linalg.matrix_power(Matrix_trans, 10)
Matrix_trans_step50 = np.linalg.matrix_power(Matrix_trans, 50)
Matrix_trans_step100 = np.linalg.matrix_power(Matrix_trans, 100)

# print the result
print('After 0 steps: \n', Matrix_trans, '\n')
print('After 3 steps: \n', Matrix_trans_step3, '\n')
print('After 5 steps: \n', Matrix_trans_step5, '\n')
print('After 10 steps: \n', Matrix_trans_step10, '\n')
print('After 50 steps: \n', Matrix_trans_step50, '\n')
print('After 100 steps: \n', Matrix_trans_step100, '\n')

# 

s0 = np.array([1.0, 0.0])
print('--------------------------------')
print('s0=', s0)
print('After 0 steps: \n', np.dot(s0, Matrix_trans), '\n')
print('After 100 steps: \n', np.dot(s0, Matrix_trans_step100), '\n')

s0 = np.array([0.5, 0.5])
print('--------------------------------')
print('s0=', s0)
print('After 0 steps: \n', np.dot(s0, Matrix_trans), '\n')
print('After 100 steps: \n', np.dot(s0, Matrix_trans_step100), '\n')
