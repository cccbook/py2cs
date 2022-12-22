# packages
from TransMat import TransMat
import numpy as np

# grid setting
grid_row = 4
grid_col = 4
grid_size = grid_row*grid_col

# parameters
TheMatrix = np.zeros((16,16,4))

for row in range(grid_row):
    for col in range(grid_col):
        now_loc = row*4+col
        TheMatrix[:, now_loc, 0] = TransMat(grid_row, grid_col, now_loc, 'up').flatten()
        TheMatrix[:, now_loc, 1] = TransMat(grid_row, grid_col, now_loc, 'left').flatten()
        TheMatrix[:, now_loc, 2] = TransMat(grid_row, grid_col, now_loc, 'down').flatten()
        TheMatrix[:, now_loc, 3] = TransMat(grid_row, grid_col, now_loc, 'right').flatten()

# set state 0 and 15 is terminal
T0 = np.zeros([16,4])
T0[0,:] = 1
T15 = np.zeros([16,4])
T15[15,:] = 1

TheMatrix[:,0,:] = T0
TheMatrix[:,15,:] = T15

np.save('T.npy', TheMatrix)
