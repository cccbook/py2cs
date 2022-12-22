# packages
import numpy as np

# generate moving result
def TransMat(num_row, num_col, now_loc, action):
    '''
    This function will generate the moving result from (loc_row, loc_col)
    Input:
      -num_row: the row of grid
      -num_col: the column of grid
      -now_loc: now location
      -action: moving direction (Valid moving: up, down, left, right)
    '''
    # size check
    if now_loc >= num_row*num_col:
        print('Error! Out of index!')
        return None

    # create map
    WholeMap = np.zeros((num_row+2,num_col+2))

    # decide the row and column of now location
    loc_row = int(now_loc / num_col)
    loc_col = now_loc - loc_row*num_col

    # now location to map location
    MapRow = loc_row+1
    MapCol = loc_col+1

    # moving
    if action == 'up':
        WholeMap[MapRow-1, MapCol] = 1
    elif action == 'left':
        WholeMap[MapRow, MapCol-1] = 1
    elif action == 'down':
        WholeMap[MapRow+1, MapCol] = 1
    elif action == 'right':
        WholeMap[MapRow, MapCol+1] = 1
    else:
        print(str(action) + 'is not a valid action !')

    # rebound from the wall
    if np.max(WholeMap[:,0]) > 0:
        idx = np.argmax(WholeMap[:,0])
        WholeMap[idx,1] += WholeMap[idx,0]

    if np.max(WholeMap[:,num_col+1]) > 0:
        idx = np.argmax(WholeMap[:,num_col+1])
        WholeMap[idx,num_col] += WholeMap[idx,num_col+1]

    if np.max(WholeMap[0,:]) > 0:
        idx = np.argmax(WholeMap[0,:])
        WholeMap[1,idx] += WholeMap[0,idx]

    if np.max(WholeMap[num_row+1,:]) > 0:
        idx = np.argmax(WholeMap[num_row+1,:])
        WholeMap[num_row,idx] += WholeMap[num_row+1,idx]

    return WholeMap[1:(num_row+1),1:(num_col+1)]
