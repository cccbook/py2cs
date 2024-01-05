## packages 
import math

## class
class IndexTable:
    def __init__(self):
        self.dictionary = {}
    
    def __str__(self):
        print('There are ' + str(len(self.dictionary)) + ' items in dictionary.')
        return
    
    def count(self):
        return len(self.dictionary)
    
    def getIndex(self, key):
        d = self.dictionary
        if key in d:
            return d[key]
        else:
            count = self.count()
            d[key] = count
            return count

## function
def tiles(idx_table, num_tiles, states):
    # check type of idx_table
    if not type(idx_table) == IndexTable:
        print('Please put the IndexTable at idx_table.')
        return
    # pre-process of states
    s_floor = []
    for state in states:
        s_floor.append(math.floor(state*num_tiles))
    # tiling the states
    Tiles = []
    for tile in range(num_tiles):
        coords = [tile]
        for s in s_floor:
            coords.append( (s+tile) // num_tiles )
        Tiles.append(idx_table.getIndex(tuple(coords)))
    return Tiles

## main function
def main():
    # define a dictionary
    idx_T = IndexTable()
    # set tiling numbers in 0~1
    numTiles = 8
    # get the index of state
    print('The index of state [1,2] is')
    print(tiles(idx_T, numTiles, [1,2]))

if __name__ == '__main__':
    main()