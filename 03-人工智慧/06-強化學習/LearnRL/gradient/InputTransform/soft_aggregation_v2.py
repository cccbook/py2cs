'''
In this script, we use the Radial Basis Function as the example of soft aggregation.
In this version, the center data is determined by K-Means
'''

import numpy as np

# parameters
MAX_ITERATIONS = 1000
NUM_INPUT = 1000
SIZE_STATE = 5
NUM_FEATURES = 5

# define radial basis function
def RBF(state, const, sigma):
    '''
    state: the state of outside world
    const: the center of the radial basis function
    sigma: the variance range of gaussian distribution
    '''
    feature = np.exp(- ( pow(np.linalg.norm(state - const, axis = 1), 2) ) / pow(sigma, 2) )
    return feature

def Kmeans(states, size_state, num_centers, num_input, max_iter):
    '''
    states: all states
    size_state: dimension of one states
    num_centers: numbers of centers
    num_input: numbers of the input data
    max_iter: the maximal number for K means
    '''
    # init centers
    center_idx = np.random.randint(low = 0, high = num_input, size = num_centers)
    last_idx = np.zeros(5)
    cluster = np.zeros(num_input)
    count = 0
    while (not np.all(last_idx == center_idx)) and (count < max_iter):
        last_idx = center_idx.copy()
        # find cluster
        for i in range(num_input):
            cluster[i] = np.argmin(np.linalg.norm(state_input[center_idx, :] - state_input[i,:], axis = 1))
        # move the center
        for j in range(num_centers):
            cluster_idx = np.where(cluster == j)
            if len(cluster_idx[0]) == 0:
                continue
            min_in_cluster = np.argmin(np.linalg.norm(state_input[cluster_idx] - np.mean(state_input[cluster_idx]), axis = 1))
            center_idx[j] = cluster_idx[0][min_in_cluster]
        count += 1
        # warning for exceeding maximal iterations
        if count == max_iter:
            print('='*30)
            print('Exceeded maximal iterations.')
            print('='*30)
    centers = np.array([states[center_idx, :]])
    return centers

## example
# init data and parameters
state_input = np.random.rand(NUM_INPUT, SIZE_STATE)
Sigma = 1

# find the center
const_input = Kmeans(state_input, SIZE_STATE, NUM_FEATURES, NUM_INPUT, MAX_ITERATIONS)

# find new state
rst = np.zeros([NUM_INPUT, NUM_FEATURES])
for i in range(NUM_INPUT):
    rst[i, :] = RBF(state_input[i, :], const_input, Sigma)

print('state: \n', state_input)
print('new state: \n', rst)