'''
In this script, we use the Radial Basis Function as the example of soft aggregation.
In this version, the center data is given by user.
'''

import numpy as np

# parameters
NUM_INPUT = 10
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

## example
state_input = np.random.rand(NUM_INPUT, SIZE_STATE)
Sigma = 1
const_input = np.array([[0]*5, [0.2]*5, [0.4]*5, [0.6]*5, [0.8]*5])

rst = np.zeros([NUM_INPUT, NUM_FEATURES])

for i in range(NUM_INPUT):
    rst[i, :] = RBF(state_input[i, :], const_input, Sigma)

print('state: \n', state_input)
print('new state: \n', rst)