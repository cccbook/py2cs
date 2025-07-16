import numpy as np
import gd
import nn

'''
[b1 w11 w21] [1 x1 x2] = o1
[b2 w12 w22] [1 x1 x2] = o2
[bo u1 u2] [1 o1 o2] = oo

b1 w11, w21, b2 w12, w22, bo u1, u2
'''

def net(p,x):
    oa = nn.neuron(p[0:3], x)
    ob = nn.neuron(p[3:6], x)
    oo = nn.neuron(p[6:9], [1, oa, ob])
    return oo

# o = [0,0,0,1] # and gate outputs
# o = [0,1,1,1] # or gate outputs
o = [0,1,1,0] # xor gate outputs 這個還是失敗
def loss(p, dump=False):
    o0 = net(p, [1,0,0])
    o1 = net(p, [1,0,1])
    o2 = net(p, [1,1,0])
    o3 = net(p, [1,1,1])
    delta = np.array([o0-o[0], o1-o[1], o2-o[2], o3-o[3]])
    if dump: print('o0={:.3f} o1={:.3f} o2={:.3f} o3={:.3f}'.format(o0,o1,o2,o3))
    return np.linalg.norm(delta, 2)

p = np.ones(9)/2 #np.zeros(9) # np.random.uniform(0,1,9) # np.rand(9) # np.zeros(9)
plearn = gd.gradientDescendent(loss, p, step=0.1, max_loops=30000)
loss(plearn, True)
