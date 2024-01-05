import numpy as np
import gd
import nn

def net(w,x):
  return nn.neuron(w,x)

o = [0,0,0,1] # and gate outputs
# o = [0,1,1,1] # or gate outputs
# o = [0,1,1,0] # xor gate outputs
def loss(p, dump=False):
    # [b,w1,w2] = p
    o0 = net(p, [1,0,0])
    o1 = net(p, [1,0,1])
    o2 = net(p, [1,1,0])
    o3 = net(p, [1,1,1])
    delta = np.array([o0-o[0], o1-o[1], o2-o[2], o3-o[3]])
    if dump: print('o0={:.3f} o1={:.3f} o2={:.3f} o3={:.3f}'.format(o0,o1,o2,o3))
    return np.linalg.norm(delta, 2)

p = [0.0, 0.0, 0.0]
plearn = gd.gradientDescendent(loss, p, max_loops=3000)
loss(plearn, True)
