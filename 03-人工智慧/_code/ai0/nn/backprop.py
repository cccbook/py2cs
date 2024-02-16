def gradient(net):
    net.forward()
    net.backward()
    return net
