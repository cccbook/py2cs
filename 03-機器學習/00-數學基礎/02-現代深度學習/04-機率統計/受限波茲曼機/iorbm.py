import rbm

def train_iorbm(inputs, outputs, n_hidden, learning_rate=0.1, k=1, training_epochs=1000):
    rng = numpy.random.RandomState(123)

    # construct IORBM
    ihrbm = RBM(input=inputs, n_visible=len(inputs[0]), n_hidden=n_hidden, numpy_rng=rng)
    hhrbm = RBM(input=???, n_visible=len(inputs[0]), n_hidden=n_hidden, numpy_rng=rng)
    horbm = RBM(input=outputs, n_visible=len(inputs[0]), n_hidden=n_hidden, numpy_rng=rng)
    # train
    for epoch in range(training_epochs):
        rbm.contrastive_divergence(lr=learning_rate, k=k)
        cost = rbm.get_reconstruction_cross_entropy()
        print(f'Training epoch {epoch}, cost is {cost}')
    return rbm

if __name__ == "__main__":
    inputs = numpy.array([[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]])
    outputs = inputs
    rbm = train_iorbm(inputs, outputs, 2)
    # test
    v = numpy.array([[0, 0, 0, 1, 1, 0], # 被歸回 001110
                     [1, 1, 0, 0, 0, 0]]) # 被歸回 111000

    print(rbm.reconstruct(v))
