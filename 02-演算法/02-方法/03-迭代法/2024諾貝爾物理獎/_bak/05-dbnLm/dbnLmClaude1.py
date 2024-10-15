import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.weights = np.random.randn(n_visible, n_hidden) * 0.1
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sample_hidden(self, visible):
        activation = np.dot(visible, self.weights) + self.hidden_bias
        p_hidden = 1 / (1 + np.exp(-activation))
        return p_hidden > np.random.random(self.n_hidden)

    def sample_visible(self, hidden):
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        p_visible = 1 / (1 + np.exp(-activation))
        return p_visible > np.random.random(self.n_visible)

    def train(self, data, learning_rate=0.1, epochs=100, batch_size=10):
        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                v0 = batch
                h0 = self.sample_hidden(v0)
                v1 = self.sample_visible(h0)
                h1 = self.sample_hidden(v1)

                # Convert boolean arrays to float
                h0_float = h0.astype(float)
                h1_float = h1.astype(float)
                v0_float = v0.astype(float)
                v1_float = v1.astype(float)

                self.weights += learning_rate * (np.dot(v0_float.T, h0_float) - np.dot(v1_float.T, h1_float))
                self.visible_bias += learning_rate * np.sum(v0_float - v1_float, axis=0)
                self.hidden_bias += learning_rate * np.sum(h0_float - h1_float, axis=0)

class DBN:
    def __init__(self, layers):
        self.layers = layers
        self.rbms = [RBM(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def pretrain(self, data, learning_rate=0.1, epochs=100, batch_size=10):
        input_data = data
        for rbm in self.rbms:
            print(f"Pretraining RBM with {rbm.n_visible} visible and {rbm.n_hidden} hidden units")
            rbm.train(input_data, learning_rate, epochs, batch_size)
            input_data = rbm.sample_hidden(input_data)

    def generate(self, num_samples):
        samples = np.random.binomial(1, 0.5, (num_samples, self.layers[0]))
        for rbm in self.rbms:
            samples = rbm.sample_hidden(samples)
        for rbm in reversed(self.rbms):
            samples = rbm.sample_visible(samples)
        return samples

# Example usage
vocab_size = 1000
sequence_length = 20
hidden_layers = [500, 250, 100]

# Create dummy data
data = np.random.binomial(1, 0.3, (1000, vocab_size * sequence_length))

# Initialize and pretrain DBN
dbn = DBN([vocab_size * sequence_length] + hidden_layers)
dbn.pretrain(data)

# Generate new samples
generated_samples = dbn.generate(10)
print("Generated samples shape:", generated_samples.shape)