import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train the Hopfield network on the provided patterns."""
        for pattern in patterns:
            pattern = np.reshape(pattern, (self.size, 1))
            self.weights += np.dot(pattern, pattern.T)
        # Zero out the diagonal (no self-connections)
        np.fill_diagonal(self.weights, 0)

    def recall(self, input_pattern, steps=5):
        """Recall a pattern from the network."""
        pattern = np.array(input_pattern)
        for _ in range(steps):
            for i in range(self.size):
                # Calculate the net input for neuron i
                net_input = np.dot(self.weights[i], pattern)
                # Update the neuron state using sign activation function
                pattern[i] = 1 if net_input > 0 else -1
        return pattern

# Example usage
if __name__ == "__main__":
    # Define training patterns (stored as 1s and -1s)
    patterns = [
        [1, 1, -1, -1],
        [-1, 1, 1, -1],
        [-1, -1, 1, 1]
    ]

    # Create a Hopfield network with the appropriate size
    size = len(patterns[0])
    hopfield_net = HopfieldNetwork(size)

    # Train the network
    for i in range(100):
        hopfield_net.train(patterns)

    # Test the network with a noisy input
    # test_input = [-1,1,1,1] 
    test_input = [1, 1, -1, 1]  # One element is noisy
    recalled_pattern = hopfield_net.recall(test_input)

    print("Input Pattern:", test_input)
    print("Recalled Pattern:", recalled_pattern)
