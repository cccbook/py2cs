import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from matplotlib import pyplot as plt

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def __call__(self, x):
        return self.model(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def __call__(self, x):
        return self.model(x)

def generate_noise(batch_size, latent_dim):
    return mx.random.normal((batch_size, latent_dim))

def train_gan(epochs=100, batch_size=64, latent_dim=100):
    # Initialize models
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    # Optimizers
    g_optimizer = optim.Adam(learning_rate=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(learning_rate=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(epochs):
        # Generate fake data
        noise = generate_noise(batch_size, latent_dim)
        fake_images = generator(noise)
        
        # Generate real data (random for demonstration)
        real_images = mx.random.uniform(-1, 1, (batch_size, 784))
        
        # Labels
        real_labels = mx.ones((batch_size, 1))
        fake_labels = mx.zeros((batch_size, 1))

        # Train Discriminator
        with mx.stream(mx.cpu):
            # Real images
            real_loss = nn.losses.binary_cross_entropy(
                discriminator(real_images), real_labels
            )
            
            # Fake images
            fake_loss = nn.losses.binary_cross_entropy(
                discriminator(fake_images), fake_labels
            )
            
            d_loss = (real_loss + fake_loss) / 2.0

        # Train Generator
        with mx.stream(mx.cpu):
            noise = generate_noise(batch_size, latent_dim)
            g_loss = nn.losses.binary_cross_entropy(
                discriminator(generator(noise)), real_labels
            )

        # Update weights
        discriminator.update(d_optimizer.step(discriminator, d_loss))
        generator.update(g_optimizer.step(generator, g_loss))

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] D_loss: {float(d_loss):.4f}, G_loss: {float(g_loss):.4f}")

        # Save generated image every 20 epochs
        if epoch % 20 == 0:
            test_noise = generate_noise(1, latent_dim)
            generated = generator(test_noise)
            img = generated.reshape((28, 28))
            
            plt.figure(figsize=(5,5))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(f'gan_epoch_{epoch}.png')
            plt.close()

    return generator, discriminator

if __name__ == "__main__":
    # Train the GAN
    generator, discriminator = train_gan()
    
    # Generate final sample
    noise = generate_noise(1, 100)
    generated = generator(noise)
    img = generated.reshape((28, 28))
    
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig('final_generated.png')
    plt.show()
