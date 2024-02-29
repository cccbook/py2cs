import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from diffusions import DenoisingDiffusionModel

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 10

# Load the data
train_dataset = datasets.MNIST(root="data", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = DenoisingDiffusionModel(
    image_size=28,
    image_channels=1,
    num_res_blocks=2,
    res_channels=64,
    diffusion_steps=1000,
    timesteps=100,
    noise_schedule="linear",
    loss_type="l2",
    optimizer_type="adam",
    optimizer_kwargs={"lr": learning_rate},
    device=device
)

# Train the model
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        loss = model(images)
        model.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss:.4f}")

# Generate samples
samples = model.sample(num_samples=16)
save_image(samples, "samples.png", nrow=4, normalize=True)