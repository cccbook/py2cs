import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.nll_loss(output, target)
        acc = (output.argmax(dim=1) == target).sum().item() / target.size(0)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adadelta(self.parameters(), lr=1.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(description='PyTorch Lightning MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False)

    model = Net()

    trainer = Trainer(max_epochs=args.epochs) # , gpus=1 if use_cuda else 0

    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
