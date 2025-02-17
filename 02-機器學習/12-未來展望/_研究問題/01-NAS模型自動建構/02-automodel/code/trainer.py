import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from net import Net

def log(msg):
	print(msg)
	# pass

n_epochs = 3
epoch_seconds_limit = 100000
# epoch_seconds_limit = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST("files/", train=True, download=True,
							 transform=torchvision.transforms.Compose([
								 torchvision.transforms.ToTensor(),
								 torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
	batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	torchvision.datasets.MNIST("files/", train=False, download=True,
							 transform=torchvision.transforms.Compose([
								 torchvision.transforms.ToTensor(),
								 torchvision.transforms.Normalize(
								 (0.1307,), (0.3081,))
							 ])),
	batch_size=batch_size_test, shuffle=True)

def init(net):
	global train_losses, train_counter, test_losses, test_counter, network, optimizer
	train_losses = []
	train_counter = []
	test_losses = []
	test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
	network = net
	optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
	tstart = datetime.now()
	network.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		tnow = datetime.now()
		dt = tnow - tstart
		if dt.seconds > epoch_seconds_limit: break
		optimizer.zero_grad()
		output = network(data)
		loss = F.nll_loss(F.log_softmax(output, dim=1), target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			log("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
			epoch, batch_idx * len(data), len(train_loader.dataset),
			100. * batch_idx / len(train_loader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
			(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def test():
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			output = network(data)
			test_loss += F.nll_loss(F.log_softmax(output, dim=1), target, size_average=False).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)
		accuracy = 100. * correct / len(test_loader.dataset)
		log("\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
		test_loss, correct, len(test_loader.dataset), accuracy))
		network.model["accuracy"] = accuracy.item()

def run(net):
	init(net)
	for epoch in range(1, n_epochs + 1):
		train(epoch)
	test()
	net.save()

def main():
	net = Net.cnn_model([28,28], [10])
	if net.exist():
		log("model exist!")
	else:
		run(net)

if __name__ == "__main__":
	main()
