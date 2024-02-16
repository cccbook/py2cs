from hillClimbing import hillClimbing
from net import Net
import model
from solutionNet import SolutionNet

def start():
	net = Net.base_model([28,28], [10])
	return SolutionNet(net)

hillClimbing(start(), max_gens=1000, max_fails=50)
