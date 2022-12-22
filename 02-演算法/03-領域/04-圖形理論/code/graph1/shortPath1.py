import networkx as nx
import numpy as np
import pylab as plt

from graph1 import G

path = nx.shortest_path(G, 0, weight='weight')
d = nx.shortest_path_length(G, 0, weight='weight')
print('path=', path)
print('d=', d)