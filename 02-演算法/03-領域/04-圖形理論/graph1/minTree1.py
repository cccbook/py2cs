import networkx as nx
import numpy as np
import pylab as plt

from graph1 import G

T = nx.minimum_spanning_tree(G, algorithm="kruskal")
c=nx.to_numpy_matrix(T)
print("c=", c)
w1 = c.sum()/2
print("tree weight:", w1)
pos = nx.circular_layout(G)
key=range(8); s=['v'+str(i) for i in key]
s = dict(zip(key, s))
nx.draw(T, pos, labels=s, font_weight='bold')
w2=nx.get_edge_attributes(T, 'weight')
nx.draw_networkx_edge_labels(T, pos, edge_labels=w2)
plt.show()
