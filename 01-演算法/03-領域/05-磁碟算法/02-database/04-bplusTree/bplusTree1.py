from bplustree import BPlusTree
tree = BPlusTree('db/bplustree.db', order=50)
N = 100 # 10000
data = [f'data_{i}' for i in range(N)]
for i in range(N):
    tree[2*i] = data[i].encode('utf8')
for i in range(2*N):
    print(f'tree[{i}] = {tree.get(i)}')
tree.close()
