import scipy.spatial.distance as distance

p1 = (1, 0)
p2 = (10, 2)
print('p1=', p1)
print('p2=', p2)
print('歐氏距離: euclidean(p1, p2)=', distance.euclidean(p1, p2))
print('曼哈頓距離: cityblock(p1, p2)=', distance.cityblock(p1, p2))
print('餘弦距離: cosine(p1, p2)=', distance.cosine(p1, p2))
