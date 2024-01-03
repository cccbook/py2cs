from walsh import walsh
import numpy as np

w2 = walsh(2)
w4 = walsh(4)
w8 = walsh(8)

print(w2)
print(w4)

i2 = np.dot(w2, w2.transpose())
i4 = np.dot(w4, w4.transpose())
print('i2=\n', i2)
print('i4=\n', i4)

i8 = np.dot(w8, w8.transpose())
print('i8=\n', i8)
