import numpy as np
x = np.array([[1, 3, 2, 4,4,3,4,5,6,3,2,1,2,2,35],[5, 5, 1, 22,12,1,1,2,3,3,2,1,2,2,5]])
k = 8
dropout_x = np.copy(x)
dropout_x = dropout_x.reshape(-1)
sorted_x = np.sort(np.absolute(dropout_x))
# print(sorted_x)
kplus1_largest = sorted_x[-(k + 1)]
small_indices = np.absolute(x) <= kplus1_largest
x[small_indices] = 0
print(x)