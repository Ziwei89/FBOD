import numpy as np
import torch


a = np.random.random((4,4,3))
b = np.ones((4,4))
b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(b)
print(a)

a[:,:,2]=b
print(a)