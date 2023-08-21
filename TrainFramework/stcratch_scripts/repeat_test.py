import numpy as np
import torch

a = np.array([4,3,6,9])
print(a)
# b = a.repeat(3,1)
a = torch.from_numpy(a)
b = a.repeat(3,1).repeat(2,1,1)
print(b)
print(b.shape)