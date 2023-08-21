import numpy as np
import torch


a = np.random.random((4,5,4))
a = torch.from_numpy(a)
b = np.random.random((4,5,4,2))
b = torch.from_numpy(b)
c = np.random.random((4,5,4,4))
c = torch.from_numpy(c)


a_mask = a > 0.5
a1 = a[a_mask]
b1 = b[a_mask]
c1 = c[a_mask]
print(a1.size())
print(b1.size())
print(c1.size())

class_id = torch.argmax(b1, dim=1)

print(class_id)
print(class_id.size())
class_id_mask = class_id>0
print(class_id_mask)

class_id = class_id[class_id_mask]
b2 = b1[class_id_mask]
c2 = c1[class_id_mask]

class_id = class_id.unsqueeze(1)
print(class_id.size())
print(c2.size())

max_core = torch.max(b2,1).values
max_core = max_core.unsqueeze(1)
print(max_core.size())

prediction = torch.concat([c2, class_id, max_core], dim=1)
print(prediction.size())
print(prediction)