import numpy as np


in_shape = np.array([384,672])
image_shape = np.array([720,1280])
ratio = np.min(in_shape/image_shape)
new_shape = image_shape * ratio

offset = (in_shape-new_shape)/2

print(ratio)
print(new_shape)
print(offset)
