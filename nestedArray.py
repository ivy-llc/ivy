# meant for testing purposes, will be removed before merging

import ivy
import numpy as np

ivy.set_backend("numpy")
x = ivy.random_uniform(shape=(2, 2))
y = ivy.random_uniform(shape=(2, 3))
z = ivy.random_uniform(shape=(2, 2))


c = ivy.NestedArray.nested_array([x, y, z], dtype=np.float64)
print(c[0])
print(c.dtype)
print(c.ndim)
print(c.shape)

c = ivy.NestedArray.from_row_lengths([3, 1, 4, 1, 5, 9, 2, 6], [4, 0, 3, 1, 0])
print(c)
print(c.dtype)
print(c.ndim)
print(c.shape)

c = ivy.NestedArray.from_row_split([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8, 8])
print(c)
print(c.dtype)
print(c.ndim)
print(c.shape)
