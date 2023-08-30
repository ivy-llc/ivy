import ivy.functional.frontends.numpy as ivy_np
import numpy as np
import ivy

ivy.set_backend("numpy")

choices = [[0, 1, 2, 3], [10, 11, 12, 13],
  [20, 21, 22, 23], [30, 31, 32, 33]]

args = [2, 3, 1, 0]

# choices = np.array(choices)
# args = np.array(args)

# a = np.array([0, 1]).reshape((2,1,1))
# c1 = np.array([1,2,3]).reshape((1,3,1))
# c2 = np.array([-1,-2,-3,-4,-5]).reshape((1,1,5))

# bshape = ivy.broadcast_shapes(*[a.shape, c1.shape, c2.shape])
# bshape = ivy.broadcast_shapes(*[args.shape, choices.shape])
# print(bshape)

# res = ivy.zeros(bshape)

# res[0,:] = choices
# res[1,:,:] = c2

# print(res)

out = np.choose(args, choices)
print(out)

out_ivy = ivy_np.choose(args, choices)
print(out_ivy)
