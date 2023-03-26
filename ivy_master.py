import ivy
# #
ivy.set_numpy_backend()
# x = ivy.array([0, 10, 15, 20, -50, 0])
# y = ivy.nonzero(x)
# print(y)
#(ivy.array([1, 2, 3, 4]),)

# x = ivy.array([ [ 0 , 1,3 ] ,[-1,-2,3] ])
# y = ivy.nonzero(x , as_tuple=True)
# print(y)
#(ivy.array([0, 0, 1, 1]), ivy.array([0, 1, 0, 1]))
# #
# x = ivy.array([[0, 2], [-1, -2]])
# y = ivy.nonzero(x, as_tuple=False)
# print(y)
# # # ivy.array([[0, 1], [1, 0], [1, 1]])
# # #
# x = ivy.array([0, 0])
# y = ivy.nonzero(x, size=5,fill_value=7)
# print(y)
# # # (ivy.array([1, 4]),)
# # #
# # # With:
# # #
# # #
# # # class: `ivy.NativeArray`
# # #
# # #
# # # input:
# # #
#x = ivy.native_array([[10, 20], [10, 0], [0, 0]])
# y = ivy.nonzero(x)
# print(y)
# #(ivy.array([0, 0, 1]), ivy.array([0, 1, 0]))
#
#x = ivy.native_array([[0], [1], [1], [0], [1]])
# y = ivy.nonzero(x)
# print(y)
# # (ivy.array([1, 2, 4]), ivy.array([0, 0, 0]))
# #
# With:
#
#
# class: `ivy.Container`
#
#
# input:
#

#x = ivy.Container(a=ivy.array([1,1,1]), b=ivy.array([0]) , c = ivy.array([1]))
#
#y = ivy.nonzero(x)
#print(y)


#ivy.unset_backend()
#print(ivy.get_backend())
#
# import torch
# ivy.set_torch_backend()
# x = torch.Tensor([3,3,5,0])
# out_tensor = torch.zeros(4, dtype=torch.int64) # create an empty tensor with the desired shape and dtype
#
# ivy.nonzero(x, as_tuple=False , out = out_tensor)
# print(out_tensor)
# #
#

# import numpy as np
# x = np.array([3,3,5,0])
# y = ivy.nonzero(x)
# print(y)

#
# import torch
# x = torch.tensor([[0, 1, 0], [2, 0, 3]])
#
# out_tensor = torch.empty((0), dtype=torch.int64) # create an empty tensor with the desired shape and dtype
# y=ivy.nonzero(x,as_tuple=False, out=out_tensor,dtype=ivy.float32)
#
# print(out_tensor)
# print(type(y[2][1].item()))
#print(type(y[0][0] ) )

import torch

# x = torch.tensor([[0, 1, 0], [2, 0, 3]])
# out_tensor = torch.empty((2), dtype=torch.int64) # create an empty tensor with the desired shape and dtype
#
# indices = torch.nonzero(x, out=out_tensor) # store the indices in the out_tensor
#
# print(out_tensor)
# #
# # import numpy as np
# x = np.array([1,2,3,4,0,1])
# ivy.set_numpy_backend()
# indices = ivy.nonzero(x,dtype = float) # store the indices in the out_tensor
# print(type(indices[0][0].item()))
#



