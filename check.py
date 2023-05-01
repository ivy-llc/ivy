import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import ivy
import torch

start = ivy.array([0])

stop=ivy.array([[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
num=2
axis=0
ivy.set_backend('torch')
print(ivy.linspace(start, stop, num, axis=axis))


ivy.set_backend('numpy')
print(ivy.linspace(start, stop, num, axis=axis))

#
#
# input=np.array([-2])
# pad_width=((0, 3),)
# mode="linear_ramp"
# stat_length=((2, 2),)
# constant_values=((0, 0),)
# end_values=((0, 1),)
# reflect_type="even"
#
# # ivy.set_backend('jax')
# # print(jnp.pad(input, pad_width, mode=mode, end_values = end_values
# #         ))
#
# # ivy.set_backend('numpy')
# print(np.pad(input, pad_width, mode=mode,end_values = end_values
#         ))
#
#
#
# #
# #
# # AssertionError:  the results from backend jax and ground truth framework numpy do not match
# # E            [-2 -1 -1  1]!=[-2 -1  0  1]
#
#start = (0,0)
#stop=torch.tensor([[0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0]])
#num=2
#axis=0
#dtype=torch.int32
# # #print(torch.linspace())
#
# inp = ivy.array([[[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0]],
#
#                        [[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0]],
#
#                        [[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0]],
#
#                        [[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0]],
#
#                        [[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0]]])
# pad_width = 1
# stat_length = 2
# constant_values = 0
# end_values = 0
#
# # ivy.set_backend('torch')
# # res = ivy.pad(inp, pad_width, mode='linear_ramp',
# #               constant_values=constant_values,
# #               stat_length=stat_length,
# #               reflect_type='even',
# #               end_values=end_values)
# #
# # res1 = np.pad(inp, pad_width, 'linear_ramp', end_values=end_values)
#
# # dtype_and_input_and_other=(['int64'],
# # E                array([0, 1]),
# # E                ((1, 3),),
# # E                ((2, 2),),
# # E                ((0, 0),),
# # E                ((0, 0),),
# # E                'wrap'),
# # E               reflect_type='even',
# #
# # ivy.array([[[0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0]],
# #
# #                        [[0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0]],
# #
# #                        [[0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0]],
# #
# #                        [[0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0]],
# #
# #                        [[0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0],
# #                         [0, 0, 0, 0, 0]]]),
# #         pad_width=1,
# #         mode="linear_ramp",
# #         stat_length=2,
# #         constant_values=0,
# #         end_values=0,
# #         reflect_type="even",
# #
# # ivy.array([[[[0]]]]),
# #         pad_width=((0, 0), (0, 0), (0, 0), (0, 1)),
# #         mode="linear_ramp",
# #         stat_length=((2, 2), (2, 2), (2, 2), (2, 2)),
# #         constant_values=((0, 0), (0, 0), (0, 0), (0, 0)),
# #         end_values=0,
# #         reflect_type="even",
# # ivy.array([[[0]]]),
# #         pad_width=((0, 0), (0, 1), (0, 0)),
# #         mode="linear_ramp",
# #         stat_length=((2, 2), (2, 2), (2, 2)),
# #         constant_values=((0, 0), (0, 0), (0, 0)),
# #         end_values=0,
# #         reflect_type="even",
#
# # ivy.array([[[[0]]]]),
# #         pad_width=((0, 0), (0, 0), (0, 1), (0, 0)),
# #         mode="linear_ramp",
# #         stat_length=((2, 2), (2, 2), (2, 2), (2, 2)),
# #         constant_values=((0, 0), (0, 0), (0, 0), (0, 0)),
# #         end_values=0,
# #         reflect_type="even",