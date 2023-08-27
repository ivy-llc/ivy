import ivy
import ivy.functional.frontends.torch as ivy_torch
import numpy as np
import torch

ivy.set_backend("torch")

x = np.array([[0, 0], [0, 0]], dtype=np.float64)
idx = np.array([[0, 0]], dtype=np.int64)
vals = np.array([[1, 1]], dtype=np.float64)
axis = 0

ivy_x = ivy.array(x)
ivy_idx = ivy.array(idx)
ivy_vals = ivy.array(vals)


torch_x = torch.tensor(x)
torch_idx = torch.tensor(idx)
torch_vals = torch.tensor(vals)


ivy_out = ivy.put_along_axis(ivy_x, ivy_idx, ivy_vals, axis, mode="add")
print(ivy_out)

torch_out = torch.scatter_reduce(torch_x, axis, torch_idx, torch_vals, "sum")
print(torch_out)

torch_frontend_out = ivy_torch.scatter_reduce(ivy_x, axis, ivy_idx, ivy_vals, "add")
print(torch_frontend_out)


# import hypothesis.strategies as st
# import hypothesis.extra.numpy as hnp


# for _ in range(10):
#     data_type = st.sampled_from([np.float32, np.float64, np.int32, np.int64]).example()

#     arr = hnp.arrays(
#         dtype=data_type,
#         shape=hnp.array_shapes(min_dims=1, min_side=3),
#         elements=st.integers(0, 100),
#     ).example()

#     idx_shape = list(arr.shape)
#     axis = st.integers(0, len(arr.shape) - 1).example()
#     idx_shape[axis] = 1

#     idx = hnp.arrays(
#         dtype=np.int64,
#         shape=idx_shape,
#         elements=st.integers(0, len(arr.shape) - 1),
#     ).example()

#     vals = hnp.arrays(
#         dtype=data_type,
#         shape=idx_shape,
#         elements=st.integers(0, 100),
#     ).example()

#     # print(f"arr: \n{arr}")
#     # print(f"idx: \n{idx}")
#     # print(f"vals: \n{vals}")
#     # print(f"axis: \n{axis}")

#     ivy_out = ivy.put_along_axis(arr, idx, vals, axis)

#     paddle_arr = paddle.to_tensor(arr)
#     paddle_idx = paddle.to_tensor(idx)
#     paddle_vals = paddle.to_tensor(vals)

#     # torch_arr = torch.tensor(arr)
#     # torch_idx = torch.tensor(idx)
#     # torch_vals = torch.tensor(vals)

#     paddle_out = paddle.put_along_axis(paddle_arr, paddle_idx, paddle_vals, axis)
#     # torch_out = torch.scatter(torch_arr, axis, torch_idx, torch_vals)

#     # print(ivy_out)
#     # print(torch_out)
#     result = np.allclose(ivy_out, paddle_out.numpy())
#     if not result:
#         print("==========================================")
#         print(f"TRIAL {_} FAILED")
#         print(f"arr: \n{arr}")
#         print(f"idx: \n{idx}")
#         print(f"vals: \n{vals}")
#         print(f"axis: \n{axis}")

#         print(ivy_out)
#         print(paddle_out)
#         print("==========================================")
#     else:
#         print(f"TRIAL {_} PASSED")
#     print("==========================================")
