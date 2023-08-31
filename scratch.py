import ivy
import numpy as np
import torch
import paddle

from typing import Optional


def put_along_axis(
    arr: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    axis: int,
    /,
    *,
    mode: str = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mode:
        if mode == "add":
            ret = torch.scatter_add(arr, axis, indices, values, out=out)
        elif mode == "mul":
            ret = torch.scatter_reduce(
                arr, axis, indices, values, reduce="prod", out=out
            )
        elif mode == "assign":
            ret = torch.scatter(arr, axis, indices, values, out=out)
        else:
            ret = torch.scatter_reduce(arr, axis, indices, values, reduce=mode, out=out)
    elif mode is None:
        ret = torch.scatter(arr, axis, indices, values, out=out)
    return ret


ivy.set_backend("torch")

# x = np.array([[-1, -1], [-1, -1]], dtype=np.float64)
# idx = np.array([[0, 0]], dtype=np.int64)
# vals = np.array([[1, 1]], dtype=np.float64)

# x = np.array([[0, 0, 8],[0, 0, 0],[0, 0, 0]], dtype=np.float32)
# idx = np.array([[0, 0, 0]], dtype=np.int64)
# vals = np.array([[2, 2, 2]], dtype=np.float32)

x = np.array([[0, 0], [0, 0]], dtype=np.float32)
idx = np.array([[0, 0]], dtype=np.int64)
vals = np.array([[1, 1]], dtype=np.float32)

# x = np.array([0], dtype=np.float32)
# idx = np.array([0, 0], dtype=np.int64)
# vals = np.array([1, 1], dtype=np.float32)

axis = 0

ivy_x = ivy.array(x)
ivy_idx = ivy.array(idx)
ivy_vals = ivy.array(vals)


torch_x = torch.tensor(x)
torch_idx = torch.tensor(idx)
torch_vals = torch.tensor(vals)

paddle_x = paddle.to_tensor(x)
paddle_idx = paddle.to_tensor(idx)
paddle_vals = paddle.to_tensor(vals)

# ret_torch = torch.scatter_reduce(torch_x, axis, torch_idx, torch_vals, "sum")
# print(ret_torch)

# ivy_out = ivy.put_along_axis(ivy_x, ivy_idx, ivy_vals, axis, mode="add")
# print(ivy_out)

# print(ivy.array([ivy.prod([2,2,2,2])]))

ret = ivy.put_along_axis(ivy_x, ivy_idx, ivy_vals, axis, mode="sum")
print(ret)

torch_ret = torch.scatter_reduce(torch_x, axis, torch_idx, torch_vals, "sum")
print(torch_ret)

torch_backend_ret = put_along_axis(torch_x, torch_idx, torch_vals, axis, mode="sum")
print(torch_backend_ret)

paddle_ret = paddle.put_along_axis(
    paddle_x, paddle_idx, paddle_vals, axis, reduce="add"
)
print(paddle_ret)

# np.put_along_axis(x, idx, vals, axis)
# print(x)

# ret_ivy = ivy_x.scatter_reduce(axis, ivy_idx, ivy_vals, "sum")
# print(ret_ivy)

# torch_out = torch.scatter_reduce(torch_x, axis, torch_idx, torch_vals, "sum")
# print(torch_out)

# torch_frontend_out = ivy_torch.scatter_reduce(ivy_x, axis, ivy_idx, ivy_vals, "add")
# print(torch_frontend_out)
