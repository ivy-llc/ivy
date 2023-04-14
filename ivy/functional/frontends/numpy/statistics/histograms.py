import ivy
from ivy.functional.frontends.numpy import handle_numpy_out, to_ivy_arrays_and_back, from_zero_dim_arrays_to_scalar,handle_numpy_dtype,handle_numpy_casting
import ivy.functional.frontends.numpy as np_frontend

@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
@handle_numpy_dtype
@handle_numpy_casting

def bincount(x, /, weights=None, minlength=None):
    x_list = []
    for i in range(x.shape[0]):
        x_list.append(int(x[i]))
    max_val = int(ivy.max(ivy.array(x_list)))
    ret = [x_list.count(i) for i in range(0, max_val + 1)]
    ret = ivy.array(ret)
    ret = ivy.astype(ret, ivy.as_ivy_dtype(ivy.int64))
    return ret



