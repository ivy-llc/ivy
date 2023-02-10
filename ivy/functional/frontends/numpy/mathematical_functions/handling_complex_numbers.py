# global
import ivy
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


# @handle_numpy_dtype
# @to_ivy_arrays_and_back
# @from_zero_dim_arrays_to_scalar
def _imag(val):
    # if dtype:
    #     val = [ivy.astype(ivy.array(v), ivy.as_ivy_dtype(dtype)) for v in val]
    return ivy.imag(val)