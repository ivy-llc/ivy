import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.numpy.func_wrapper import handle_numpy_dtype


@handle_numpy_dtype
@to_ivy_arrays_and_back
def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    ret = ivy.softmax(data, axis=axis)
    if dtype:
        ivy.utils.assertions.check_elem_in_list(
            dtype, ["float16", "float32", "float64"]
        )
        ret = ivy.astype(ret, dtype)
    return ret
