import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back
<<<<<<< HEAD


@to_ivy_arrays_and_back
def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    return ivy.softmax(data, axis=axis)
=======
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
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
