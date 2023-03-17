import ivy
from ivy.functional.frontends.mxnet.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def min(a, axis=None, out=None, keepdims=False, where=None):
    response = ivy.min(a, axis=axis, out=out, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        response = ivy.where(where, response, ivy.default(out, ivy.zeros_like(response)), out=out)
    return response