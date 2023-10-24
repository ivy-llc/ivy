import ivy
from ivy.functional.frontends.numpy import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def issparse(x):
    return ivy.is_ivy_sparse_array(x)
