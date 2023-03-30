# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
import ivy.functional.frontends.scipy as scipy_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def euclidean(u, v, w=None):
    if(w is None):
        w = 1 
    return ivy.linalg.vector_norm(w*(u-v), ord=2)