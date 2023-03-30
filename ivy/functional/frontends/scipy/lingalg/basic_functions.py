# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
import ivy.functional.frontends.scipy as scipy_frontend
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def inv(a):
    return ivy.inv(a)
