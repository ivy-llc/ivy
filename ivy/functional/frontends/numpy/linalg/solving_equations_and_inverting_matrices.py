# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs

#*************************Important***************************************
# module level unsupported dtype has been enabled for this module, if a new
# function is added, and it doesn't adhere to the module level specification
# then attribute `override` must be assigned to it w±o the appropriate decorator



# solve
@to_ivy_arrays_and_back
def solve(a, b):
    a, b = promote_types_of_numpy_inputs(a, b)
    return ivy.solve(a, b)


# inv
@to_ivy_arrays_and_back
def inv(a):
    return ivy.inv(a)


# pinv
# TODO: add hermitian functionality
@to_ivy_arrays_and_back
def pinv(a, rtol=1e-15, hermitian=False):
    return ivy.pinv(a, rtol=rtol)
