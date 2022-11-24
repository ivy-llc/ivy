import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def diagonal(x, offset, axis1, axis2):
    # ivy.assertions.check_equal(
    #   axis1 == axis2, message="Both axis values should not be the same"
    # )
    # ivy.assertions.check_equal(x.ndim, 2, message="x must be 2-dimensional")

    return ivy.diagonal(x, offset, axis1, axis2)
