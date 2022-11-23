import ivy


def diagonal(x, offset, axis1, axis2):
    ivy.assertions.check_equal(
        axis1 == axis2, message="Both axis values should not be the same"
    )
    ivy.assertions.check_equal(x.ndim, 2, message="a must be 2-dimensional")

    return ivy.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
