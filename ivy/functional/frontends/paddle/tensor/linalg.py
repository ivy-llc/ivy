# global
import ivy
from ivy.functional.frontends.paddle import promote_types_of_paddle_inputs
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)


# matmul
@to_ivy_arrays_and_back
def matmul(x, y, transpose_x=False, transpose_y=False, name=None):
    x, y = promote_types_of_paddle_inputs(x, y)
    return ivy.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)


# eig
@to_ivy_arrays_and_back
def eig(x, name=None):
    return ivy.eig(x)


# eigvals
@to_ivy_arrays_and_back
def eigvals(x, name=None):
    return ivy.eigvals(x)


# eigvalsh
@to_ivy_arrays_and_back
def eigvalsh(x, UPLO="L", name=None):
    return ivy.eigvalsh(x, UPLO=UPLO)


# eigh
@to_ivy_arrays_and_back
def eigh(x, UPLO="L", name=None):
    return ivy.eigh(x, UPLO=UPLO)


# pinv
@to_ivy_arrays_and_back
def pinv(x, rcond=1e-15, hermitian=False, name=None):
    # TODO: Add hermitian functionality
    return ivy.pinv(x, rtol=rcond)
