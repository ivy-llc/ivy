# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    from_zero_dim_arrays_to_scalar,
)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def eigvalsh(a, /, UPLO="L"):
    return ivy.eigvalsh(a, UPLO=UPLO)
<<<<<<< HEAD
=======


@to_ivy_arrays_and_back
def eig(a):
    return ivy.eig(a)


@from_zero_dim_arrays_to_scalar
def eigh(a, /, UPLO="L"):
    return ivy.eigh(a, UPLO=UPLO)
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
