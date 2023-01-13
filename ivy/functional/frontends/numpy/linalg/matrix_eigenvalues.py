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
