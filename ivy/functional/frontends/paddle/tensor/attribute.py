# global
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import (
    with_supported_dtypes,
)


@with_supported_dtypes(
    {
        "2.4.2 and below": (
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def is_complex(x):
    return ivy.is_complex_dtype(x)
