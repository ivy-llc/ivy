from typing import Union

from .tensorflow__helpers import tensorflow_as_ivy_dtype_1


def tensorflow_as_ivy_dtype(dtype_in: Union[str, str], /):
    return tensorflow_as_ivy_dtype_1(dtype_in)
