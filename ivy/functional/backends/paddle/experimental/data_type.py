# global
from typing import Union

import paddle
from ivy.exceptions import IvyNotImplementedException

# local
from ivy.functional.backends.paddle import ivy_dtype_dict


def is_native_dtype(dtype_in: Union[paddle.dtype, str], /) -> bool:
    if dtype_in in ivy_dtype_dict:
        return True
    else:
        return False
