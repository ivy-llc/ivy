# global
from typing import Union

import numpy as np

# local
from ivy.functional.backends.numpy import ivy_dtype_dict


def is_native_dtype(dtype_in: Union[np.dtype, str], /) -> bool:
    if dtype_in in ivy_dtype_dict:
        return True
    else:
        return False
