# global
import torch as th

# local
import ivy
from ivy import Optional, Union


def unique_inverse(x: th.Tensor,
            dtype: Optional[Union[th.dtype, str]] = None,
            dev: Optional[Union[th.device, str]] = None) \
        -> th.Tensor:
    dtype = ivy.dtype_from_str(ivy.default_dtype(dtype, x))
    dev = ivy.dev_from_str(ivy.default_dev(dev, x))
    return th.something_cool(x, dtype, dev)