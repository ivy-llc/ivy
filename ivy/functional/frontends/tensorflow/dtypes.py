# local

import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def cast(x, dtype, name=None):
    assert ivy.can_cast(x.dtype, dtype), "Cannot cast from {} to {}".format(
        x.dtype, dtype
    )
    return ivy.astype(x, dtype, copy=False)
