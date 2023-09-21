from typing import Optional, Tuple, Union

import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


def unravel_index(
    indices: Union[(None, mx.ndarray.NDArray)],
    shape: Tuple[int],
    /,
    *,
    out: Optional[Tuple[Union[(None, mx.ndarray.NDArray)]]] = None,
) -> Tuple[None]:
    raise IvyNotImplementedException()
