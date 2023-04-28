from typing import Union, Optional, Tuple


def unravel_index(
    indices: Union[(None, tf.Variable)],
    shape: Tuple[int],
    /,
    *,
    out: Optional[Tuple[Union[(None, tf.Variable)]]] = None,
) -> Tuple[None]:
    raise NotImplementedError("mxnet.unravel_index Not Implemented")
