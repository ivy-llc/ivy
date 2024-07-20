import tensorflow

from typing import Optional
from typing import Sequence
from typing import Literal
from typing import Union

from .tensorflow__helpers import tensorflow__get_size_bknd
from .tensorflow__helpers import tensorflow_exists_bknd
from .tensorflow__helpers import tensorflow_inplace_update
from .tensorflow__helpers import tensorflow_shape


def tensorflow_interpolate(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    size: Union[Sequence[int], int],
    /,
    *,
    mode: Literal[
        "linear",
        "bilinear",
        "trilinear",
        "nd",
        "nearest",
        "area",
        "nearest_exact",
        "tf_area",
        "tf_bicubic",
        "bicubic",
        "mitchellcubic",
        "lanczos3",
        "lanczos5",
        "gaussian",
    ] = "linear",
    scale_factor: Optional[Union[Sequence[int], int]] = None,
    recompute_scale_factor: Optional[bool] = None,
    align_corners: bool = False,
    antialias: bool = False,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    input_size = tensorflow_shape(x)[2:]
    dims = len(input_size)
    size, _ = tensorflow__get_size_bknd(scale_factor, size, dims, input_size)
    if all(a == b for a, b in zip(size, input_size)):
        ret = x
    else:
        remove_dim = False
        if mode in ["linear", "tf_area", "lanczos3", "lanczos5", "nearest-exact"]:
            if dims == 1:
                size = (1,) + tuple(size)
                x = tensorflow.expand_dims(x, axis=-2)
                dims = 2
                remove_dim = True
            mode = (
                "bilinear"
                if mode == "linear"
                else (
                    "area"
                    if mode == "tf_area"
                    else "nearest" if mode == "nearest-exact" else mode
                )
            )
        if mode == "tf_bicubic":
            mode = "bicubic"
        x = tensorflow.transpose(x, (0, *range(2, dims + 2), 1))
        ret = tensorflow.transpose(
            tensorflow.cast(
                tensorflow.image.resize(x, size=size, method=mode, antialias=antialias),
                x.dtype,
            ),
            (0, dims + 1, *range(1, dims + 1)),
        )
        if remove_dim:
            ret = tensorflow.squeeze(ret, axis=-2)
    if tensorflow_exists_bknd(out):
        return tensorflow_inplace_update(out, ret)
    return ret
