from typing import Union, Optional, Sequence, Tuple, NamedTuple, List
from numbers import Number
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version
import tensorflow as tf
import ivy


def moveaxis(
    a: Union[tf.Tensor, tf.Variable],
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.moveaxis(a, source, destination)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, backend_version)
def heaviside(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.cast(tf.experimental.numpy.heaviside(x1, x2), x1.dtype)


def flipud(
    m: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.flipud(m)


def vstack(
    arrays: Union[Sequence[tf.Tensor], Sequence[tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.vstack(arrays)


def hstack(
    arrays: Union[Sequence[tf.Tensor], Sequence[tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.hstack(arrays)


def rot90(
    m: Union[tf.Tensor, tf.Variable],
    /,
    *,
    k: int = 1,
    axes: Tuple[int, int] = (0, 1),
    out: Union[tf.Tensor, tf.Variable] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.rot90(m, k, axes)


def top_k(
    x: tf.Tensor,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: bool = True,
    out: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if not largest:
        indices = tf.experimental.numpy.argsort(x, axis=axis)
        indices = tf.experimental.numpy.take(
            indices, tf.experimental.numpy.arange(k), axis=axis
        )
        indices = tf.dtypes.cast(indices, tf.int32)
    else:
        x = -x
        indices = tf.experimental.numpy.argsort(x, axis=axis)
        indices = tf.experimental.numpy.take(
            indices, tf.experimental.numpy.arange(k), axis=axis
        )
        indices = tf.dtypes.cast(indices, tf.int32)
        x = -x
    topk_res = NamedTuple("top_k", [("values", tf.Tensor), ("indices", tf.Tensor)])
    val = tf.experimental.numpy.take_along_axis(x, indices, axis=axis)
    indices = tf.dtypes.cast(indices, tf.int64)
    return topk_res(val, indices)


def fliplr(
    m: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.fliplr(m)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16",)}, backend_version)
def i0(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.bessel_i0(x, name=None)


def vsplit(
    ary: Union[tf.Tensor, tf.Variable],
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.vsplit(ary, indices_or_sections)


def dsplit(
    ary: Union[tf.Tensor, tf.Variable],
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[Union[tf.Tensor, tf.Variable]]:
    if len(ary.shape) < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return tf.experimental.numpy.dsplit(ary, indices_or_sections)


def atleast_1d(
    *arys: Union[tf.Tensor, tf.Variable, bool, Number],
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.atleast_1d(*arys)


def dstack(
    arrays: Union[Sequence[tf.Tensor], Sequence[tf.Variable]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.dstack(arrays)


def atleast_2d(
    *arys: Union[tf.Tensor, tf.Variable],
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.atleast_2d(*arys)


def atleast_3d(
    *arys: Union[tf.Tensor, tf.Variable, bool, Number],
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.atleast_3d(*arys)


def take_along_axis(
    arr: Union[tf.Tensor, tf.Variable],
    indices: Union[tf.Tensor, tf.Variable],
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if len(arr.shape) != len(indices.shape):
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {len(arr.shape)} vs {len(indices.shape)}"
        )
    indices = tf.dtypes.cast(indices, tf.int32)
    if mode not in ["clip", "fill", "drop"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are 'clip', 'fill', 'drop'."
        )
    arr_shape = arr.shape
    if axis < 0:
        axis += len(arr.shape)
    if mode == "clip":
        max_index = arr.shape[axis] - 1
        indices = tf.clip_by_value(indices, 0, max_index)
    elif mode == "fill" or mode == "drop":
        if "float" in str(arr.dtype):
            fill_value = tf.constant(float("nan"), dtype=arr.dtype)
        elif "uint" in str(arr.dtype):
            fill_value = tf.constant(arr.dtype.max, dtype=arr.dtype)
        else:
            fill_value = tf.constant(-arr.dtype.max - 1, dtype=arr.dtype)
        indices = tf.where((indices < 0) | (indices >= arr.shape[axis]), -1, indices)
        arr_shape = list(arr_shape)
        arr_shape[axis] = 1
        fill_arr = tf.fill(arr_shape, fill_value)
        arr = tf.concat([arr, fill_arr], axis=axis)
    return tf.experimental.numpy.take_along_axis(arr, indices, axis)


def hsplit(
    ary: Union[tf.Tensor, tf.Variable],
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.experimental.numpy.hsplit(ary, indices_or_sections)


def broadcast_shapes(
    *shapes: Union[List[int], List[Tuple]],
) -> Tuple[int, ...]:
    if len(shapes) > 1:
        desired_shape = tf.broadcast_dynamic_shape(shapes[0], shapes[1])
        if len(shapes) > 2:
            for i in range(2, len(shapes)):
                desired_shape = tf.broadcast_dynamic_shape(desired_shape, shapes[i])
    else:
        return [shapes[0]]
    return tuple(desired_shape.numpy().tolist())


def expand(
    x: Union[tf.Tensor, tf.Variable],
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    shape = list(shape)
    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    return tf.broadcast_to(x, shape)


def fold(
    input: Union[tf.Tensor, tf.Variable],
    output_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    /,
    *,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Union[tf.Tensor, tf.Variable]:
    if len(input.shape) == 2:
        input = tf.expand_dims(input, axis=0)

    output_size, kernel_size, dilation, padding, stride = map(
        lambda x: [x, x] if isinstance(x, int) else x,
        [output_size, kernel_size, dilation, padding, stride]
    )

    batch_size, channels, l = input.shape
    input_channels = channels // (kernel_size[0] * kernel_size[1])

    output_channels = input_channels
    output_height = output_size[0]
    output_width = output_size[1]

    input = tf.reshape(input, (batch_size, input_channels, kernel_size[0], kernel_size[1], -1))

    input = tf.transpose(input, perm=[0, 1, 4, 2, 3])

    unfolded = tf.nn.conv_transpose(
        input,
        tf.eye(output_channels, dtype=input.dtype),
        output_shape=[batch_size, output_channels, output_height, output_width],
        strides=[1, 1, stride[0], stride[1]],
        padding=[[0, 0], [0, 0], [padding[0], padding[0]], [padding[1], padding[1]]],
        dilations=[1, 1, dilation[0], dilation[1]]
    )
    return unfolded


def unfold(
    input: Union[tf.Tensor, tf.Variable],
    kernel_size: Union[int, Tuple[int, int]],
    /,
    *,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Union[tf.Tensor, tf.Variable]:
    if len(input.shape) == 3:
        input = tf.expand_dims(input, axis=0)
    kernel_size, dilation, padding, stride = map(
        lambda x: [x, x] if isinstance(x, int) else x,
        [kernel_size, dilation, padding, stride]
    )
    sizes, strides, rates = map(lambda x: [1, *x, 1], [kernel_size, stride, dilation])
    input = tf.transpose(
        tf.pad(input, [[0, 0], [0, 0], [padding[0]]*2, [padding[1]]*2]),
        perm=[0, 2, 3, 1],
    )
    patches = tf.image.extract_patches(input, sizes, strides, rates, 'VALID')
    output = tf.reshape(
        tf.transpose(patches, perm=[0, 3, 1, 2]),
        [input.shape[0], kernel_size[0] * kernel_size[1] * input.shape[-1], -1],
    )
    return output
