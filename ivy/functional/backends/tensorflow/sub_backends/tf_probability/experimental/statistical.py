from typing import Optional, Sequence, Tuple, Union
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.backends.numpy.experimental.statistical import (
    _handle_axis,
    _quantile,
    _validate_quantile,
)
import tensorflow_probability as tfp
import tensorflow as tf
from .... import backend_version


def histogram(
    a: tf.Tensor,
    /,
    *,
    bins: Optional[Union[int, tf.Tensor]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[tf.DType] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[tf.Tensor] = None,
    density: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    min_a = tf.reduce_min(a)
    max_a = tf.reduce_max(a)
    if isinstance(bins, tf.Tensor) and range:
        raise ivy.exceptions.IvyException(
            "Must choose between specifying bins and range or bin edges directly"
        )
    if range:
        if isinstance(bins, int):
            bins = tf.cast(
                tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype
            )
    elif isinstance(bins, int):
        range = (min_a, max_a)
        bins = tf.cast(
            tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype
        )
    if tf.shape(bins)[0] < 2:
        raise ivy.exceptions.IvyException("bins must have at least 1 bin (size > 1)")
    if min_a < bins[0] and not extend_lower_interval:
        raise ivy.exceptions.IvyException(
            "Values of x outside of the intervals cause errors in tensorflow backend. "
            "Consider using extend_lower_interval to deal with this."
        )
    if max_a > bins[-1] and not extend_upper_interval:
        raise ivy.exceptions.IvyException(
            "Values of x outside of the intervals cause errors in tensorflow backend. "
            "Consider using extend_upper_interval to deal with this."
        )
    ret = tfp.stats.histogram(
        x=a,
        edges=bins,
        axis=axis,
        weights=weights,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype,
        name="histogram",
    )
    if density:
        pass
    # TODO: Tensorflow native dtype argument is not working
    if dtype:
        ret = tf.cast(ret, dtype)
        bins = tf.cast(bins, dtype)
    # TODO: weird error when returning bins: return ret, bins
    return ret


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float",
            "complex",
        )
    },
    backend_version,
)
def median(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.stats.percentile(
        input,
        50.0,
        axis=axis,
        interpolation="midpoint",
        keepdims=keepdims,
    )


def nanmedian(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if overwrite_input:
        copied_input = tf.identity(input)
        return _nanmedian_helper(copied_input, axis, keepdims)

    else:
        result = _nanmedian_helper(input, axis, keepdims)
        return result


def _nanmedian_helper(input, axis=None, keepdims=False):
    """The approach to Handle Nans in single dimensional plus multi-dimensional
    inputs are composed on two-parts.

    PART 1:  In this part, you have axis=None, it means we have to work on
    flattened data, we don't need to work on different axis.there are two cases here

    Case 1: which is if our input data does contain all the Nans or not,
    if our input have just Nans (means no numbers) then we'll not use
    temp[~tf.math.is_nan(temp)] function with our input because it will remove all Nans
    and we get empty tensor and this raise an error when it sent to percentile function,
    in this case we need to keep this input but just we flatten the input and percentile
    function returns nan if it find nan in median and here all the input is nan then we
    get our result.

    Case 2: if we have a number (0.4, 0.3, 0. ,1., 2., .....) with nans then we use this
    function temp[~tf.math.is_nan(temp)], it will return a tensor by extracting the nans
    and just keeping the values, but remember the returned tensor will be flattened and
    axis=None work on flattene inputs, so in this case we are also on same page :)

    for example: [[12.0 ,4.0 ,ivy.nan], [ivy.nan, ivy.nan,2.2]] => returned:
    [12.0 ,4.0, 2.2] now this will be our new input in percentile function.

    PART 2: In this case you have to do more work because we now don't allow to work
    directly on flattened data, Here are two cases also.

    CASE 1: we need to consider axis parameter here, but percentile axis does work
    differently and we don't have median function in tensorflow yet, so we need to make
    our input data compatible to the axis, then we compute nanmedian along that specific
    axis. we transpose the input data according to our axis, axis can be (0,), (1,),
    (0,1), (0,1,2) and input can be multi-dimensional, so we need to take care of edge
    cases before making it compatible.

    CASE 2: Here the main Nan handling part comes, you can only use 1D inputs here so we
    have to flatten the input then we have jump parameter which is use to say how many
    iterations we want to make because we have to calculate the row-wise median along
    axis=None now, so we slice out some data from the flattened input and then we use
    that 1D Input to remove the nans and use it in our percentile.

    For example: input = [[ivy.nan, 3, ivy.nan, 7],[4, ivy.nan,6, 9]], axis=1

    flatten data -> [[nan  3. nan  7.  4. nan  6.  9.]]
    num_jumps -> 2 because we have to slice out this in (1, 4) and (1,4),
    then it works same as PART 1 CASE 1 AND CASE 2.
    now for first slice we get -> 5.0 and for second we get -> 6.0, these calculated
    along axis=1 now we append the data into result, so to make the shape of result
    compatible with the numpy output, we reshaped it.

    the result which we get from our _nanmedian_helper = [5., 6.]
    """
    dtype = input.dtype
    temp = tf.cast(input, tf.float64)
    num_dim = tf.rank(temp)
    keepdim_shape = tf.shape(temp)
    q = 50.0

    # PART 1
    if axis is None:
        # PART 1 CASE 1
        if tf.reduce_all(tf.math.is_nan(temp)):
            temp = tf.reshape(temp, shape=(1, -1))
        else:
            # PART 1 CASE 2
            temp = temp[~tf.math.is_nan(temp)]

        ret = tfp.stats.percentile(
            temp,
            q,
            axis=axis,
            interpolation="midpoint",
            keepdims=keepdims,
        )
        if dtype in [tf.int32, tf.int64, tf.float64]:
            ret = tf.cast(ret, dtype=tf.float64)
        elif dtype in [tf.float16, tf.bfloat16]:
            ret = tf.cast(ret, dtype=tf.float16)
        else:
            ret = tf.cast(ret, dtype=tf.float32)
        return ret

    axis = [axis] if isinstance(axis, int) else list(axis)
    # PART 2 CASE 1
    for i in axis:
        keepdim_shape = tf.tensor_scatter_nd_update(keepdim_shape, [[i]], [1])
    axis = [num_dim + x if x < 0 else x for x in axis]
    axis.sort()
    dimension = tf.size(temp.shape)
    while tf.size(axis) > 0:
        axis1 = axis[0]
        for axis2 in range(axis1 + 1, dimension):
            temp = tf.transpose(
                temp,
                perm=tf.tensor_scatter_nd_update(
                    tf.range(tf.rank(temp)), [[axis1], [axis2]], [axis2, axis1]
                ),
            )
            axis1 = axis2
        axis = [x - 1 for x in axis]
        axis.pop(0)
        dimension = dimension - 1
    temp = tf.reshape(
        temp, shape=tf.concat([tf.shape(temp)[: (dimension - len(axis))], [-1]], axis=0)
    )

    tensor = tf.reshape(temp, shape=(1, -1))
    shape = temp.shape
    dim = temp.ndim
    slice_size = shape[len(shape) - 1]
    num_jumps = 1
    result = []

    if slice_size == 1:
        if dim == 2 and input.shape[0] == 1:
            return tensor
        if dim > 2 and input.shape[0] == 1:
            return tf.reshape(tensor, shape=input.shape)

        tensor = tf.reshape(tensor, shape=shape[:-1])
        return tensor
    # PART 2 CASE 2
    i = dim
    while i > 1:
        num_jumps *= shape[len(shape) - i]
        i -= 1

    for i in range(num_jumps):
        start = i * slice_size
        end = (i + 1) * slice_size
        arr = tensor[:, start:end]
        if tf.reduce_all(tf.math.is_nan(arr)):
            arr = tf.reshape(arr, shape=(1, -1))
        else:
            arr = arr[~tf.math.is_nan(arr)]

        ret = tfp.stats.percentile(
            arr, q, axis=None, interpolation="midpoint", keepdims=keepdims
        )
        if keepdims:
            ret = tf.squeeze(ret)

        result.append(ret)

    result = tf.reshape(result, shape=shape[:-1])

    if keepdims:
        keepdim_shape = tuple(keepdim_shape)
        result = tf.reshape(result, shape=keepdim_shape)

    if dtype in [tf.int32, tf.int64, tf.float64]:
        result = tf.cast(result, dtype=tf.float64)
    elif dtype in [tf.float16, tf.bfloat16]:
        result = tf.cast(result, dtype=tf.float16)
    else:
        result = tf.cast(result, dtype=tf.float32)

    return result


def _compute_quantile_wrapper(
    x,
    q,
    axis=None,
    keepdims=False,
    interpolation="linear",
):
    if not _validate_quantile(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if interpolation in [
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
        "nearest_jax",
    ]:
        if interpolation == "nearest_jax":
            return _handle_axis(x, q, _quantile, keepdims=keepdims, axis=axis)
        else:
            axis = tuple(axis) if isinstance(axis, list) else axis

            return tfp.stats.percentile(
                x,
                tf.math.multiply(q, 100),
                axis=axis,
                interpolation=interpolation,
                keepdims=keepdims,
            )
    else:
        raise ValueError(
            "Interpolation must be 'linear', 'lower', 'higher', 'midpoint' or 'nearest'"
        )


def quantile(
    a: Union[tf.Tensor, tf.Variable],
    q: Union[tf.Tensor, float],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # added the nearest_jax mode to enable jax-like calculations for method="nearest"
    return _compute_quantile_wrapper(
        a,
        q,
        axis=axis,
        keepdims=keepdims,
        interpolation=interpolation,
    )
