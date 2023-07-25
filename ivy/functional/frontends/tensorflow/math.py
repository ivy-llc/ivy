# global
import ivy
from ivy import with_supported_dtypes, with_unsupported_dtypes
from ivy.functional.frontends.tensorflow import check_tensorflow_casting
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
    to_ivy_dtype,
)


@with_supported_dtypes(
    {"2.13.0 and below": ("float16", "float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
@to_ivy_arrays_and_back
def imag(input, name=None):
    return ivy.imag(input)


@to_ivy_arrays_and_back
def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
    return ivy.sum(inputs, axis=0)


@to_ivy_arrays_and_back
def add(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.add(x, y)


@to_ivy_arrays_and_back
def exp(x, name=None):
    return ivy.exp(x)


@to_ivy_arrays_and_back
def expm1(x, name=None):
    return ivy.expm1(x)


@to_ivy_arrays_and_back
def sqrt(x, name=None):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def negative(x, name=None):
    return ivy.negative(x)


@to_ivy_arrays_and_back
def argmax(input, axis, output_type=None, name=None):
    output_type = to_ivy_dtype(output_type)
    if output_type in ["uint16", "int16", "int32", "int64"]:
        return ivy.astype(ivy.argmax(input, axis=axis), output_type)
    else:
        return ivy.astype(ivy.argmax(input, axis=axis), "int64")


@to_ivy_arrays_and_back
def asinh(x, name="asinh"):
    return ivy.asinh(x)


@handle_tf_dtype
@to_ivy_arrays_and_back
def confusion_matrix(
    labels, predictions, num_classes=None, weights=None, dtype=ivy.int32, name=None
):
    labels = ivy.astype(
        ivy.squeeze(ivy.array(labels), axis=None), ivy.int64, copy=False
    )
    predictions = ivy.astype(
        ivy.squeeze(ivy.array(predictions), axis=None), ivy.int64, copy=False
    )
    # failsafe for (1,) array will be squeeze to 0-dim
    labels = ivy.expand_dims(labels, axis=-1) if labels.ndim == 0 else labels
    predictions = (
        ivy.expand_dims(predictions, axis=-1) if predictions.ndim == 0 else predictions
    )

    # Sanity check (potential optimization)
    ivy.utils.assertions.check_greater(
        labels, 0, allow_equal=True, message="labels contains negative values"
    )
    ivy.utils.assertions.check_greater(
        predictions, 0, allow_equal=True, message="predictions contains negative values"
    )

    if num_classes is None:
        num_classes = max(ivy.max(labels), ivy.max(predictions)) + 1
    else:
        num_classes_int64 = ivy.astype(ivy.array(num_classes), ivy.int64, copy=False)
        ivy.utils.assertions.check_less(
            labels, num_classes_int64, message="labels out of bound"
        )
        ivy.utils.assertions.check_less(
            predictions, num_classes_int64, message="predictions out of bound"
        )

    if weights is not None:
        weights = ivy.array(weights)
        ivy.utils.assertions.check_equal(
            ivy.shape(predictions),
            ivy.shape(weights),
            message="weights shape do not match predictions",
            as_array=False,
        )
        weights = ivy.astype(weights, dtype, copy=False)

    shape = ivy.stack([num_classes, num_classes])
    indices = ivy.stack([labels, predictions], axis=1)
    values = ivy.ones_like(predictions, dtype=dtype) if weights is None else weights
    return ivy.scatter_nd(indices, values, shape=shape)


@handle_tf_dtype
@to_ivy_arrays_and_back
def count_nonzero(input, axis=None, keepdims=None, dtype=ivy.int64, name=None):
    x = ivy.array(input)
    if keepdims is None:
        keepdims = False
    zero = ivy.zeros(ivy.shape(x), dtype=x.dtype)
    return ivy.astype(
        ivy.sum(
            ivy.astype(ivy.not_equal(x, zero), ivy.int64),
            axis=axis,
            keepdims=keepdims,
        ),
        dtype,
        copy=False,
    )


def cumprod(x, axis, exclusive=False, reverse=False, name=None):
    return ivy.astype(
        ivy.cumprod(x, axis=axis, exclusive=exclusive, reverse=reverse), x.dtype
    )


def cumsum(x, axis, exclusive=False, reverse=False, name=None):
    return ivy.astype(
        ivy.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse), x.dtype
    )


@to_ivy_arrays_and_back
def divide(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.divide(x, y)


@to_ivy_arrays_and_back
def divide_no_nan(x, y, name="divide_no_nan"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.where(
        y == 0,
        ivy.array(0.0, dtype=ivy.promote_types(x.dtype, y.dtype)),
        x / y,
    )


@to_ivy_arrays_and_back
def maximum(x, y, name=None):
    return ivy.maximum(x, y)


@to_ivy_arrays_and_back
def erfcinv(x, name="erfcinv"):
    return 1 / (1 - ivy.erf(x))


@to_ivy_arrays_and_back
def is_inf(x, name=None):
    return ivy.isinf(x)


@to_ivy_arrays_and_back
def is_non_decreasing(x, name="is_non_decreasing"):
    if ivy.array(x).size < 2:
        return ivy.array(True)
    if ivy.array(x).size == 2:
        return ivy.array([x[0] <= x[1]])
    return ivy.all(ivy.less_equal(x, ivy.roll(x, -1)))


@to_ivy_arrays_and_back
def is_strictly_increasing(x, name="is_strictly_increasing"):
    if ivy.array(x).size < 2:
        return ivy.array(True)
    if ivy.array(x).size == 2:
        return ivy.array(x[0] < x[1])
    return ivy.all(ivy.less(x, ivy.roll(x, -1)))


@to_ivy_arrays_and_back
def log_sigmoid(x, name=None):
    return -ivy.softplus(-x)


@to_ivy_arrays_and_back
def logical_not(x, name="logical_not"):
    return ivy.logical_not(x)


@to_ivy_arrays_and_back
def log1p(x, name=None):
    return ivy.log1p(x)


@to_ivy_arrays_and_back
def logical_and(x, y, name="LogicalAnd"):
    return ivy.logical_and(x, y)


@to_ivy_arrays_and_back
def logical_xor(x, y, name="LogicalXor"):
    return ivy.logical_xor(x, y)


@to_ivy_arrays_and_back
def logical_or(x, y, name="logical_or"):
    return ivy.logical_or(x, y)


@to_ivy_arrays_and_back
def multiply(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.multiply(x, y)


@to_ivy_arrays_and_back
def multiply_no_nan(x, y, name="multiply_no_nan"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.where(
        y == 0,
        ivy.array(0.0, dtype=ivy.promote_types(x.dtype, y.dtype)),
        x * y,
    )


@to_ivy_arrays_and_back
def polyval(coeffs, x, name=None):
    ivy.utils.assertions.check_isinstance(coeffs, list)
    x = ivy.array(x)
    if len(coeffs) < 1:
        return ivy.zeros_like(x, dtype=x.dtype)
    coeffs = [ivy.array(_) for _ in coeffs]
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + p * x
    return p


@to_ivy_arrays_and_back
def pow(x, y, name="pow"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.pow(x, y)


@to_ivy_arrays_and_back
def reciprocal(x, name="reciprocal"):
    return ivy.reciprocal(x)


@to_ivy_arrays_and_back
def reciprocal_no_nan(x, name="reciprocal_no_nan"):
    return ivy.where(
        x == 0,
        ivy.array(0.0, dtype=x.dtype),
        ivy.ones_like(x, dtype=x.dtype) / x,
    )


@to_ivy_arrays_and_back
def reduce_all(input_tensor, axis=None, keepdims=False, name="reduce_all"):
    return ivy.all(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def reduce_any(input_tensor, axis=None, keepdims=False, name="reduce_any"):
    return ivy.any(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def reduce_euclidean_norm(
    input_tensor, axis=None, keepdims=False, name="reduce_euclidean_norm"
):
    return ivy.vector_norm(
        input_tensor, axis=axis, keepdims=keepdims, ord=2
    )  # ord = '2' is the euclidean norm


@to_ivy_arrays_and_back
def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name="reduce_logsumexp"):
    # stable logsumexp trick
    max_input_tensor = ivy.max(input_tensor, axis=axis, keepdims=False)
    return (
        ivy.log(
            ivy.sum(
                ivy.exp(input_tensor - max_input_tensor),
                axis=axis,
                keepdims=keepdims,
            )
        )
        + max_input_tensor
    ).astype(input_tensor.dtype)


@to_ivy_arrays_and_back
def reduce_max(input_tensor, axis=None, keepdims=False, name="reduce_max"):
    return ivy.max(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def reduce_mean(input_tensor, axis=None, keepdims=False, name="reduce_mean"):
    if ivy.exists(axis):
        axis = ivy.to_list(axis)
    return ivy.mean(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def reduce_min(input_tensor, axis=None, keepdims=False, name="reduce_min"):
    return ivy.min(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def reduce_prod(input_tensor, axis=None, keepdims=False, name="reduce_prod"):
    return ivy.prod(input_tensor, axis=axis, keepdims=keepdims).astype(
        input_tensor.dtype
    )


@to_ivy_arrays_and_back
def reduce_std(input_tensor, axis=None, keepdims=False, name="reduce_std"):
    return ivy.std(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def reduce_sum(input_tensor, axis=None, keepdims=False, name="reduce_sum"):
    input_tensor = ivy.array(input_tensor)
    return ivy.sum(input_tensor, axis=axis, keepdims=keepdims).astype(
        input_tensor.dtype
    )


@to_ivy_arrays_and_back
def reduce_variance(input_tensor, axis=None, keepdims=False, name="reduce_variance"):
    return ivy.var(input_tensor, axis=axis, keepdims=keepdims)


@to_ivy_arrays_and_back
def scalar_mul(scalar, x, name="scalar_mul"):
    scalar, x = check_tensorflow_casting(scalar, x)
    return ivy.multiply(x, scalar).astype(x.dtype)


@to_ivy_arrays_and_back
def subtract(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.subtract(x, y)


@with_supported_dtypes(
    {
        "2.13.0 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def squared_difference(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    res = ivy.square(ivy.subtract(x, y))
    if isinstance(res, complex):
        res = res.real - res.imag * 1j  # Changing the sign of the imaginary part
        return res
    return res


@with_supported_dtypes(
    {
        "2.13.0 and below": (
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def sin(x, name=None):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def tan(x, name=None):
    return ivy.tan(x)


@to_ivy_arrays_and_back
def unsorted_segment_mean(
    data, segment_ids, num_segments, name="unsorted_segment_mean"
):
    ivy.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    x = ivy.zeros(tuple([num_segments] + (list(data.shape))[1:]))
    count = ivy.zeros((num_segments,))
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = x[segment_ids[i]] + data[i]
        count[segment_ids[i]] += 1
    for j in range(num_segments):
        x[j] = ivy.divide(x[j], count[j])
    return x


@to_ivy_arrays_and_back
def unsorted_segment_sum(
    data, segment_ids, num_segments, name="unsorted_segment_sum"
):
    data = ivy.array(data)
    segment_ids = ivy.array(segment_ids)
    ivy.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    sum_array = ivy.zeros(
        tuple([num_segments] + (list(data.shape))[1:]), 
        dtype=ivy.int32
    )
    for i in range((segment_ids).shape[0]):
        sum_array[segment_ids[i]] = sum_array[segment_ids[i]] + data[i]
    return sum_array


@to_ivy_arrays_and_back
def unsorted_segment_sqrt_n(
    data, segment_ids, num_segments, name="unsorted_segement_sqrt_n"
):
    ivy.utils.assertions.check_equal(
        list(segment_ids.shape), [list(data.shape)[0]], as_array=False
    )
    x = ivy.zeros(tuple([num_segments] + (list(data.shape))[1:]))
    count = ivy.zeros((num_segments,))
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = x[segment_ids[i]] + data[i]
        count[segment_ids[i]] += 1
    for j in range(num_segments):
        x[j] = ivy.divide(x[j], ivy.sqrt(count[j]))
    return x


@to_ivy_arrays_and_back
def zero_fraction(value, name="zero_fraction"):
    zero = ivy.zeros(tuple(list(value.shape)), dtype=ivy.float32)
    x = ivy.array(value, dtype=ivy.float32)
    count_zero = ivy.sum(ivy.equal(x, zero))
    count_nonzero = ivy.sum(ivy.not_equal(x, zero))
    return ivy.divide(count_zero, ivy.add(count_zero, count_nonzero))


@to_ivy_arrays_and_back
def argmin(input, axis=None, output_type="int64", name=None):
    output_type = to_ivy_dtype(output_type)
    if output_type in ["int32", "int64"]:
        return ivy.astype(ivy.argmin(input, axis=axis), output_type)
    else:
        return ivy.astype(ivy.argmin(input, axis=axis), "int64")


@to_ivy_arrays_and_back
def truediv(x, y, name="truediv"):
    x, y = check_tensorflow_casting(x, y)
    x_dtype = ivy.dtype(x)
    if x_dtype in ["int8", "uint8", "int16", "uint16"]:
        return ivy.divide(ivy.astype(x, ivy.float32), ivy.astype(y, ivy.float32))
    elif x_dtype in ["int32", "uint32", "int64", "uint64"]:
        return ivy.divide(ivy.astype(x, ivy.float64), ivy.astype(y, ivy.float64))
    return ivy.divide(x, y)


@to_ivy_arrays_and_back
def equal(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.equal(x, y)


@to_ivy_arrays_and_back
def not_equal(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.not_equal(x, y)


@to_ivy_arrays_and_back
def floor(x, name=None):
    return ivy.floor(x)


@to_ivy_arrays_and_back
def floordiv(x, y, name=None):
    return ivy.floor_divide(x, y)


@to_ivy_arrays_and_back
def ceil(x, name=None):
    return ivy.ceil(x)


@to_ivy_arrays_and_back
def round(x, name=None):
    return ivy.round(x)


@to_ivy_arrays_and_back
def minimum(x, y, name=None):
    return ivy.minimum(x, y)


@to_ivy_arrays_and_back
def sigmoid(x, name=None):
    return ivy.sigmoid(x)


@with_supported_dtypes(
    {"2.13.0 and below": ("float16", "float32", "float64", "complex64", "complex128")},
    "tensorflow",
)
@to_ivy_arrays_and_back
def tanh(x, name=None):
    return ivy.tanh(x)


@to_ivy_arrays_and_back
def rsqrt(x, name=None):
    return ivy.reciprocal(ivy.sqrt(x))


@to_ivy_arrays_and_back
def nextafter(x1, x2, name=None):
    return ivy.nextafter(x1, x2)


@with_unsupported_dtypes(
    {
        "1.2.0": ("float16", "complex64", "complex128"),
        "1.8.0 and below": ("float16",),
        "2.13.0 and below": ("int8", "int16", "uint8", "uint16", "uint32", "uint64"),
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def abs(x, name=None):
    dtype = ivy.dtype(x)
    if dtype in ["complex64", "complex128"]:
        return ivy.sqrt(ivy.square(ivy.real(x)) + ivy.square(ivy.imag(x)))
    return ivy.abs(x)


@to_ivy_arrays_and_back
def log_softmax(logits, axis=None):
    return ivy.log_softmax(logits, axis=axis)


@to_ivy_arrays_and_back
def asin(x, name=None):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def acos(x, name="acos"):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def acosh(x, name="acosh"):
    return ivy.acosh(x)


@to_ivy_arrays_and_back
def square(x, name=None):
    return ivy.square(x)


@to_ivy_arrays_and_back
def is_nan(x, name=None):
    return ivy.isnan(x)


@with_supported_dtypes(
    {
        "2.11.0 and below": ("bfloat16", "half", "float32", "float64"),
    },
    "tensorflow",
)
@to_ivy_arrays_and_back
def is_finite(x, name=None):
    return ivy.isfinite(x)


@to_ivy_arrays_and_back
def atan(x, name=None):
    return ivy.atan(x)


@to_ivy_arrays_and_back
def atan2(y, x, name=None):
    return ivy.atan2(y, x)


@to_ivy_arrays_and_back
def log(x, name=None):
    return ivy.log(x)


@to_ivy_arrays_and_back
def add_n(inputs, name=None):
    inputs = ivy.array(inputs)
    return ivy.sum(inputs, dtype=inputs.dtype, axis=0)


@to_ivy_arrays_and_back
def floormod(x, y, name=None):
    return ivy.remainder(x, y)


@to_ivy_arrays_and_back
def less_equal(x, y, name="LessEqual"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.less_equal(x, y)


@to_ivy_arrays_and_back
def greater(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.greater(x, y)


@to_ivy_arrays_and_back
def less(x, y, name="None"):
    x, y = check_tensorflow_casting(x, y)
    return ivy.less(x, y)


@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def sinh(x, name=None):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def softmax(logits, axis=None, name=None):
    return ivy.softmax(logits, axis=axis)


@to_ivy_arrays_and_back
def softplus(features, name=None):
    return ivy.softplus(features)


@to_ivy_arrays_and_back
def xlogy(x, y, name=None):
    return ivy.xlogy(x, y)


@to_ivy_arrays_and_back
def cosh(x, name=None):
    return ivy.cosh(x)


@to_ivy_arrays_and_back
def angle(input, name=None):
    return ivy.angle(input)


@to_ivy_arrays_and_back
@with_supported_dtypes(
    {
        "2.11.0 and below": ("float32", "float64"),
    },
    "tensorflow",
)
def zeta(x, q, name=None):
    return ivy.zeta(x, q)


@to_ivy_arrays_and_back
def greater_equal(x, y, name=None):
    x, y = check_tensorflow_casting(x, y)
    return ivy.greater_equal(x, y)


@to_ivy_arrays_and_back
def in_top_k(target, pred, k, name=None):
    top_k = ivy.top_k(target, k)
    return ivy.array([val in top_k.values for val in target])


@to_ivy_arrays_and_back
def conj(x, name=None):
    return ivy.conj(x)


@to_ivy_arrays_and_back
def top_k(input, k=1, sorted=True, name=None):
    return ivy.top_k(input, k, sorted=sorted)


@to_ivy_arrays_and_back
def real(input, name=None):
    return ivy.real(input)


@to_ivy_arrays_and_back
def atanh(x, name="atanh"):
    return ivy.atanh(x)


@with_supported_dtypes(
    {"2.13.0 and below": ("int32",)},
    "tensorflow",
)
@to_ivy_arrays_and_back
def bincount(
    arr,
    weights=None,
    minlength=None,
    maxlength=None,
    dtype=ivy.int32,
    name=None,
    axis=None,
    binary_output=False,
):
    return ivy.bincount(arr, weights=weights, minlength=minlength)
