# global
import ivy


def add(x, y, name=None):
    return ivy.add(x, y)


def argmax(input, axis, output_type, name=None):
    return ivy.argmax(input, axis=axis)


def asinh(x, name="asinh"):
    return ivy.asinh(x)


def confusion_matrix(
    labels, predictions, num_classes=None, weights=None, dtype=ivy.int32, name=None
):
    labels = ivy.astype(ivy.squeeze(ivy.array(labels)), ivy.int64, copy=False)
    predictions = ivy.astype(ivy.squeeze(ivy.array(predictions)), ivy.int64, copy=False)
    # Sanity check (potential optimization)
    ivy.assertions.check_greater(
        labels, 0, allow_equal=True, message="labels contains negative values"
    )
    ivy.assertions.check_greater(
        predictions, 0, allow_equal=True, message="predictions contains negative values"
    )

    if num_classes is None:
        num_classes = max(ivy.max(labels), ivy.max(predictions)) + 1
    else:
        num_classes_int64 = ivy.astype(ivy.array(num_classes), ivy.int64, copy=False)
    ivy.assertions.check_less(labels, num_classes_int64, message="labels out of bound")
    ivy.assertions.check_less(
        predictions, num_classes_int64, message="predictions out of bound"
    )

    if weights is not None:
        weights = ivy.array(weights)
        ivy.assertions.check_equal(
            ivy.shape(predictions),
            ivy.shape(weights),
            message="weights shape do not match predictions",
        )
        weights = ivy.astype(weights, dtype, copy=False)

    shape = ivy.stack([num_classes, num_classes])
    indices = ivy.stack([labels, predictions], axis=1)
    values = ivy.ones_like(predictions, dtype=dtype) if weights is None else weights
    return ivy.scatter_nd(indices, values, shape=shape)


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


def cumprod(x, axis=0, exclusive=False, reverse=False, name=None):
    ret = ivy.cumprod(x, axis, exclusive)
    if reverse:
        return ivy.flip(ret, axis)
    return ret


def divide(x, y, name=None):
    return ivy.divide(x, y)


def divide_no_nan(x, y, name="divide_no_nan"):
    return ivy.where(
        y == 0,
        ivy.array(0.0, dtype=ivy.promote_types(x.dtype, y.dtype)),
        x / y,
    )


def maximum(a, b):
    return ivy.maximum(a, b)


def multiply_no_nan(x, y, name="multiply_no_nan"):
    return ivy.where(
        y == 0,
        ivy.array(0.0, dtype=ivy.promote_types(x.dtype, y.dtype)),
        x * y,
    )


def erfcinv(x, name="erfcinv"):
    return 1 / (1 - ivy.erf(x))


def is_non_decreasing(x, name="is_non_decreasing"):
    if ivy.array(x).size < 2:
        return ivy.array(True)
    if ivy.array(x).size == 2:
        return ivy.array(x[0] <= x[1])
    return ivy.all(ivy.less_equal(x, ivy.roll(x, -1)))


def is_strictly_increasing(x, name="is_strictly_increasing"):
    if ivy.array(x).size < 2:
        return ivy.array(True)
    if ivy.array(x).size == 2:
        return ivy.array(x[0] < x[1])
    return ivy.all(ivy.less(x, ivy.roll(x, -1)))


def log_sigmoid(x, name=None):
    return -ivy.softplus(-x)


def logical_and(x, y, name="LogicalAnd"):
    return ivy.logical_and(x, y)


def logical_xor(x, y, name="LogicalXor"):
    return ivy.logical_xor(x, y)


def multiply(x, y, name=None):
    return ivy.multiply(x, y)


def negative(x, name=None):
    return ivy.negative(x)


def polyval(coeffs, x, name=None):
    ivy.assertions.check_isinstance(coeffs, list)
    x = ivy.array(x)
    if len(coeffs) < 1:
        return ivy.zeros_like(x)
    coeffs = [ivy.array(_) for _ in coeffs]
    p = coeffs[0]
    for c in coeffs[1:]:
        p = c + p * x
    return p


def reciprocal_no_nan(input_tensor, name="reciprocal_no_nan"):
    return ivy.where(
        input_tensor == 0,
        ivy.array(0.0, dtype=input_tensor.dtype),
        ivy.ones_like(input_tensor, dtype=input_tensor.dtype) / input_tensor,
    )


def reduce_all(input_tensor, axis=None, keepdims=False, name="reduce_all"):
    return ivy.all(input_tensor, axis=axis, keepdims=keepdims)


def reduce_any(input_tensor, axis=None, keepdims=False, name="reduce_any"):
    return ivy.any(input_tensor, axis=axis, keepdims=keepdims)


def reduce_euclidean_norm(
    input_tensor, axis=None, keepdims=False, name="reduce_euclidean_norm"
):
    return ivy.vector_norm(
        input_tensor, axis=axis, keepdims=keepdims, ord=2
    )  # ord = '2' is the euclidean norm


def reduce_logsumexp(input_tensor, axis=None, keepdims=False, name="reduce_logsumexp"):
    # stable logsumexp trick
    max_input_tensor = ivy.max(input_tensor, axis=axis, keepdims=True)
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


def reduce_max(input_tensor, axis=None, keepdims=False, name="reduce_max"):
    return ivy.max(input_tensor, axis=axis, keepdims=keepdims)


def reduce_min(input_tensor, axis=None, keepdims=False, name="reduce_min"):
    return ivy.min(input_tensor, axis=axis, keepdims=keepdims)


def reduce_prod(input_tensor, axis=None, keepdims=False, name="reduce_prod"):
    return ivy.prod(input_tensor, axis=axis, keepdims=keepdims).astype(
        input_tensor.dtype
    )


def reduce_std(input_tensor, axis=None, keepdims=False, name="reduce_std"):
    return ivy.std(input_tensor, axis=axis, keepdims=keepdims)


def reduce_sum(input_tensor, axis=None, keepdims=False, name="reduce_sum"):
    return ivy.sum(input_tensor, axis=axis, keepdims=keepdims).astype(
        input_tensor.dtype
    )


def reduce_mean(input_tensor, axis=None, keepdims=False, name="reduce_mean"):
    return ivy.mean(input_tensor, axis=axis, keepdims=keepdims)


def reduce_variance(input_tensor, axis=None, keepdims=False, name="reduce_variance"):
    return ivy.var(input_tensor, axis=axis, keepdims=keepdims)


def scalar_mul(scalar, x, name="scalar_mul"):
    return ivy.multiply(x, ivy.array([scalar])).astype(x.dtype)


def subtract(x, y, name=None):
    return ivy.subtract(x, y)


def tan(x, name=None):
    return ivy.tan(x)


def unsorted_segment_mean(
    data, segment_ids, num_segments, name="unsorted_segment_mean"
):
    ivy.assertions.check_equal(list(segment_ids.shape), [list(data.shape)[0]])
    x = ivy.zeros(tuple([num_segments] + (list(data.shape))[1:]))
    count = ivy.zeros((num_segments,))
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = x[segment_ids[i]] + data[i]
        count[segment_ids[i]] += 1
    for j in range(num_segments):
        x[j] = ivy.divide(x[j], count[j])
    return x


def unsorted_segment_sqrt_n(
    data, segment_ids, num_segments, name="unsorted_segement_sqrt_n"
):
    ivy.assertions.check_equal(list(segment_ids.shape), [list(data.shape)[0]])
    x = ivy.zeros(tuple([num_segments] + (list(data.shape))[1:]))
    count = ivy.zeros((num_segments,))
    for i in range((segment_ids).shape[0]):
        x[segment_ids[i]] = x[segment_ids[i]] + data[i]
        count[segment_ids[i]] += 1
    for j in range(num_segments):
        x[j] = ivy.divide(x[j], ivy.sqrt(count[j]))
    return x


def zero_fraction(value, name="zero_fraction"):
    zero = ivy.zeros(tuple(list(value.shape)), dtype=ivy.float32)
    x = ivy.array(value, dtype=ivy.float32)
    count_zero = ivy.sum(ivy.equal(x, zero))
    count_nonzero = ivy.sum(ivy.not_equal(x, zero))
    return ivy.divide(count_zero, ivy.add(count_zero, count_nonzero))


# TODO: Ibeta for Future Release
