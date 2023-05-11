import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back


def _binary_matches(y_true, y_pred, threshold=0.5):
    threshold = ivy.astype(ivy.array(threshold), y_pred.dtype)
    y_pred = ivy.astype(ivy.greater(y_pred, threshold), y_pred.dtype)
    return ivy.astype(
        ivy.equal(y_true, y_pred), ivy.default_float_dtype(as_native=True)
    )


def _cond_convert_labels(y_true):
    are_zeros = ivy.equal(y_true, 0.0)
    are_ones = ivy.equal(y_true, 1.0)
    is_binary = ivy.all(ivy.logical_or(are_zeros, are_ones))
    # convert [0, 1] labels to [-1, 1]
    if is_binary:
        return 2.0 * y_true - 1
    return y_true


@to_ivy_arrays_and_back
def _sparse_categorical_matches(y_true, y_pred):
    reshape = False
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)
    y_true_org_shape = ivy.shape(y_true)
    y_true_rank = y_true.ndim
    y_pred_rank = y_pred.ndim
    # y_true shape to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(ivy.shape(y_true)) == len(ivy.shape(y_pred)))
    ):
        y_true = ivy.squeeze(y_true, axis=-1)
        reshape = True
    y_pred = ivy.argmax(y_pred, axis=-1)
    # cast prediction type to be the same as ground truth
    y_pred = ivy.astype(y_pred, y_true.dtype, copy=False)
    matches = ivy.astype(ivy.equal(y_true, y_pred), ivy.float32)
    if reshape:
        matches = ivy.reshape(matches, shape=y_true_org_shape)
    return matches


@to_ivy_arrays_and_back
def _sparse_top_k_categorical_matches(y_true, y_pred, k=5):
    # Temporary composition
    def _in_top_k(targets, predictions, topk):
        # Sanity check
        ivy.utils.assertions.check_equal(
            targets.ndim, 1, message="targets must be 1-dimensional"
        )
        ivy.utils.assertions.check_equal(
            predictions.ndim, 2, message="predictions must be 2-dimensional"
        )
        targets_batch = ivy.shape(targets)[0]
        pred_batch = ivy.shape(predictions)[0]
        ivy.utils.assertions.check_equal(
            targets_batch,
            pred_batch,
            message=(
                "first dim of predictions: {} must match targets length: {}".format(
                    pred_batch, targets_batch
                )
            ),
        )

        # return array of top k values from the input
        def _top_k(input, topk):
            x = ivy.array(input)
            sort = ivy.argsort(x, descending=True)
            topk = min(x.shape[-1], topk)

            # Safety check for equal values
            result = []
            for ind, li in enumerate(sort):
                temp = [x[ind, _] for _ in li[:topk]]
                result.append(temp)

            return ivy.array(result)

        top_k = _top_k(predictions, topk)

        labels = ivy.shape(predictions)[1]
        # float comparison?
        return ivy.array(
            [
                (
                    0 <= res < labels
                    and ivy.min(top_k[ind] - predictions[ind, res]) <= 1e-9
                )
                for ind, res in enumerate(targets)
            ]
        )

    reshape = False
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)
    y_true_org_shape = ivy.shape(y_true)
    y_true_rank = y_true.ndim
    y_pred_rank = y_pred.ndim

    # y_pred shape to (batch_size, num_samples), y_true shape to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None):
        if y_pred_rank > 2:
            y_pred = ivy.reshape(y_pred, shape=[-1, y_pred.shape[-1]])
        if y_true_rank > 1:
            reshape = True
            y_true = ivy.reshape(y_true, shape=[-1])

    matches = ivy.astype(
        _in_top_k(targets=ivy.astype(y_true, ivy.int32), predictions=y_pred, topk=k),
        ivy.float32,
    )

    # return to original shape
    if reshape:
        return ivy.reshape(matches, shape=y_true_org_shape)
    return matches


@to_ivy_arrays_and_back
def binary_accuracy(y_true, y_pred, threshold=0.5):
    return ivy.mean(_binary_matches(y_true, y_pred, threshold), axis=-1)


@to_ivy_arrays_and_back
def binary_crossentropy(
    y_true, y_pred, from_logits: bool = False, label_smoothing: float = 0.0
):
    y_pred = ivy.asarray(y_pred)
    y_true = ivy.asarray(y_true, dtype=y_pred.dtype)
    label_smoothing = ivy.asarray(label_smoothing, dtype=y_pred.dtype)
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        zeros = ivy.zeros_like(y_pred, dtype=y_pred.dtype)
        cond = y_pred >= zeros
        relu_logits = ivy.where(cond, y_pred, zeros)
        neg_abs_logits = ivy.where(cond, -y_pred, y_pred)
        bce = ivy.add(relu_logits - y_pred * y_true, ivy.log1p(ivy.exp(neg_abs_logits)))
    else:
        epsilon_ = 1e-7
        y_pred = ivy.clip(y_pred, epsilon_, 1.0 - epsilon_)
        bce = y_true * ivy.log(y_pred + epsilon_)
        bce += (1 - y_true) * ivy.log(1 - y_pred + epsilon_)
        bce = -bce
    return ivy.mean(bce, axis=-1).astype(y_pred.dtype)


@to_ivy_arrays_and_back
def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.mean(ivy.categorical_cross_entropy(y_true, y_pred, label_smoothing))


@to_ivy_arrays_and_back
def binary_focal_crossentropy(
    y_true, y_pred, gamma=2.0, from_logits=False, label_smoothing=0.0, axis=-1
):
    y_pred = ivy.asarray(y_pred)
    y_true = ivy.asarray(y_true, dtype=y_pred.dtype)
    label_smoothing = ivy.asarray(label_smoothing, dtype=y_pred.dtype)
    gamma = ivy.asarray(gamma, dtype=y_pred.dtype)

    if label_smoothing > 0.0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if from_logits:
        sigmoidal = ivy.sigmoid(y_pred)
    else:
        sigmoidal = y_pred

    p_t = (y_true * sigmoidal) + ((1 - y_true) * (1 - sigmoidal))
    focal_factor = ivy.pow(1.0 - p_t, gamma)

    if from_logits:
        zeros = ivy.zeros_like(y_pred, dtype=y_pred.dtype)
        cond = y_pred >= zeros
        relu_logits = ivy.where(cond, y_pred, zeros)
        neg_abs_logits = ivy.where(cond, -y_pred, y_pred)
        bce = ivy.add(relu_logits - y_pred * y_true, ivy.log1p(ivy.exp(neg_abs_logits)))
    else:
        epsilon_ = 1e-7
        y_pred = ivy.clip(y_pred, epsilon_, 1.0 - epsilon_)
        bce = y_true * ivy.log(y_pred + epsilon_)
        bce += (1 - y_true) * ivy.log(1 - y_pred + epsilon_)
        bce = -bce
    bfce = focal_factor * bce
    return ivy.mean(bfce, axis=ivy.to_scalar(axis))


@to_ivy_arrays_and_back
def categorical_accuracy(y_true, y_pred):
    return _sparse_categorical_matches(ivy.argmax(y_true, axis=-1), y_pred)


@to_ivy_arrays_and_back
def hinge(y_true, y_pred):
    y_true = ivy.astype(ivy.array(y_true), y_pred.dtype, copy=False)
    y_true = _cond_convert_labels(y_true)
    return ivy.mean(ivy.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


@to_ivy_arrays_and_back
def kl_divergence(y_true, y_pred):
    # clip to range but avoid div-0
    y_true = ivy.clip(y_true, 1e-7, 1)
    y_pred = ivy.clip(y_pred, 1e-7, 1)
    return ivy.sum(y_true * ivy.log(y_true / y_pred), axis=-1).astype(y_true.dtype)


kld = kl_divergence


kullback_leibler_divergence = kl_divergence


@to_ivy_arrays_and_back
def log_cosh(y_true, y_pred):
    y_true = ivy.astype(y_true, y_pred.dtype)
    diff = y_pred - y_true
    log_val = ivy.astype(ivy.log(2.0), diff.dtype)
    return ivy.mean(diff + ivy.softplus(-2.0 * diff) - log_val, axis=-1)


logcosh = log_cosh


@to_ivy_arrays_and_back
def mean_absolute_error(y_true, y_pred):
    return ivy.mean(ivy.abs(y_true - y_pred), axis=-1)


mae = mean_absolute_error


@to_ivy_arrays_and_back
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = ivy.astype(y_true, y_pred.dtype, copy=False)

    diff = ivy.abs((y_true - y_pred) / ivy.maximum(ivy.abs(y_true), 1e-7))
    return 100.0 * ivy.mean(diff, axis=-1)


mape = mean_absolute_percentage_error


@to_ivy_arrays_and_back
def mean_squared_error(y_true, y_pred):
    return ivy.mean(ivy.square(ivy.subtract(y_true, y_pred)), axis=-1)


mse = mean_squared_error


@to_ivy_arrays_and_back
def mean_squared_logarithmic_error(y_true, y_pred):
    y_true = ivy.astype(y_true, y_pred.dtype)
    first_log = ivy.log(ivy.maximum(y_pred, 1e-7) + 1.0)
    second_log = ivy.log(ivy.maximum(y_true, 1e-7) + 1.0)
    return ivy.mean(ivy.square(ivy.subtract(first_log, second_log)), axis=-1)


msle = mean_squared_logarithmic_error


@to_ivy_arrays_and_back
def poisson(y_true, y_pred):
    y_true = ivy.astype(y_true, y_pred.dtype, copy=False)
    return ivy.mean(y_pred - y_true * ivy.log(y_pred + 1e-7), axis=-1)


@to_ivy_arrays_and_back
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.sparse_cross_entropy(y_true, y_pred, axis=axis)


@to_ivy_arrays_and_back
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return _sparse_top_k_categorical_matches(y_true, y_pred, k)


@to_ivy_arrays_and_back
def squared_hinge(y_true, y_pred):
    y_true = ivy.astype(ivy.array(y_true), y_pred.dtype)
    y_true = _cond_convert_labels(y_true)
    return ivy.mean(ivy.square(ivy.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)


@to_ivy_arrays_and_back
def cosine_similarity(y_true, y_pred):
    y_pred = ivy.asarray(y_pred)
    y_true = ivy.asarray(y_true)

    if len(y_pred.shape) == len(y_pred.shape) and len(y_true.shape) == 2:
        numerator = ivy.sum(y_true * y_pred, axis=1)
    else:
        numerator = ivy.vecdot(y_true, y_pred)
    denominator = ivy.matrix_norm(y_true) * ivy.matrix_norm(y_pred)
    return numerator / denominator
