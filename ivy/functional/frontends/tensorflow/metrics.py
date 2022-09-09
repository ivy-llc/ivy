import ivy


def binary_matches(y_true, y_pred, threshold=0.5):
    y_pred = ivy.array(y_pred)
    threshold = ivy.astype(ivy.array(threshold), y_pred.dtype)
    y_pred = ivy.astype(ivy.greater(y_pred, threshold), y_pred.dtype)
    return ivy.astype(
        ivy.equal(y_true, y_pred), ivy.default_float_dtype(as_native=True)
    )


def binary_accuracy(y_true, y_pred, threshold=0.5):
    return ivy.mean(binary_matches(y_true, y_pred, threshold), axis=-1)


def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.sparse_cross_entropy(y_true, y_pred, axis=axis)


def mean_absolute_error(y_true, y_pred):
    return ivy.mean(ivy.abs(y_true - y_pred))


def binary_crossentropy(
    y_true, y_pred, from_logits: bool = False, label_smoothing: float = 0.0
):
    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.mean(ivy.binary_cross_entropy(y_true, y_pred, label_smoothing))


def _sparse_top_k_categorical_matches(y_true, y_pred, k=5):
    # Temporary composition
    def _in_top_k(targets, predictions, topk):
        # Sanity check
        assert targets.ndim == 1, "targets must be 1-dimensional"
        assert predictions.ndim == 2, "predictions must be 2-dimensional"
        targets_batch = targets.shape[0]
        pred_batch = predictions.shape[0]
        assert targets_batch == pred_batch, (
            f"First dimension of predictions {pred_batch} "
            f"must match length of targets {targets_batch}"
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

        labels = predictions.shape[1]
        # float comparison?
        return ivy.array(
            [
                (
                    0 <= res < labels
                    and ivy.min(top_k[ind] - predictions[ind, res]) < 1e-6
                )
                for ind, res in enumerate(targets)
            ]
        )

    reshape = False
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)
    y_true_org_shape = y_true.shape
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


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return _sparse_top_k_categorical_matches(y_true, y_pred, k)


def _sparse_categorical_matches(y_true, y_pred):
    reshape = False
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)
    y_true_org_shape = y_true.shape
    y_true_rank = y_true.ndim
    y_pred_rank = y_pred.ndim

    # y_true shape to (num_samples,)
    if (
        (y_true_rank is not None)
        and (y_pred_rank is not None)
        and (len(y_true.shape) == len(y_pred.shape))
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


def categorical_accuracy(y_true, y_pred):
    return _sparse_categorical_matches(ivy.argmax(y_true, axis=-1), y_pred)


def kl_divergence(y_true, y_pred):
    y_true = ivy.array(y_true)
    y_pred = ivy.array(y_pred)
    y_true = ivy.astype(y_true, y_pred.dtype)
    # clip to range but avoid div-0
    y_true = ivy.clip(y_true, 1e-7, 1)
    y_pred = ivy.clip(y_pred, 1e-7, 1)

    return ivy.sum(y_true * ivy.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    y_pred = ivy.array(y_pred)
    y_true = ivy.array(y_true)
    y_true = ivy.astype(y_true, y_pred.dtype, copy=False)

    return ivy.mean(y_pred - y_true * ivy.log(y_pred + 1e-7), axis=-1)


def mean_squared_error(y_true, y_pred):
    return ivy.mean(ivy.square(ivy.subtract(y_true, y_pred)), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    y_pred = ivy.array(y_pred)
    y_true = ivy.array(y_true)
    y_true = ivy.astype(y_true, y_pred.dtype, copy=False)

    diff = ivy.abs((y_true - y_pred) / ivy.maximum(ivy.abs(y_true), 1e-7))
    return 100.0 * ivy.mean(diff, axis=-1)


def _cond_convert_labels(y_true):
    are_zeros = ivy.equal(y_true, 0.0)
    are_ones = ivy.equal(y_true, 1.0)
    is_binary = ivy.all(ivy.logical_or(are_zeros, are_ones))

    # convert [0, 1] labels to [-1, 1]
    if is_binary:
        return 2.0 * y_true - 1

    return y_true


def hinge(y_true, y_pred):
    y_pred = ivy.array(y_pred)
    y_true = ivy.astype(ivy.array(y_true), y_pred.dtype, copy=False)
    y_true = _cond_convert_labels(y_true)
    return ivy.mean(ivy.maximum(1.0 - y_true * y_pred, 0.0), axis=-1)


def squared_hinge(y_true, y_pred):
    y_pred = ivy.array(y_pred)
    y_true = ivy.astype(ivy.array(y_true), y_pred.dtype)
    y_true = _cond_convert_labels(y_true)
    return ivy.mean(ivy.square(ivy.maximum(1.0 - y_true * y_pred, 0.0)), axis=-1)
