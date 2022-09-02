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


sparse_categorical_crossentropy.unsupported_dtypes = {
    "numpy": ("float16", "bfloat16", "float32", "float64"),
}


def mean_absolute_error(y_true, y_pred):
    return ivy.mean(ivy.abs(y_true - y_pred))


mean_absolute_error.unsupported_dtypes = {
    "numpy": ("int8", "float64"),
    "torch": ("int8", "float64"),
}


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
        return ivy.reshape(matches, y_true_org_shape)

    return matches


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    return _sparse_top_k_categorical_matches(y_true, y_pred, k)
