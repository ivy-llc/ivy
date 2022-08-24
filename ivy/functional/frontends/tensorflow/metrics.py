import ivy


def binary_crossentropy(y_true, y_pred,
                        from_logits: bool =False,
                        label_smoothing: float =0.) -> ivy.Array:
    """Computes the binary crossentropy loss.
    Parameters
    ----------
        y_true:
            Ground truth values.
        y_pred:
            The predicted values.
        from_logits: bool
            Whether `y_pred` is expected to be a logits array/tensor. By default,
            we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float
            in [0, 1]. If > `0` then smooth the labels by
            squeezing them towards 0.5 That is, using `1. - 0.5 * label_smoothing`
            for the target class and `0.5 * label_smoothing` for the non-target class.
    Returns
    -------
        Binary crossentropy loss value.
    """

    if from_logits:
        y_pred = ivy.softmax(y_pred)
    return ivy.mean(ivy.binary_cross_entropy(y_true,
                                             y_pred, label_smoothing))
