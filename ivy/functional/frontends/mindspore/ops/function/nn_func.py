"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@with_supported_dtypes(
    {
        "2.0.0 and below": (
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        )
    },
    "mindspore",
)
@to_ivy_arrays_and_back
def dropout2d(input, p=0.5, training=True):
    return ivy.dropout2d(input, p, training=training, data_format="NCHW")


@with_supported_dtypes({"2.0.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def selu(input_x):
    return ivy.selu(input_x)


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def softsign(x):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def kl_div(logits, labels, reduction="mean"):
    """
    Computes the Kullback-Leibler (KL) Divergence between the logits and the labels.

    Parameters:
        logits (numpy array): The input logits array.
        labels (numpy array): The label array which has the same shape as logits.
        reduction (str): Specifies the reduction to be applied to the output.
                         Its value must be one of 'none', 'mean', 'batchmean',
                         or 'sum'. Default: 'mean'.

    Returns:
        float or numpy array: If reduction is 'none', then output is
        a numpy array and has the same shape as logits.
                              Otherwise, it is a scalar (float).
    """
    assert ivy.shape(logits) == ivy.shape(
        labels
    ), "logits and labels must have the same shape."
    L = labels * (ivy.log(labels) - logits)
    if reduction == "none":
        return L
    elif reduction == "mean":
        return ivy.mean(L)
    elif reduction == "batchmean":
        return ivy.mean(L, axis=0)
    elif reduction == "sum":
        return ivy.sum(L)
    else:
        raise ValueError(
            "Invalid reduction mode. Supported values are 'none', 'mean', 'batchmean',"
            " or 'sum'."
        )
