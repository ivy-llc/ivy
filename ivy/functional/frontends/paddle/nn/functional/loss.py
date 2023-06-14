# local
import ivy
from ivy.func_wrapper import with_supported_dtypes
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


@handle_exceptions
@to_ivy_arrays_and_back
@with_supported_dtypes({"2.4.2 and below": ("float32", "float64")}, "paddle")
def cosine_embedding_loss(
    input1, input2, label, margin=0.0, reduction="mean", name=None
):
    if len(label.shape) != 1:
        raise ValueError("1D target tensor expected, multi-target not supported")

    if input1.shape != input2.shape:
        raise ValueError(
            "the shape of input tensor 1 should be equal to input tensor 2, but found"
            " inputs with different sizes"
        )

    if len(input1.shape) > 2:
        raise ValueError(
            "1D target tensor expects 1D or 2D input tensors, but found inputs with"
            " different sizes"
        )

    prod_sum = (input1 * input2).sum(axis=-1)
    mag_square1 = ivy.square(input1).sum(axis=-1) + 10e-12
    mag_square2 = ivy.square(input2).sum(axis=-1) + 10e-12
    denom = ivy.sqrt(mag_square1 * mag_square2)
    cos = prod_sum / denom
    zeros = ivy.zeros_like(cos)
    pos = 1 - cos
    neg = ivy.clip(cos - margin, 0, 0)
    out_pos = ivy.where(label == 1, pos, zeros)
    out_neg = ivy.where(label == -1, neg, zeros)
    out = out_pos + out_neg

    if reduction == "none":
        out = ivy.expand_dims(out, axis=-1) if out.ndim == 0 else out
        return out
    if reduction == "mean":
        out = ivy.mean(out)
        out = ivy.expand_dims(out, axis=-1) if out.ndim == 0 else out
        return out
    elif reduction == "sum":
        out = ivy.sum(out)
        out = ivy.expand_dims(out, axis=-1) if out.ndim == 0 else out
        return out
