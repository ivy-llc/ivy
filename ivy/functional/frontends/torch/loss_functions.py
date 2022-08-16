# global
import ivy


def cross_entropy(
    input,
    target,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
):
    return ivy.cross_entropy(input, target)
 

cross_entropy.unsupported_dtypes = ('uint16', 'float16', 'uint64', 'uint32')
