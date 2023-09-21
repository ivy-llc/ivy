import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes


@to_ivy_arrays_and_back
def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    # TODO: add support for the remaining arguments
    ivy.utils.assertions.check_equal(
        len(weight.shape), 2, message="weight must be 2-d", as_array=False
    )
    input = ivy.astype(input, "int64")
    if max_norm is None:
        ret = ivy.embedding(weight, input)
    else:
        if norm_type == 2.0:
            ret = ivy.embedding(weight, input, max_norm=max_norm)
        else:
            ret = ivy.embedding(weight, input, max_norm=None)
            # perform the re-norm using ivy functions
            norms = ivy.vector_norm(ret, ord=norm_type, axis=-1, keepdims=True)
            norms = ivy.repeat(norms, ret.shape[-1], axis=-1)
            ret = ivy.where(norms > max_norm, ret * max_norm / norms, ret)
            ret = ivy.where(norms < -max_norm, ret * -max_norm / norms, ret)
    return ret


@with_supported_dtypes({"2.0.1 and below": ("int64",)}, "torch")
@to_ivy_arrays_and_back
def one_hot(tensor, num_classes=-1):
    return ivy.astype(ivy.one_hot(tensor, num_classes), tensor.dtype)
