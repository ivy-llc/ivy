import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back
import ivy.functional.frontends.torch as torch_frontend
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
    ivy.utils.assertions.check_equal(len(weight.shape), 2, message="weight must be 2-d")
    ret = ivy.empty(
        input.shape + (weight.shape[1],), dtype=weight.dtype, device=weight.device
    )
    for i, x in ivy.ndenumerate(input):
        if i == padding_idx:
            ret[i] = ivy.stop_gradient(
                ivy.zeros(1, weight.shape[1]), preserve_type=True
            )
        if ivy.exists(max_norm):
            ret[i] = ivy.clip_vector_norm(weight[x, :], max_norm, p=norm_type)
        else:
            ret[i] = weight[x, :]
    ret_dtype = torch_frontend.promote_types_torch(input.dtype, weight.dtype)
    return ivy.astype(ret, ret_dtype)


@with_supported_dtypes({"2.0.1 and below": ("int64",)}, "torch")
@to_ivy_arrays_and_back
def one_hot(tensor, num_classes=-1):
    return ivy.astype(ivy.one_hot(tensor, num_classes), tensor.dtype)
