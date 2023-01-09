import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def embedding(
        input,
        weight,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False
):
    ivy.assertions.check_equal(
        len(weight.shape), 2, message="weight must be 2-d"
    )
    ret = ivy.empty(
        input.shape+(weight.shape[1],),
        dtype=weight.dtype,
        device=weight.device
    )
    for i, x in ivy.ndenumerate(input):
        ret[i] = weight[x, :]
    return ret
