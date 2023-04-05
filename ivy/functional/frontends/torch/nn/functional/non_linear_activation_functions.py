# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


def _rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    if training:
        # alpha = ivy.random_uniform(low=lower, high=upper)
        # ToDo implement alpha correctly after fixing ivy.random_uniform
        pass
    else:
        alpha = (lower + upper) / 2
    ret = ivy.subtract(
        ivy.relu(input), ivy.multiply(alpha, ivy.relu(ivy.negative(input)))
    )
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def sigmoid(input):
    return ivy.sigmoid(input)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def leaky_relu(input, negative_slope=0.01, inplace=False):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def gelu(
    input,
):
    return ivy.gelu(input, approximate=False)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def tanh(input):
    return ivy.tanh(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def logsigmoid(input):
    return ivy.logsigmoid(input)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def threshold(input, threshold, value, inplace=False):
    out = input if inplace else None
    return ivy.threshold(input, threshold, value, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def threshold_(input, threshold, value):
    return ivy.threshold(input, threshold, value, out=input)


def relu6(input, inplace=False):
    ret = ivy.relu6(input)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def elu(input, alpha=1.0, inplace=False):
    out = input if inplace else None
    return ivy.elu(input, alpha=alpha, out=out)


def elu_(input, alpha=1.0):
    return ivy.elu(input, alpha=alpha, out=input)


def prelu(input, weight):
    return ivy.parametric_relu(input, weight)


def celu(input, alpha=1.0, inplace=False):
    out = input if inplace else None
    return ivy.celu(input, alpha=alpha, out=out)


def mish(input, inplace=False):
    ret = ivy.mish(input)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


@to_ivy_arrays_and_back
def relu(input, inplace=False):
    ret = ivy.relu(input)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def relu_(input):
    ret = ivy.relu(input)
    ivy.inplace_update(input, ret)
    return input


def selu(input, inplace=False):
    out = input if inplace else None
    return ivy.selu(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    return _rrelu(input, lower, upper, training, inplace)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return _rrelu(input, lower, upper, training, inplace=True)


@to_ivy_arrays_and_back
def hardshrink(input, lambd=0.5):
    return ivy.hardshrink(input, lambd=lambd)


@to_ivy_arrays_and_back
def softshrink(input, lambd=0.5):
    return ivy.softshrink(input, lambd=lambd)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def silu(input, inplace=False):
    out = input if inplace else None
    return ivy.silu(input, out=out)


@to_ivy_arrays_and_back
def glu(input, dim=-1):
    return ivy.glu(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def tanhshrink(input):
    return ivy.subtract(input, ivy.tanh(input))


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def leaky_relu_(input, negative_slope=0.01):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    ivy.inplace_update(input, ret)
    return input


@to_ivy_arrays_and_back
def hardswish(input, inplace=False):
    out = input if inplace else None
    return ivy.hard_silu(input, out=out)


@to_ivy_arrays_and_back
def hardsigmoid(input, inplace=False):
    out = input if inplace else None
    return ivy.hard_sigmoid(input, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    out = input if inplace else None
    return ivy.hard_tanh(input, min_value=min_val, max_value=max_val, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def hardtanh_(input, min_val=-1.0, max_val=1.0):
    return ivy.hard_tanh(input, min_value=min_val, max_value=max_val, out=input)


@to_ivy_arrays_and_back
def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    return ivy.lp_normalize(input, p=p, axis=dim)


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, "torch")
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    shape = ivy.shape(input)
    if isinstance(normalized_shape, int) and normalized_shape == shape[-1]:
        axis = [-1]
    else:
        assert normalized_shape == shape[-len(normalized_shape) :]
        axis = list(range(len(shape) - len(normalized_shape), len(shape)))
    return ivy.layer_norm(input, axis, scale=weight, b=bias, epsilon=eps)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "float16",
            "bfloat16",
        )
    },
    "torch",
)
def softplus(input, beta=1, threshold=20):
    return ivy.softplus(input, beta=beta, threshold=threshold)


@to_ivy_arrays_and_back
def softsign(input):
    return ivy.softsign(input)


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def batch_norm(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-5,
):
    # momentum is not practically used in the functional api
    return ivy.batch_norm(
        input,
        running_mean,
        running_var,
        offset=bias,
        scale=weight,
        training=training,
        eps=eps,
    )


@with_unsupported_dtypes(
    {
        "1.11.0 and below": (
            "bfloat16",
            "float16",
        )
    },
    "torch",
)
@to_ivy_arrays_and_back
def group_norm(
    input,
    num_groups,
    weight=None,
    bias=None,
    eps=1e-5,
):
    return ivy.group_norm(
        input,
        num_groups,
        weight=weight,
        bias=bias,
        eps=eps,
    )
