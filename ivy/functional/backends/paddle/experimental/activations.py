from typing import Optional, Union

# global
import paddle
import paddle.nn.functional as F
import ivy

# local
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logit(x: paddle.Tensor, /, *, eps: Optional[float] = None, out=None):
    if x.dtype in [paddle.float32, paddle.float64]:
        return paddle.logit(x, eps)
    with ivy.ArrayMode(False):
        if eps is None:
            nan = paddle.to_tensor(float("nan"), dtype=x.dtype)
            x = ivy.where(ivy.logical_or(ivy.greater(x, 1), ivy.less(x, 0)), nan, x)
        else:
            x = ivy.minimum(ivy.maximum(x, eps), 1 - eps)
        return ivy.log(ivy.divide(x, ivy.subtract(1, x))).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def thresholded_relu(
    x: paddle.Tensor,
    /,
    *,
    threshold: Optional[Union[int, float]] = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.thresholded_relu(x, threshold=threshold)
    with ivy.ArrayMode(False):
        x, threshold = ivy.promote_types_of_inputs(x, threshold)
        return ivy.where(ivy.greater_equal(x, threshold), x, 0)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def relu6(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.relu6(x)
    if paddle.is_complex(x):
        return paddle.complex(F.relu6(x.real()), F.relu6(x.imag()))
    return F.relu6(x.cast("float32")).cast(x.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def logsigmoid(input: paddle.Tensor) -> paddle.Tensor:
    if input.dtype in [paddle.float32, paddle.float64]:
        return F.log_sigmoid(input)
    if paddle.is_complex(input):
        with ivy.ArrayMode(False):
            return ivy.log(ivy.divide(1, (ivy.add(1, ivy.exp(input)))))
    return F.log_sigmoid(input.cast("float32")).cast(input.dtype)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def selu(x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    if x.dtype in [paddle.float32, paddle.float64]:
        return F.selu(x)
    if paddle.is_complex(x):
        with ivy.ArrayMode(False):
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            ret = ivy.multiply(
                scale,
                ivy.where(ivy.greater(x, 0), x, ivy.multiply(alpha, ivy.expm1(x))),
            )
            return ret
    return F.selu(x.cast("float32")).cast(x.dtype)
