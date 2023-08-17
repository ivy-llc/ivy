# global
from typing import Optional
import paddle
import paddle.nn.functional as F

# local

unsupported_dtypes = [
    paddle.int8,
    paddle.int16,
    paddle.int32,
    paddle.int64,
    paddle.uint8,
    paddle.float16,
    paddle.complex64,
    paddle.complex128,
    paddle.bool,
]


def l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    reduction: str = "mean",
    name: Optional[str] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if input.dtype in unsupported_dtypes:
        if paddle.is_complex(input):
            real_loss = F.l1_loss(input.real(), target.real(), reduction=reduction)
            imag_loss = F.l1_loss(input.imag(), target.imag(), reduction=reduction)
            return paddle.complex(real_loss, imag_loss)
        return F.l1_loss(
            input.cast("float32"),
            target.cast("float32"),
            reduction=reduction,
        ).cast(input.dtype)

    return F.l1_loss(input, target, reduction=reduction, name=name)
