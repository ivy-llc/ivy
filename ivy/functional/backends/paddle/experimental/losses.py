# global
from typing import Optional
import paddle
import paddle.nn.functional as F

# local
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "float16",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def l1_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    reduction: str = "mean",
    name: Optional[str] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.l1_loss(input, target, reduction=reduction, name=name)
