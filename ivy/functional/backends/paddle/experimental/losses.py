from typing import Optional
import paddle
import paddle.nn.functional as F
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version

# Assuming ivy and backend_version are imported and defined properly


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
def hinge_embedding_loss(
    input: paddle.Tensor,
    target: paddle.Tensor,
    /,
    *,
    margin: Optional[float] = 1.0,
    reduction: Optional[str] = "mean",
    name: Optional[str] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return F.hinge_embedding_loss(
        input,
        target,
        margin=margin,
        reduction=reduction,
        name=name,
        out=out,
    )
