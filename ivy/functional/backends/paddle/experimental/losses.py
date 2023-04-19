# global
import math

import paddle
import paddle.nn.functional as F
from ivy.utils.exceptions import IvyNotImplementedException
from typing import Optional, Tuple

import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes

from .. import backend_version

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size
import numpy as np
@with_unsupported_dtypes({"1.13.0 and below": ("float16",)}, backend_version)

def ctc_loss(
    log_probs: paddle.Tensor,
    targets: paddle.Tensor,
    input_lengths: paddle.Tensor,
    target_lengths: paddle.Tensor,
    blank: Optional[int] = 0,
    zero_infinity: Optional[bool] = False,
    reduction: Optional[str] = "mean",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:

    return F.ctc_loss(
        log_probs=log_probs,
        labels=paddle.cast(targets, paddle.int32),
        input_lengths=paddle.cast(input_lengths, paddle.int64),
        label_lengths=paddle.cast(target_lengths, paddle.int64),
        blank=np.int32(blank),
        )




