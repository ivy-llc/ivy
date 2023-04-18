# global
import math

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Sequence, Union

import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size

import numpy as np

@with_unsupported_dtypes({"1.13.0 and below": ("float16",)}, backend_version)
def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    log_probs = log_probs.transpose(0, 1)

    #Compute the CTC loss using the Pytorch implementation
    ctc_loss = F.ctc_loss(
        log_probs=log_probs,
        targets=targets.to(torch.int32),
        input_lengths=input_lengths.to(torch.int64),
        target_lengths=target_lengths.to(torch.int64),
        blank=np.int32(blank),
        reduction=reduction,
        zero_infinity=zero_infinity,

    )

    return ctc_loss


