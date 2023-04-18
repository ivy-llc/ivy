import math
from typing import Optional, Tuple, Sequence, Union, Any
import numpy as np

import ivy

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size
def ctc_loss(
    log_probs: np.ndarray,
    targets: np.ndarray,
    input_lengths: np.ndarray,
    target_lengths: Optional[np.ndarray],
    blank: Optional[int] = 0,
    reduction: Optional[str] = "mean",
    zero_infinity: Optional[bool] = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:

    targets = targets.astype(np.int32)
    input_lengths = input_lengths.astype(np.int64)
    target_lengths = target_lengths.astype(np.int64)
    blank = np.int32(blank)
    
    batch_size = log_probs.shape[0]
    max_input_length = log_probs.shape[1]
    num_classes = log_probs.shape[2]

    # Compute the forward probabilities using dynamic programming
    alpha = np.zeros((batch_size, max_input_length, num_classes))
    alpha[:, 0, 0] = 1.0
    for t in range(1, max_input_length):
        for s in range(num_classes):
            for b in range(batch_size):
                if t < input_lengths[b]:
                    p = log_probs[b, t, s]
                    if s == 0:
                        alpha[b, t, s] = alpha[b, t - 1, s] * p
                    else:
                        alpha[b, t, s] = alpha[b, t - 1, s] * p + alpha[b, t - 1, s - 1] * p

                else:
                    alpha[b, t, s] = alpha[b, t - 1, s] + alpha[b, t - 1, s - 1]

    # Compute the backward probabilities using dynamic programming
    beta = np.zeros((batch_size, max_input_length, num_classes))
    beta[:, -1, 0] = 1.0
    for t in range(max_input_length - 2, -1, -1):
        for s in range(num_classes):
            for b in range(batch_size):
                if t < input_lengths[b]:
                    p = log_probs[b, t+1, s]
                    if s == 0:
                        beta[b, t, s] = beta[b, t + 1, s] * p
                    else:
                        beta[b, t, s] = beta[b, t + 1, s] * p + beta[b, t + 1, s - 1] * p

                else:
                    beta[b, t, s] = beta[b, t + 1, s] + beta[b, t + 1, s - 1]

    # Compute the label probabilities using the forward and backward probabilities
    label_prob = np.zeros((batch_size, num_classes))
    for s in range(num_classes):
        for b in range(batch_size):
            label_prob[b, s] = alpha[b, input_lengths[b] - 1, s] * beta[b, input_lengths[b] - 1, s]

    # Compute the CTC Loss for each sample in the batch
    ctc_loss = np.zeros((batch_size, 1))
    for b in range(batch_size):
        p = 1.0
        for s in range(targets.shape[1]):
            if targets[b, s] == 0:
                continue
            elif s > 0 and targets[b, s] == targets[b, s - 1]:
                p *= label_prob[b, targets[b, s]] / 2.0

            else:
                p *= label_prob[b, targets[b, s]]

        ctc_loss[b] = -np.log(p)

    return ctc_loss

