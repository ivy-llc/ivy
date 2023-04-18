import math
from typing import Optional, Tuple, Sequence, Union
import jax
import jax.nn
import jax.numpy as jnp
from jax import jit

import ivy
import numpy as np

from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size


@jit
def ctc_loss(
    log_probs: jnp.ndarray,
    targets: jnp.ndarray,
    input_lengths: jnp.ndarray,
    target_lengths: Optional[jnp.ndarray],
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: Optional[bool] = True,
    out: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:

    targets = targets.astype(jnp.int32)
    input_lengths = input_lengths.astype(jnp.int64)
    target_lengths = target_lengths.astype(jnp.int64)
    blank = np.int32(blank)


    batch_size = log_probs.shape[0]
    max_input_length = log_probs.shape[1]
    num_classes = log_probs.shape[2]

    alpha = jnp.zeros((batch_size, max_input_length, num_classes))
    alpha = alpha.at[:, 0, 0].set(1.0)


    for t in range(1, max_input_length):
        for s in range(num_classes):
            for b in range(batch_size):
                if t < input_lengths[b]:
                    p = log_probs[b, t, s]
                    if s == 0:
                        alpha = alpha.at[b, t, s].set(alpha[b, t - 1, s] * p)
                    else:
                        alpha = alpha.at[b, t, s].set(alpha[b, t - 1, s] * p + alpha[b, t - 1, s - 1] * p)

                else:
                    alpha = alpha.at[b, t, s].set(alpha[b, t - 1, s] + alpha[b, t - 1, s - 1])



    beta = jnp.zeros((batch_size, max_input_length, num_classes))
    beta = beta.at[:, -1, 0].set(1.0)

    for t in range(max_input_length - 2, -1, -1):
        for s in range(num_classes):
            for b in range(batch_size):
                if t < input_lengths[b]:
                    p = log_probs[b, t+1, s]
                    if s == 0:
                        beta = beta.at[b, t, s].set(beta[b, t + 1, s] * p)
                    else:
                        beta = beta.at[b, t, s].set(beta[b, t + 1, s] * p + beta[b, t + 1, s - 1] * p)

                else:
                    beta = beta.at[b, t, s].set(beta[b, t + 1, s] + beta[b, t + 1, s - 1])

    label_prob = jnp.zeros((batch_size, num_classes))
    for s in range(num_classes):
        for b in range(batch_size):
            label_prob = label_prob.at[b, s].set(alpha[b, input_lengths[b] - 1, s] * beta[b, input_lengths[b] - 1, s])

    ctc_loss = jnp.zeros((batch_size,))
    for b in range(batch_size):
        p = 1.0
        for s in range(targets.shape[1]):
            if targets[b, s] == blank:
                continue
            elif s > 0 and targets[b, s] == targets[b, s - 1]:
                p *= label_prob[b, targets[b, s]] / 2.0

            else:
                p *= label_prob[b, targets[b, s]]

        ctc_loss = ctc_loss.at[b].set(-jnp.log(p))

    if reduction == "mean":
        loss = jnp.mean(ctc_loss)

    elif reduction == "sum":
        loss = jnp.sum(ctc_loss)

    elif reduction == "none":
        loss = ctc_loss

    else:
        raise ValueError("reduction must be one of 'mean', 'sum', 'none'")

    return loss

