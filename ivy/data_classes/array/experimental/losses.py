# global
import abc
from typing import Optional, Union

# local
import ivy
from ivy import handle_view
import numpy as np

class _ArrayWithLossesExperimental(abc.ABC):
    @handle_view
    def ctc_loss(
        pred: Union[ivy.Array, ivy.NativeArray],
        true : Union[ivy.Array, ivy.NativeArray],
        pred_lengths: Union[ivy.Array, ivy.NativeArray],
        true_lengths : Optional[Union[ivy.Array, ivy.NativeArray]],
        blank : Optional[int] = 0,
        zero_infinity: Optional[bool] = False,
        reduction : Optional[str] = "mean",
        out : Optional[ivy.Array] = None,
    ) -> ivy.Array:

        """
        ivy.Array instance method variant of ivy.ctc_loss. This method simply wraps the function, and so the docstring for
        ivy.ctc_loss also applies to this method with minimal changes.

        Parameters
        ----------
        self: input array of true labels.
        pred: input array of predicted labels.
        true_lengths: input array of true label lengths.
        pred_lengths: input array of predicted label lengths.
        blank: index of the blank label. Default: 0.
        reduction: specifies the reduction to apply to the output. Default: "mean". Allowed values: "none", "mean", "sum".
        zero_infinity: whether to zero infinite losses and the associated gradients. Default: True.
        out:  optional output array, for writing the result to. It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret: The CTC loss between the given distributions.

        Examples
        --------
        >>> logits = ivy.Array([[[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1]],
                           [[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]]])
        >>> labels = [ivy.Array([[1, 2], ivy.Array([1, 2]]))]
        >>> label_lengths = [ivy.Array([2, 2])]
        >>> logits_lengths = [ivy.Array([3, 3])]
        >>> loss = ivy.Array([0., 0.])
        >>> loss = loss.ctc_loss(labels, logits, label_lengths, logits_lengths)
        >>> print(loss)
        ivy.Array([2.9921765, 2.9650042])
        """

        #true = true.astype(ivy.int32)
        #true_lengths = true_lengths.astype(ivy.int64)
        #pred_lengths = pred_lengths.astype(ivy.int64)
        #blank = np.int32(blank)
        return ivy.ctc_loss(
            pred,
            true._data,
            pred_lengths,
            true_lengths,
            blank=blank,
            zero_infinity=zero_infinity,
            reduction=reduction,
            out=out)

    
