# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithLossesExperimental(abc.ABC):
    # ... (other methods)

    def hinge_embedding_loss(
        self: ivy.Array,
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        margin: Optional[float] = 1.0,
        reduction: Optional[str] = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.hinge_embedding_loss. This method
        simply wraps the function, and so the docstring for ivy.hinge_embedding_loss
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        pred
            input array containing predicted scores or logits.
        margin
            Margin value for the hinge loss. Default: 1.0.
        reduction
            Specifies the reduction to apply to the output. Default: "mean".
            - "none": No reduction will be applied.
            - "mean": The output will be averaged.
            - "sum": The output will be summed.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The hinge embedding loss between the given predictions and true labels.

        Examples
        --------
        >>> x = ivy.array([1, -1, 2])
        >>> y = ivy.array([1, -1, -1])
        >>> z = x.hinge_embedding_loss(y)
        >>> print(z)
        ivy.array(0.6666667)
        """
        return ivy.hinge_embedding_loss(
            self._data, target, margin=margin, reduction=reduction, out=out
        )
