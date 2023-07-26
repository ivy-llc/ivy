"""Collection of Ivy's losses as stateful classes."""

# local
import ivy


def LogPoissonLoss(Module):
    def __init__(
        self,
        true,
        pred,
        *,
        compute_full_loss: bool = False,
        axis: int = -1,
        reduction: str = "none"
    ):
        self._true = true
        self._pred = pred
        self._compute_full_loss = compute_full_loss
        self._axis = axis
        self._reduction = reduction
        Module.__init__(self)

    def _forward(
        self, true, pred, *, compute_full_loss=None, axis=None, reduction=None
    ):
        """
        Perform forward pass of the Log Poisson Loss.

        true
            input array containing true labels.
        pred
            input array containing Predicted labels.
        compute_full_loss
            whether to compute the full loss. If false, a constant term is dropped
            in favor of more efficient optimization. Default: ``False``.
        axis
            the axis along which to compute the log-likelihood loss. If axis is ``-1``,
            the log-likelihood loss will be computed along the last dimension.
            Default: ``-1``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The binary log-likelihood loss between the given distributions.
        """
        return ivy.log_poisson_loss(
            true,
            pred,
            compute_full_loss=ivy.default(compute_full_loss, self._compute_full_loss),
            axis=ivy.default(axis, self._axis),
            reduction=ivy.default(reduction, self._reduction),
        )
