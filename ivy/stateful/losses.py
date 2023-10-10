"""Collection of Ivy's losses as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module


class LogPoissonLoss(Module):
    def __init__(
        self,
        *,
        compute_full_loss: bool = False,
        axis: int = -1,
        reduction: str = "none",
    ):
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


class CrossEntropyLoss(Module):
    def __init__(
        self,
        *,
        axis: int = -1,
        epsilon: float = 1e-7,
        reduction: str = "sum",
    ):
        self._axis = axis
        self._epsilon = epsilon
        self._reduction = reduction
        Module.__init__(self)

    def _forward(self, true, pred, *, axis=None, epsilon=None, reduction=None):
        """
        Perform forward pass of the Cross Entropy Loss.

        true
            input array containing true labels.
        pred
            input array containing Predicted labels.
        axis
            the axis along which to compute the cross-entropy loss. If axis is ``-1``,
            the cross-entropy loss will be computed along the last dimension.
            Default: ``-1``.
        epsilon
            small value to avoid division by zero. Default: ``1e-7``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'sum'``.

        Returns
        -------
        ret
            The cross-entropy loss between the given distributions.
        """
        return ivy.cross_entropy(
            true,
            pred,
            axis=ivy.default(axis, self._axis),
            epsilon=ivy.default(epsilon, self._epsilon),
            reduction=ivy.default(reduction, self._reduction),
        )


class BinaryCrossEntropyLoss(Module):
    def __init__(
        self,
        *,
        from_logits: bool = False,
        epsilon: float = 0.0,
        reduction: str = "none",
    ):
        self._from_logits = from_logits
        self._epsilon = epsilon
        self._reduction = reduction
        Module.__init__(self)

    def _forward(
        self,
        true,
        pred,
        *,
        from_logits=None,
        epsilon=None,
        reduction=None,
        pos_weight=None,
        axis=None,
    ):
        """
        Parameters
        ----------
        true
            input array containing true labels.
        pred
            input array containing Predicted labels.
        from_logits
            Whether `pred` is expected to be a logits tensor. By
            default, we assume that `pred` encodes a probability distribution.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied. Default: ``0``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``.
        pos_weight
            a weight for positive examples. Must be an array with length equal to the
            number of classes.
        axis
            Axis along which to compute crossentropy.

        Returns
        -------
        ret
            The binary cross entropy between the given distributions.
        """
        return ivy.binary_cross_entropy(
            true,
            pred,
            from_logits=ivy.default(from_logits, self._from_logits),
            epsilon=ivy.default(epsilon, self._epsilon),
            reduction=ivy.default(reduction, self._reduction),
            pos_weight=pos_weight,
            axis=axis,
        )


class DINOLoss(Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.center = ivy.zeros(1, out_dim)
        self.teacher_temp_schedule = ivy.concat(
            (
                ivy.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                ivy.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def _forward(self, student_output, teacher_output, epoch):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = ivy.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip if teacher and student work on the same view
                    continue
                loss = ivy.sum(-q * ivy.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        ivy.stop_gradient(teacher_output)
        batch_center = ivy.sum(teacher_output, dim=0, keepdim=True)
        # TODO: Check dist.all_reduce implementation in IVY
        batch_center = batch_center / len(
            teacher_output
        )  # check for dist.get_world_size()
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
