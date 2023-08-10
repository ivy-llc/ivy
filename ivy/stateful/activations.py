"""Collection of Ivy neural network activations as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module
from typing import Literal


class GELU(Module):
    def __init__(
        self,
        *,
        approximate: bool = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """Apply the GELU activation function."""
        self._approximate = approximate
        self._complex_mode = complex_mode
        Module.__init__(self)

    def _forward(
        self,
        x,
        /,
        *,
        approximate=None,
        complex_mode=None,
    ):
        """
        Perform forward pass of the GELU activation.

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.

        Returns
        -------
        ret
            The outputs following the GELU activation *[batch_shape, d]*
        """
        return ivy.gelu(
            x,
            approximate=ivy.default(approximate, self._approximate),
            complex_mode=ivy.default(complex_mode, self._complex_mode),
        )


class GEGLU(Module):
    def __init__(self):
        """Apply the GEGLU activation function."""
        Module.__init__(self)

    def _forward(self, inputs):
        """
        Perform forward pass of the GEGLU activation.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_shape, 2d]*.

        Returns
        -------
        ret
            The outputs following the GEGLU activation *[batch_shape, d]*
        """
        x, gates = ivy.split(inputs, num_or_size_splits=2, axis=-1)
        return ivy.gelu(gates) * x


class ReLU(Module):
    def __init__(
        self,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """Apply the RELU activation function."""
        self._complex_mode = complex_mode
        Module.__init__(self)

    def _forward(self, x, complex_mode=None):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.

        Returns
        -------
        ret
            The outputs following the RELU activation *[batch_shape, d]*
        """
        return ivy.relu(x, complex_mode=ivy.default(complex_mode, self._complex_mode))


class LeakyReLU(Module):
    def __init__(
        self,
        alpha: float = 0.2,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """
        Apply the LEAKY RELU activation function.

        Parameters
        ----------
        alpha
             Negative slope for ReLU.
        complex_mode
             Specifies how to handle complex input.
        """
        self._alpha = alpha
        self._complex_mode = complex_mode
        Module.__init__(self)

    def _forward(self, x, *, alpha=None, complex_mode=None):
        """

        Parameters
        ----------
        x
              Inputs to process *[batch_shape, d]*.
        alpha
              Negative slope for ReLU.
        complex_mode
              Specifies how to handle complex input.

        Returns
        -------
        ret
            The outputs following the LEAKY RELU activation *[batch_shape, d]*
        """
        return ivy.leaky_relu(
            x,
            alpha=ivy.default(alpha, self._alpha),
            complex_mode=ivy.default(complex_mode, self._complex_mode),
        )


class LogSoftmax(Module):
    def __init__(self):
        """Apply the LOG SOFTMAX activation function."""
        Module.__init__(self)

    def _forward(self, x, *, axis=None):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.
        axis
            The dimension log_softmax would be performed on. The default is ``None``
        Returns
        -------
         ret
            The outputs following the LOG SOFTMAX activation *[batch_shape, d]*
        """
        return ivy.log_softmax(x, axis=axis)


class Softmax(Module):
    def __init__(self):
        """Apply the SOFTMAX activation function."""
        Module.__init__(self)

    def _forward(self, x, *, axis=None):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.
        axis
            The dimension softmax would be performed on. The default is ``None``.

        Returns
        -------
          ret
            The outputs following the SOFTMAX activation *[batch_shape, d]*

        """
        return ivy.softmax(x, axis=axis)


class Softplus(Module):
    def __init__(self):
        """Apply the SOFTPLUS activation function."""
        Module.__init__(self)

    def _forward(self, x, *, beta=None, threshold=None):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.
        beta
            The beta value for the softplus formation. Default: ``None``.

        threshold
             values above this revert to a linear function. Default: ``None``.

        Returns
        -------
        ret
            The outputs following the SOFTPLUS activation *[batch_shape, d]*

        """
        return ivy.softplus(x, beta=beta, threshold=threshold)


class Mish(Module):
    def __init__(self):
        """Apply the MISH activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the MISH activation *[batch_shape, d]*
        """
        return ivy.mish(x)


class SiLU(Module):
    def __init__(self):
        """Apply the SiLU activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the SiLU activation *[batch_shape, d]*
        """
        return ivy.silu(x)


class Sigmoid(Module):
    def __init__(self):
        """Apply the SIGMOID activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the SIGMOID activation *[batch_shape, d]*
        """
        return ivy.sigmoid(x)


class Tanh(Module):
    def __init__(self, complex_mode: Literal["split", "magnitude", "jax"] = "jax"):
        """
        Apply the TANH activation function.

        Parameters
        ----------
        complex_mode
            Specifies how to handle complex input.
        """
        self._complex_mode = complex_mode
        Module.__init__(self)

    def _forward(self, x, complex_mode=None):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.
        complex_mode
             Specifies how to handle complex input.

        Returns
        -------
         ret
            The outputs following the TANH activation *[batch_shape, d]*
        """
        return ivy.tanh(x, complex_mode=ivy.default(complex_mode, self._complex_mode))


class ReLU6(Module):
    def __init__(self):
        """Apply the RELU6 activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the RELU6 activation *[batch_shape, d]*
        """
        return ivy.relu6(x)


class Hardswish(Module):
    def __init__(self):
        """Apply the HARDSWISH activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the HARDSWISH activation *[batch_shape, d]*
        """
        return ivy.hardswish(x)


class Logit(Module):
    def __init__(self):
        """Apply the LOGIT activation function."""
        Module.__init__(self)

    def _forward(self, x, eps=None):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.
        eps
            The epsilon value for the logit formation. Default: ``None``.

        Returns
        -------
        ret
            The outputs following the LOGIT activation *[batch_shape, d]*
        """
        return ivy.logit(x, eps=eps)


class PReLU(Module):
    def __init__(self):
        """Apply the PRELU activation function."""
        Module.__init__(self)

    def _forward(self, x, slope):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.
        slope
            The slope value for the prelu formation.

        Returns
        -------
        ret
            The outputs following the PRELU activation *[batch_shape, d]*
        """
        return ivy.prelu(x, slope)


class SeLU(Module):
    def __init__(self):
        """Apply the SELU activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
             Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the SELU activation *[batch_shape, d]*
        """
        return ivy.selu(x)


class ELU(Module):
    def __init__(self):
        """Apply the ELU activation function."""
        Module.__init__(self)

    def _forward(self, x, alpha=1.0):
        """
        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.
        alpha
            scaler for controlling the slope of the function for x <= 0 Default: 1.0
        Returns
        -------
        ret
            The outputs following the ELU activation *[batch_shape, d]*
        """
        return ivy.elu(x, alpha=alpha)


class LogSigmoid(Module):
    def __init__(self):
        """Apply the LogSigmoid activation function."""
        Module.__init__(self)

    def _forward(self, x):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.

        Returns
        -------
        ret
            The outputs following the LogSigmoid activation *[batch_shape, d]*
        """
        return ivy.logsigmoid(x)
