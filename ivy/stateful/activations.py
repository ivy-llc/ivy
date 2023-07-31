"""Collection of Ivy neural network activations as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module
from ivy.core.container import Container
from ivy.framework_handler import choose_framework


class GELU(Module):
    def __init__(self, *, approximate: bool = False):
        """Apply the GELU activation function."""
        self._approximate = approximate
        Module.__init__(self)

    def _forward(self, x, /, *, approximate=None):
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
        return ivy.gelu(x, approximate=ivy.default(approximate, self._approximate))


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
    def __init__(self):
        """Apply the RELU activation function."""
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
            The outputs following the RELU activation *[batch_shape, d]*
        """
        return ivy.relu(x)


class LeakyReLU(Module):
    def __init__(self, alpha: float = 0.2):
        """
        Apply the LEAKY RELU activation function.

        Parameters
        ----------
        alpha
             Negative slope for ReLU.
        """
        self._alpha = alpha
        Module.__init__(self)

    def _forward(self, x, *, alpha=None):
        """

        Parameters
        ----------
        x
              Inputs to process *[batch_shape, d]*.
        alpha
              Negative slope for ReLU.

        Returns
        -------
        ret
            The outputs following the LEAKY RELU activation *[batch_shape, d]*
        """
        return ivy.leaky_relu(x, alpha=ivy.default(alpha, self._alpha))


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
    def __init__(self):
        """Apply the TANH activation function."""
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
            The outputs following the TANH activation *[batch_shape, d]*
        """
        return ivy.tanh(x)


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

class ThresholdedReLU(Module):
    def __init__(self, threshold=0.0):
        super(ThresholdedReLU, self).__init__()
        self.threshold = threshold

    def _forward(self, x):
        """
        Apply the ThresholdedReLU activation function to the input tensor x.

        Parameters
        ----------
        x : array-like
            Input tensor.

        Returns
        -------
        array-like
            Output tensor after applying ThresholdedReLU activation function.
        """
        backend = choose_framework(x)
        return ivy.maximum(x - self.threshold, backend.zeros_like(x))

    def forward(self, x):
        """
        Apply the ThresholdedReLU activation function to the input tensor x.

        Parameters
        ----------
        x : array-like
            Input tensor.

        Returns
        -------
        Container
            Container with a single entry for the output tensor after applying ThresholdedReLU activation function.
        """
        return Container(self._forward(x))