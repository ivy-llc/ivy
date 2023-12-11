"""Collection of Ivy neural network activations as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module
from typing import Literal, Optional


class GELU(Module):
    def __init__(
        self,
        *,
        approximate: bool = False,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """Apply the GELU activation function.

        Parameters
        ----------
        approximate
            whether to use the gelu approximation algorithm or exact formulation.
        complex_mode
            Specifies how to handle complex input. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._approximate = approximate
        self._complex_mode = complex_mode
        Module.__init__(self)

    def _forward(self, x):
        """Perform forward pass of the GELU activation.

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
            approximate=self._approximate,
            complex_mode=self._complex_mode,
        )

    def _extra_repr(self) -> str:
        return f"approximate={self._approximate}, complex_mode={self._complex_mode}"


class GEGLU(Module):
    def __init__(self):
        """Apply the GEGLU activation function."""
        Module.__init__(self)

    def _forward(self, inputs):
        """Perform forward pass of the GEGLU activation.

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
        """Apply the RELU activation function.

        Parameters
        ----------
        complex_mode
            Specifies how to handle complex input. See
             ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._complex_mode = complex_mode
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
        return ivy.relu(x, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"complex_mode={self._complex_mode}"


class LeakyReLU(Module):
    def __init__(
        self,
        alpha: float = 0.2,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """Apply the LEAKY RELU activation function.

        Parameters
        ----------
        alpha
            Negative slope for ReLU.
        complex_mode
            Specifies how to handle complex input. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._alpha = alpha
        self._complex_mode = complex_mode
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
            The outputs following the LEAKY RELU activation *[batch_shape, d]*
        """
        return ivy.leaky_relu(
            x,
            alpha=self._alpha,
            complex_mode=self._complex_mode,
        )

    def _extra_repr(self) -> str:
        return f"alpha={self._alpha}, complex_mode={self._complex_mode}"


class LogSoftmax(Module):
    def __init__(
        self,
        axis: Optional[int] = -1,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """Apply the LOG SOFTMAX activation function.

        Parameters
        ----------
        axis
            The dimension log_softmax would be performed on. The default is ``None``
        complex_mode
            optional specifier for how to handle complex data types. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        Module.__init__(self)
        self._axis = axis
        self._complex_mode = complex_mode

    def _forward(self, x):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.

        Returns
        -------
         ret
            The outputs following the LOG SOFTMAX activation *[batch_shape, d]*
        """
        return ivy.log_softmax(x, axis=self._axis, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"axis={self._axis}, complex_mode={self._complex_mode}"


class Softmax(Module):
    def __init__(
        self,
        axis: int = -1,
        complex_mode: Literal["split", "magnitude", "jax"] = "jax",
    ):
        """Apply the SOFTMAX activation function.

        Parameters
        ----------
        axis
            The axis which we apply softmax op on.
        complex_mode
            Specifies how to handle complex input. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        Module.__init__(self)
        self._axis = axis
        self._complex_mode = complex_mode

    def _forward(self, x):
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
        return ivy.softmax(x, axis=self._axis, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"axis={self._axis}, complex_mode={self._complex_mode}"


class Softplus(Module):
    def __init__(self, beta=1.0, threshold=None):
        """Apply the SOFTPLUS activation function."""
        Module.__init__(self)
        self._beta = beta
        self._threshold = threshold

    def _forward(self, x):
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
        return ivy.softplus(x, beta=self._beta, threshold=self._threshold)

    def _extra_repr(self) -> str:
        return f"beta={self._beta}, threshold={self._threshold}"


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
    def __init__(self, complex_mode: Literal["split", "magnitude", "jax"] = "jax"):
        """Apply the SIGMOID activation function.

        Parameter
        ----------
        complex_mode
            Specifies how to handle complex input. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._complex_mode = complex_mode
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
        return ivy.sigmoid(x, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"complex_mode={self._complex_mode}"


class Tanh(Module):
    def __init__(self, complex_mode: Literal["split", "magnitude", "jax"] = "jax"):
        """Apply the TANH activation function.

        Parameters
        ----------
        complex_mode
            Specifies how to handle complex input. See
             ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._complex_mode = complex_mode
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
        return ivy.tanh(x, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"complex_mode={self._complex_mode}"


class ReLU6(Module):
    def __init__(self, complex_mode: Literal["split", "magnitude", "jax"] = "jax"):
        """Apply the TANH activation function.

        Parameters
        ----------
        complex_mode
            Specifies how to handle complex input. See
             ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._complex_mode = complex_mode
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
        return ivy.relu6(x, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"complex_mode={self._complex_mode}"


class Hardswish(Module):
    def __init__(self, complex_mode: Literal["split", "magnitude", "jax"] = "jax"):
        """Apply the HARDSWISH activation function.

        Parameters
        ----------
        complex_mode
            Specifies how to handle complex input. See
             ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._complex_mode = complex_mode
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
        return ivy.hardswish(x, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"complex_mode={self._complex_mode}"


class Logit(Module):
    def __init__(
        self,
        eps=None,
        complex_mode="jax",
    ):
        """Apply the LOGIT activation function.

        Parameters
        ----------
        eps
             The epsilon value for the logit formation. Default: ``None``.
        complex_mode
             optional specifier for how to handle complex data types. See
             ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        Module.__init__(self)
        self._eps = eps
        self._complex_mode = complex_mode

    def _forward(self, x):
        """

        Parameters
        ----------
        x
            Inputs to process *[batch_shape, d]*.

        Returns
        -------
        ret
            The outputs following the LOGIT activation *[batch_shape, d]*
        """
        return ivy.logit(
            x,
            eps=self._eps,
            complex_mode=self._complex_mode,
        )

    def _extra_repr(self) -> str:
        return f"eps={self._eps}, complex_mode={self._complex_mode}"


class PReLU(Module):
    def __init__(self, slope):
        """Apply the PRELU activation function."""
        Module.__init__(self)
        self._slope = slope

    def _forward(self, x):
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
        return ivy.prelu(x, self._slope)

    def _extra_repr(self) -> str:
        return f"slope={self._slope}"


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
    def __init__(self, alpha=1.0):
        """Apply the ELU activation function."""
        Module.__init__(self)
        self._alpha = alpha

    def _forward(self, x):
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
        return ivy.elu(x, alpha=self._alpha)

    def _extra_repr(self) -> str:
        return f"alpha={self._alpha}"


class LogSigmoid(Module):
    def __init__(self, complex_mode: Literal["split", "magnitude", "jax"] = "jax"):
        """Apply the LogSigmoid activation function.

        Parameter
        ----------
        complex_mode
            Specifies how to handle complex input. See
            ``ivy.func_wrapper.handle_complex_input`` for more detail.
        """
        self._complex_mode = complex_mode
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
        return ivy.logsigmoid(x, complex_mode=self._complex_mode)

    def _extra_repr(self) -> str:
        return f"complex_mode={self._complex_mode}"
