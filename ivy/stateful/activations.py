"""Collection of Ivy neural network activations as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module


class GELU(Module):
    def __init__(self, *, approximate=True):
        """Applies the GELU activation function."""
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

    _forward.unsupported_dtypes = {"torch": ("float16",)}


class GEGLU(Module):
    def __init__(self, *, approximate=True):
        """Applies the GEGLU activation function."""
        self._approximate = approximate
        Module.__init__(self)

    def _forward(self, inputs, gates, /, *, approximate=None):
        """
        Perform forward pass of the GEGLU activation. Calculated by the following
        formula: activation = GELU(inputs) * gates

        Parameters
        ----------
        inputs
            Inputs to process

        gates
            Array of gates used in the GEGLU activation function. Must have the same
            shape as the inputs

        Returns
        -------
        ret
            The outputs the GEGLU activation of the inputs and the gates

        """
        return (
            ivy.gelu(inputs, approximate=ivy.default(approximate, self._approximate))
            * gates
        )

    _forward.unsupported_dtypes = {"torch": ("float16",)}
