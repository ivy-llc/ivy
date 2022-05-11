"""Collection of Ivy neural network activations as stateful classes."""

# local
import ivy
from ivy.stateful.module import Module


class GELU(Module):
    def __init__(self):
        """Applies the GELU activation function."""
        Module.__init__(self)

    def _forward(self, inputs):
        """
        Perform forward pass of the GELU activation.

        Parameters
        ----------
        inputs
            Inputs to process *[batch_shape, d]*.

        Returns
        -------
        ret
            The outputs following the GELU activation *[batch_shape, d]*

        """
        return ivy.gelu(inputs)


class GEGLU(Module):
    def __init__(self):
        """Applies the GEGLU activation function."""
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
        x, gates = ivy.split(inputs, 2, -1)
        return ivy.gelu(gates) * x
