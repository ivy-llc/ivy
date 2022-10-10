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
        x, gates = ivy.split(inputs, num_or_size_splits=2, axis=-1)
        return ivy.gelu(gates) * x
