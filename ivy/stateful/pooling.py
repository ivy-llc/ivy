"""Collection of Ivy pooling classes."""

# local
import ivy
from ivy.stateful.module import Module


class MaxPool2D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Max Pooling over a mini-batch of inputs

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.max_pool2d(inputs, self._kernel_size, self._stride, self._padding)


class AvgPool2D(Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        /,
        *,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Average Pooling over a mini-batch of inputs

        Parameters
        ----------
        kernel_size
            The size of the window to take a max over.
        stride
            The stride of the window. Default value: 1
        padding
            Implicit zero padding to be added on both sides.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
        """
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        Module.__init__(self, device=device, dtype=dtype)

    def _forward(self, inputs):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x
            The input to the layer.

        Returns
        -------
        The output of the layer.
        """
        return ivy.avg_pool2d(inputs, self._kernel_size, self._stride, self._padding)
