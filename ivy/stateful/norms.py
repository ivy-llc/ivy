"""Collection of Ivy normalization classes."""

# local
import ivy
from ivy.stateful.module import Module
from ivy.stateful.initializers import Zeros, Ones


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        /,
        *,
        epsilon: float = ivy._MIN_BASE,
        elementwise_affine: bool = True,
        new_std: float = 1.0,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Layer Normalization over a mini-batch of inputs

        Parameters
        ----------
        normalized_shape
            Trailing shape to applying the normalization to.
        epsilon
            small constant to add to the denominator,
            use global ivy._MIN_BASE by default.
        elementwise_affine
            Whether to include learnable affine parameters, default is ``True``.
        new_std
            The standard deviation of the new normalized values. Default is 1.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        v
            the variables for each submodule in the sequence,
            constructed internally by default.
        """
        self._normalized_idxs = [-(i + 1) for i in range(len(normalized_shape))]
        self._epsilon = epsilon
        self._elementwise_affine = elementwise_affine
        self._new_std = new_std
        self._weight_shape = normalized_shape
        self._bias_shape = normalized_shape
        self._weight_init = Ones()
        self._bias_init = Zeros()
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer"""
        if self._elementwise_affine:
            return {
                "weight": self._weight_init.create_variables(
                    self._weight_shape, device, dtype=dtype
                ),
                "bias": self._bias_init.create_variables(
                    self._bias_shape, device, dtype=dtype
                ),
            }
        return {}

    def _forward(self, inputs):
        """
        Perform forward pass of the LayerNorm layer.

        Parameters
        ----------
        inputs
            Inputs to process.

        Returns
        -------
        ret
            The outputs following the layer normalization operation.
        """
        return ivy.layer_norm(
            inputs,
            self._normalized_idxs,
            epsilon=self._epsilon,
            scale=self.v.weight if self._elementwise_affine else None,
            b=self.v.bias if self._elementwise_affine else None,
            new_std=self._new_std,
        )
