"""
Collection of Ivy normalization classes.
"""

# local
import ivy
from ivy.neural_net_stateful.module import Module
from ivy.neural_net_stateful.initializers import Zeros, Ones


class LayerNorm(Module):

    def __init__(self, normalized_shape, epsilon=None, elementwise_affine=True, new_std=None, dev=None, v=None):
        """
        Class for applying Layer Normalization over a mini-batch of inputs

        :param normalized_shape: Trailing shape to applying the normalization to.
        :type normalized_shape: int or sequence of ints
        :param epsilon: small constant to add to the denominator, use global ivy._MIN_BASE by default.
        :type epsilon: float, optional
        :param elementwise_affine: Whether to include learnable affine parameters, default is True.
        :type elementwise_affine: bool, optional
        :param new_std: The standard deviation of the new normalized values. Default is 1.
        :type new_std: float, optional
        :param dev: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
        :type dev: ivy.Device, optional
        :param v: the variables for each submodule in the sequence, constructed internally by default.
        :type v: ivy container of variables, optional
        """
        self._normalized_idxs = [-(i+1) for i in range(len(normalized_shape))]
        self._epsilon = epsilon
        self._elementwise_affine = elementwise_affine
        self._new_std = new_std
        self._scale_shape = normalized_shape
        self._offset_shape = normalized_shape
        self._scale_init = Ones()
        self._offset_init = Zeros()
        Module.__init__(self, dev, v)

    def _create_variables(self, dev):
        """
        Create internal variables for the layer
        """
        if self._elementwise_affine:
            return {'scale': self._scale_init.create_variables(self._scale_shape, dev),
                    'offset': self._offset_init.create_variables(self._offset_shape, dev)}
        return {}

    def _forward(self, inputs):
        """
        Perform forward pass of the LayerNorm layer.

        :param inputs: Inputs to process.
        :type inputs: array
        :return: The outputs following the layer normalization operation.
        """
        return ivy.layer_norm(inputs, self._normalized_idxs, epsilon=self._epsilon,
                              scale=self.v.scale if self._elementwise_affine else None,
                              offset=self.v.offset if self._elementwise_affine else None,
                              new_std=self._new_std)
