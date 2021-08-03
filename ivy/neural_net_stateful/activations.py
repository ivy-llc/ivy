"""
Collection of Ivy neural network activations as stateful classes.
"""

# local
import ivy
from ivy.neural_net_stateful.module import Module


class GEGLU(Module):

    def __init__(self):
        """
        Applies the GEGLU activation function.
        """
        Module.__init__(self, None, None)

    def _create_variables(self, dev_str):
        """
        Create internal variables for the layer
        """
        return {}

    def _forward(self, inputs):
        """
        Perform forward pass of the GEGLU activation.

        :param inputs: Inputs to process *[batch_shape, 2d]*.
        :type inputs: array
        :return: The outputs following the GEGLU activation *[batch_shape, d]*
        """
        x, gates = ivy.split(inputs, 2, -1)
        return x * ivy.gelu(gates)
