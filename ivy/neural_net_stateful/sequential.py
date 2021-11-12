"""
Base class for deriving trainable modules
"""

# local
from ivy.neural_net_stateful.module import Module


class Sequential(Module):

    def __init__(self, *sub_modules, dev_str=None, v=None):
        """
        A sequential container. Modules will be added to it in the order they are passed in the constructor.

        :param submodules: Submodules to chain together into a sequence.
        :type submodules: sequence of ivy.Module instances
        :param dev_str: device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu' etc.
        :type dev_str: str, optional
        :param v: the variables for each submodule in the sequence, constructed internally by default.
        :type v: ivy container of variables, optional
        """
        if v is not None:
            for i, submod in enumerate(sub_modules):
                try:
                    submod.v = v['submodules']['v' + str(i)]
                except KeyError:
                    if submod.v:
                        raise Exception('variables v passed to Sequential class must have key chains in the form of'
                                        '"submodules/v{}", where {} is an idx')
        self._submodules = list(sub_modules)
        Module.__init__(self, dev_str, v)

    def _forward(self, inputs):
        """
        Perform forward pass of the Linear layer.

        :param inputs: Inputs to process.
        :type inputs: array
        :return: The outputs following the linear operation and bias addition.
        """
        x = inputs
        for i, submod in enumerate(self._submodules):
            try:
                x = submod(x, v=self.v.submodules['v' + str(i)])
            except KeyError:
                if submod.v:
                    raise Exception('variables v passed to Sequential class must have key chains in the form of'
                                    '"submodules/v{}", where {} is an idx')
                x = submod(x)
        return x
