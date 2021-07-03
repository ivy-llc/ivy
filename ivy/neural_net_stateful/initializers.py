# global
import ivy


class Zeros:

    def __init__(self, var_shape=None, input_channels=None, output_channels=None):
        self._var_shape = var_shape

    def create_variables(self, dev_str, var_shape=None):
        """
        Create internal variables for the layer
        """
        var_shape = var_shape if var_shape is not None else self._var_shape
        return ivy.variable(ivy.zeros(var_shape, dev_str=dev_str))


class Xavier:

    def __init__(self, var_shape=None, input_channels=None, output_channels=None):
        self._var_shape = var_shape
        if input_channels is None:
            raise Exception('input_channels must be specified for Xavier initializer.')
        self._input_channels = input_channels
        if output_channels is None:
            raise Exception('output_channels must be specified for Xavier initializer.')
        self._output_channels = output_channels

    def create_variables(self, dev_str, var_shape=None):
        """
        Create internal variables for the layer
        """
        wlim = (6 / (self._output_channels + self._input_channels)) ** 0.5
        var_shape = var_shape if var_shape is not None else self._var_shape
        return ivy.variable(ivy.random_uniform(-wlim, wlim, var_shape, dev_str=dev_str))


class FirstLayerSiren:

    def __init__(self, var_shape=None, input_channels=None, output_channels=None):
        self._var_shape = var_shape
        if input_channels is None:
            raise Exception('input_channels must be specified for Xavier initializer.')
        self._input_channels = input_channels
        if output_channels is None:
            raise Exception('output_channels must be specified for Xavier initializer.')
        self._output_channels = output_channels

    def create_variables(self, dev_str, var_shape=None):
        """
        Create internal variables for the layer
        """
        wlim = 1 / self._input_channels
        var_shape = var_shape if var_shape is not None else self._var_shape
        return ivy.variable(ivy.random_uniform(-wlim, wlim, var_shape, dev_str=dev_str))


class Siren:

    def __init__(self, var_shape=None, input_channels=None, output_channels=None):
        self._var_shape = var_shape
        if input_channels is None:
            raise Exception('input_channels must be specified for Xavier initializer.')
        self._input_channels = input_channels
        if output_channels is None:
            raise Exception('output_channels must be specified for Xavier initializer.')
        self._output_channels = output_channels

    def create_variables(self, dev_str, var_shape=None):
        """
        Create internal variables for the layer
        """
        wlim = ((6 / self._input_channels) ** 0.5) / 30
        var_shape = var_shape if var_shape is not None else self._var_shape
        return ivy.variable(ivy.random_uniform(-wlim, wlim, var_shape, dev_str=dev_str))
