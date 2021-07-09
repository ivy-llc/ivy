# local
import ivy


# Constant #
# ---------#

class Constant:

    def __init__(self, constant):
        self._constant = constant

    def create_variables(self, var_shape, dev_str, fan_out=None, fan_in=None):
        """
        Create internal variables for the layer
        """
        return ivy.variable(ivy.ones(var_shape, dev_str=dev_str) * self._constant)


class Zeros(Constant):

    def __init__(self):
        super().__init__(0.)


class Ones(Constant):

    def __init__(self):
        super().__init__(1.)


# Uniform #
# --------#

class Uniform:

    def __init__(self, numerator, denominator_mode, power, gain):
        if denominator_mode not in ['fan_in', 'fan_out', 'fan_sum', 'fan_avg']:
            raise Exception('Invalid denominator mode, must be one of [ fan_in | fan_out | fan_sum | fan_avg ]')
        self._numerator = numerator
        self._denominator_mode = denominator_mode
        self._power = power
        self._gain = gain

    def create_variables(self, var_shape, dev_str, fan_out=None, fan_in=None):
        """
        Create internal variables for the layer
        """
        if self._denominator_mode == 'fan_in':
            if fan_in is None:
                raise Exception('input_channels must be specified for fan_in denominator mode.')
            denom = fan_in
        elif self._denominator_mode == 'fan_out':
            if fan_in is None:
                raise Exception('output_channels must be specified for fan_out denominator mode.')
            denom = fan_out
        elif self._denominator_mode == 'fan_sum':
            if fan_in is None or fan_out is None:
                raise Exception('input_channels and output_channels must both be specified for'
                                'fan_sum denominator mode.')
            denom = fan_in + fan_out
        elif self._denominator_mode == 'fan_avg':
            if fan_in is None or fan_out is None:
                raise Exception('input_channels and output_channels must both be specified for'
                                'fan_avg denominator mode.')
            denom = (fan_in + fan_out) / 2
        else:
            raise Exception('Invalid denominator mode, must be one of [ fan_in | fan_out | fan_sum | fan_avg ]')
        wlim = ((self._numerator / denom) ** self._power) * self._gain
        return ivy.variable(ivy.random_uniform(-wlim, wlim, var_shape, dev_str=dev_str))


class GlorotUniform(Uniform):

    def __init__(self):
        super().__init__(6, 'fan_sum', 0.5, 1)


class FirstLayerSiren(Uniform):

    def __init__(self):
        super().__init__(1, 'fan_in', 1, 1)


class Siren(Uniform):

    def __init__(self, w0=30):
        super().__init__(6, 'fan_in', 0.5, 1/w0)
