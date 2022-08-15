# local
from typing import Tuple, Union
import ivy


class Initalizer:
    def create_variables(
        self, 
        var_shape: Tuple[int, int], 
        device: Union[ivy.Device, ivy.NativeDevice], 
        fan_out: float = None, 
        fan_in: float = None, 
        dtype: Union[ivy.Dtype, ivy.NativeDtype] = None
    ) -> ivy.Variable:
        """
        Create internal variables for the layer
        
        Parameters
        ----------
        var_shape
            Tuple representing the shape of the desired array. If considering the array as a rectangular matrix, this tuple is represented as '(ROWS, COLUMNS)'.
        device
            Device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        fan_out
            The number of nodes in the next layer.
        fan_in
            The number of nodes in the previous layer.
        dtype
            Desired data type.
        """
        return None


# Constant #
# ---------#


class Constant(Initalizer):
    def __init__(self, constant):
        """
        Constant initializer, will fill in all values with the value of `constant`.

        Parameters
        ----------
        constant
            Constant value for initialization.
        """
        self._constant = constant

    def create_variables(
        self, 
        var_shape: Tuple[int, int], 
        device: Union[ivy.Device, ivy.NativeDevice], 
        fan_out: float = None, 
        fan_in: float = None, 
        dtype: Union[ivy.Dtype, ivy.NativeDtype] = None
    ) -> ivy.Variable:
        return ivy.variable(
            ivy.full(var_shape, self._constant, device=device, dtype=dtype),
        )


class Zeros(Constant):
    def __init__(self):
        """
        A constant initalizer that fills with the constant value `0.0`.
        """
        super().__init__(0.0)


class Ones(Constant):
    def __init__(self):
        """
        A constant initalizer that fills with the constant value `1.0`.
        """
        super().__init__(1.0)


# Uniform #
# --------#


class Uniform(Initalizer):
    def __init__(self, numerator, fan_mode, power, gain):
        if fan_mode not in ["fan_in", "fan_out", "fan_sum", "fan_avg"]:
            raise Exception(
                "Invalid fan mode, must be one of [ fan_in | fan_out | fan_sum | "
                "fan_avg ] "
            )
        self._numerator = numerator
        self._fan_mode = fan_mode
        self._power = power
        self._gain = gain

    def create_variables(
        self, var_shape, device, fan_out=None, fan_in=None, dtype=None
    ):
        """Create internal variables for the layer"""
        if self._fan_mode == "fan_in":
            if fan_in is None:
                raise Exception(
                    "input_channels must be specified for fan_in denominator mode."
                )
            fan = fan_in
        elif self._fan_mode == "fan_out":
            if fan_out is None:
                raise Exception(
                    "output_channels must be specified for fan_out denominator mode."
                )
            fan = fan_out
        elif self._fan_mode == "fan_sum":
            if fan_in is None or fan_out is None:
                raise Exception(
                    "input_channels and output_channels must both be specified for"
                    "fan_sum denominator mode."
                )
            fan = fan_in + fan_out
        elif self._fan_mode == "fan_avg":
            if fan_in is None or fan_out is None:
                raise Exception(
                    "input_channels and output_channels must both be specified for"
                    "fan_avg denominator mode."
                )
            fan = (fan_in + fan_out) / 2
        else:
            raise Exception(
                "Invalid denominator mode, must be one of [ fan_in | fan_out | "
                "fan_sum | fan_avg ] "
            )
        wlim = ((self._numerator / fan) ** self._power) * self._gain
        return ivy.variable(
            ivy.random_uniform(
                low=-wlim, high=wlim, shape=var_shape, device=device, dtype=dtype
            ),
        )


class GlorotUniform(Uniform):
    def __init__(self):
        super().__init__(6, "fan_sum", 0.5, 1)


class FirstLayerSiren(Uniform):
    def __init__(self):
        super().__init__(1, "fan_in", 1, 1)


class Siren(Uniform):
    def __init__(self, w0=30):
        super().__init__(6, "fan_in", 0.5, 1 / w0)


# Gaussian #
# ---------#


class KaimingNormal(Initalizer):
    def __init__(self, mean=0, fan_mode="fan_in"):
        if fan_mode not in ["fan_in", "fan_out", "fan_sum", "fan_avg"]:
            raise Exception(
                "Invalid fan mode, must be one of [ fan_in | fan_out | fan_sum | "
                "fan_avg ] "
            )
        self._mean = mean
        self._fan_mode = fan_mode

    def create_variables(
        self, var_shape, device, fan_out=None, fan_in=None, negative_slope=0.0
    ):
        """Create internal variables for the layer"""
        if self._fan_mode == "fan_in":
            if fan_in is None:
                raise Exception(
                    "input_channels must be specified for fan_in denominator mode."
                )
            fan = fan_in
        elif self._fan_mode == "fan_out":
            if fan_in is None:
                raise Exception(
                    "output_channels must be specified for fan_out denominator mode."
                )
            fan = fan_out
        elif self._fan_mode == "fan_sum":
            if fan_in is None or fan_out is None:
                raise Exception(
                    "input_channels and output_channels must both be specified for"
                    "fan_sum denominator mode."
                )
            fan = fan_in + fan_out
        elif self._fan_mode == "fan_avg":
            if fan_in is None or fan_out is None:
                raise Exception(
                    "input_channels and output_channels must both be specified for"
                    "fan_avg denominator mode."
                )
            fan = (fan_in + fan_out) / 2
        else:
            raise Exception(
                "Invalid denominator mode, must be one of [ fan_in | fan_out | "
                "fan_sum | fan_avg ] "
            )
        std = (2 / ((1 + negative_slope**2) * fan)) ** 0.5
        return ivy.variable(
            ivy.random_normal(mean=self._mean, std=std, shape=var_shape, device=device)
        )
