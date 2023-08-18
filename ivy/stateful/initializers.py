# global
from typing import Tuple, Union, Optional
import abc

# local
import ivy
from ivy.functional.ivy.gradients import _variable


# Initializer #
# ----------- #


class Initializer(abc.ABC):
    """
    An initializer for internal variables for a layer.

    A neuron is a function of the form `a = g(z)`, where `g` is the
    activation functions and `z = w_1x_1 + w_2x_2 + ... + w_nx_n` where the
    `w_i` are the weights and the `x_i` are the inputs. To prevent this
    `z` from vanishing (getting too small) or exploding (getting too big), the
    initial weights must be picked carefully.
    """

    @abc.abstractmethod
    def create_variables(
        self,
        var_shape: Tuple[int, int],
        device: Union[ivy.Device, ivy.NativeDevice],
        fan_out: Optional[float] = None,
        fan_in: Optional[float] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    ) -> ivy.Array:
        """
        Create internal variables for the layer.

        Parameters
        ----------
        var_shape
            Tuple representing the shape of the desired array. If considering
             the array as a rectangular matrix, this tuple is represented as
             '(ROWS, COLUMNS)'.
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


class Constant(Initializer):
    def __init__(self, constant: float):
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
        fan_out: Optional[float] = None,
        fan_in: Optional[float] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    ) -> ivy.Array:
        return _variable(
            ivy.full(var_shape, self._constant, device=device, dtype=dtype),
        )


class Zeros(Constant):
    def __init__(self):
        """Constant initalizer that fills with the constant value `0.0`."""
        super().__init__(0.0)


class Ones(Constant):
    def __init__(self):
        """Constant initalizer that fills with the constant value `1.0`."""
        super().__init__(1.0)


# Uniform #
# --------#


class Uniform(Initializer):
    def __init__(self, numerator, fan_mode, power, gain):
        """
        Initialize based on a uniform distribution, will fill in all values with values
        drawn from a uniform (all values have an equal probability) distribution.

        with range `[-wlim, wlim]` (endpoints included) with `wlim` being calculated as
        `gain * (numerator / fan)**power`. This distribution helps with issues when
        trying to optimize and train networks. The expected value of this distribution
        is `0` and the variance is
        `(gain * numerator / fan)^power / 4`.

        This is intended as a base-class for special predefined initialzers.

        Parameters
        ----------
        numerator
        fan_mode
            Determines how `fan` is calculated.
            - `fan_out` sets `fan` to the number of output features of this neuron.
              This is useful when training using back-propogation.
            - `fan_in` sets `fan` to the number of input features of this neuron.
              This is useful when training using forward-propogation.
            - `fan_sum` sets `fan` to the sum of the number of input features and
              output features of this neuron.
            - `fan_avg` sets `fan` to the average of the number of input features and
              output features of this neuron.
        power
            Sets the drop-off factor for the calculated `fan`.
        gain
            Scales the output of the distribution.
        """
        ivy.utils.assertions.check_elem_in_list(
            fan_mode, ["fan_in", "fan_out", "fan_sum", "fan_avg"]
        )
        self._numerator = numerator
        self._fan_mode = fan_mode
        self._power = power
        self._gain = gain

    def create_variables(
        self, var_shape, device, fan_out=None, fan_in=None, dtype=None
    ):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        var_shape
            Tuple representing the shape of the desired array. If considering
             the array as a rectangular matrix, this tuple is represented as
             '(ROWS, COLUMNS)'.
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
        if self._fan_mode == "fan_in":
            ivy.utils.assertions.check_exists(
                fan_in,
                message="input_channels must be specified for fan_in denominator mode",
            )
            fan = fan_in
        elif self._fan_mode == "fan_out":
            ivy.utils.assertions.check_exists(
                fan_out,
                message=(
                    "output_channels must be specified for fan_out denominator mode"
                ),
            )
            fan = fan_out
        elif self._fan_mode == "fan_sum":
            ivy.utils.assertions.check_all_or_any_fn(
                fan_in,
                fan_out,
                fn=ivy.exists,
                type="all",
                message=(
                    "input_channels and output_channels must both be"
                    " specified for fan_sum denominator mode."
                ),
                as_array=False,
            )
            fan = fan_in + fan_out
        elif self._fan_mode == "fan_avg":
            ivy.utils.assertions.check_all_or_any_fn(
                fan_in,
                fan_out,
                fn=ivy.exists,
                type="all",
                message=(
                    "input_channels and output_channels must both be"
                    " specified for fan_avg denominator mode."
                ),
                as_array=False,
            )
            fan = (fan_in + fan_out) / 2
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid denominator mode, must be one of [ fan_in | fan_out | "
                "fan_sum | fan_avg ] "
            )
        wlim = ((self._numerator / fan) ** self._power) * self._gain
        return _variable(
            ivy.random_uniform(
                low=-wlim, high=wlim, shape=var_shape, device=device, dtype=dtype
            ),
        )


class GlorotUniform(Uniform):
    def __init__(self):
        """
        Initialize Glorot uniform, also known as the Xavier uniform initializer.

        It draws values from a uniform distribtion `[-limit, limit]` where
        `limit = sqrt(6 / (fan_in + fan_out))` where `fan_in` and `fan_out` are the
        number of input and output features respectively.
        """
        super().__init__(numerator=6, fan_mode="fan_sum", power=0.5, gain=1)


class FirstLayerSiren(Uniform):
    def __init__(self):
        """
        Initialize Siren uniform for the first layer.

        It draws values from a uniform distribtion `[-limit, limit]`
        where `limit=fan_in` where `fan_in` is the number of input
        features.
        """
        super().__init__(numerator=1, fan_mode="fan_in", power=1, gain=1)


class Siren(Uniform):
    def __init__(self, w0=30):
        """
        Initialize Siren uniform initializer for the first layer.

        It draws values from a uniform distribtion `[-limit, limit]`
        where `limit=sqrt(6 / fan_in) / w0` where `fan_in` is the number
        of input features.
        """
        super().__init__(numerator=6, fan_mode="fan_in", power=0.5, gain=1 / w0)


# Gaussian #
# ---------#


class KaimingNormal(Initializer):
    def __init__(self, mean=0, fan_mode="fan_in"):
        """
        Initialize Kaiming normal, also known as He Initialization.

        It is an method for initializing layers that takes into account the
        non-linearity of activation functions. It uses a normal distribution centered
        at `mean` with standard distribution `sqrt(2 / ((1 + negative_slope^2) * fan))`.

        Parameters
        ----------
        mean
            Sets the expected value, average, and center of the normal distribution.
        fan_mode
            Determines how `fan` is calculated.
            - `fan_out` sets `fan` to the number of output features of this neuron.
              This is useful when training using back-propogation.
            - `fan_in` sets `fan` to the number of input features of this neuron.
              This is useful when training using forward-propogation.
            - `fan_sum` sets `fan` to the sum of the number of input features and
              output features of this neuron.
            - `fan_sum` sets `fan` to the average of the number of input features and
              output features of this neuron.
        """
        ivy.utils.assertions.check_elem_in_list(
            fan_mode, ["fan_in", "fan_out", "fan_sum", "fan_avg"]
        )
        self._mean = mean
        self._fan_mode = fan_mode

    def create_variables(
        self,
        var_shape,
        device,
        fan_out=None,
        fan_in=None,
        negative_slope=0.0,
        dtype=None,
    ):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        var_shape
            Tuple representing the shape of the desired array. If considering
             the array as a rectangular matrix, this tuple is represented as
             '(ROWS, COLUMNS)'.
        device
            Device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        fan_out
            The number of nodes in the next layer.
        fan_in
            The number of nodes in the previous layer.
        negative_slope
            How much a higher `fan` should lower the standard deviation. A value of `0`
            gives a relationship proportional to `1/fan`.
        dtype
            Desired data type.
        """
        if self._fan_mode == "fan_in":
            ivy.utils.assertions.check_exists(
                fan_in,
                message="input_channels must be specified for fan_in denominator mode",
            )
            fan = fan_in
        elif self._fan_mode == "fan_out":
            ivy.utils.assertions.check_exists(
                fan_out,
                message=(
                    "output_channels must be specified for fan_out denominator mode"
                ),
            )
            fan = fan_out
        elif self._fan_mode == "fan_sum":
            ivy.utils.assertions.check_all_or_any_fn(
                fan_in,
                fan_out,
                fn=ivy.exists,
                type="all",
                message=(
                    "input_channels and output_channels must both be"
                    " specified for fan_sum denominator mode."
                ),
                as_array=False,
            )
            fan = fan_in + fan_out
        elif self._fan_mode == "fan_avg":
            ivy.utils.assertions.check_all_or_any_fn(
                fan_in,
                fan_out,
                fn=ivy.exists,
                type="all",
                message=(
                    "input_channels and output_channels must both be"
                    " specified for fan_avg denominator mode."
                ),
                as_array=False,
            )
            fan = (fan_in + fan_out) / 2
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid denominator mode, must be one of [ fan_in | fan_out | "
                "fan_sum | fan_avg ] "
            )
        std = (2 / ((1 + negative_slope**2) * fan)) ** 0.5
        return _variable(
            ivy.random_normal(
                mean=self._mean, std=std, shape=var_shape, device=device, dtype=dtype
            )
        )


class RandomNormal(Initializer):
    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        """
        Initialize with Random Normal Distribution.

        It draws values from a Random Normal Distribution with given mean and
        standard deviation.

        Parameters
        ----------
        mean
            Sets the expected value, average, and center of the normal distribution.
        stddev
            Sets the standard deviation of the normal distribution.
        seed
            Used to create a random seed distribution.(Default:None)
        """
        self._mean = mean
        self._stddev = stddev
        self._seed = seed

    def create_variables(
        self,
        var_shape=None,
        device=None,
        dtype=None,
    ):
        """
        Create internal variables for the layer.

        Parameters
        ----------
        var_shape
            Tuple representing the shape of the desired array. If considering
             the array as a rectangular matrix, this tuple is represented as
             '(ROWS, COLUMNS)'.
        device
            Device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cpu.
        dtype
            Desired data type.
        """
        return _variable(
            ivy.random_normal(
                mean=self._mean,
                std=self._stddev,
                shape=var_shape,
                seed=self._seed,
                device=device,
                dtype=dtype,
            )
        )
