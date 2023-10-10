"""Collection of Ivy normalization classes."""

# local
import ivy
from ivy.stateful.module import Module
from ivy.stateful.initializers import Zeros, Ones, RandomNormal


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        /,
        *,
        eps: float = 1e-05,
        elementwise_affine: bool = True,
        new_std: float = 1.0,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Layer Normalization over a mini-batch of inputs.

        Parameters
        ----------
        normalized_shape
            Trailing shape to applying the normalization to.
        epsilon
            small constant to add to the denominator,
            use global ivy.min_base by default.
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
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self._normalized_idxs = [-(i + 1) for i in range(len(normalized_shape))]
        self._epsilon = eps
        self._elementwise_affine = elementwise_affine
        self._new_std = new_std
        self._weight_shape = normalized_shape
        self._bias_shape = normalized_shape
        self._weight_init = Ones()
        self._bias_init = Zeros()
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer."""
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
            eps=self._epsilon,
            scale=self.v.weight if self._elementwise_affine else None,
            offset=self.v.bias if self._elementwise_affine else None,
            new_std=self._new_std,
        )


class BatchNorm1D(Module):
    def __init__(
        self,
        num_features,
        /,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        data_format: str = "NSC",
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        v=None,
        dtype=None,
    ):
        """
        Class for applying Layer Normalization over a mini-batch of inputs.

        Parameters
        ----------
        num_features
            Trailing shape to applying the normalization to.
        epsilon
            small constant to add to the denominator,
            use global ivy.min_base by default.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
        affine
            Whether to include learnable affine parameters, default is ``True``.
        track_running_stats
            is a boolean flag that determines whether
            the running statistics should be updated
            during training in batch normalization.
        momentum
             The value used for the running_mean and running_var computation.
              Default is 0.1.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        v
            the variables for each submodule in the sequence,
            constructed internally by default.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            internal `mean` and `variance` when affine is True.
        """
        self.num_features = num_features
        self._affine = affine
        self.data_format = data_format
        self._epsilon = eps
        self._momentum = momentum
        self._track_running_stats = track_running_stats
        self._weight_shape = num_features
        self._bias_shape = num_features
        self._running_mean_shape = num_features
        self._running_var_shape = num_features
        self._weight_init = Ones()
        self._bias_init = Zeros()
        self._running_mean_init = Zeros()
        self._running_var_init = Ones()
        Module.__init__(self, device=device, v=v, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer."""
        if self._affine:
            return {
                "b": self._bias_init.create_variables(
                    self._bias_shape, device, dtype=dtype
                ),
                "running_mean": self._running_mean_init.create_variables(
                    self._running_mean_shape, device, dtype=dtype
                ),
                "running_var": self._running_var_init.create_variables(
                    self._running_var_shape, device, dtype=dtype
                ),
                "w": self._weight_init.create_variables(
                    self._weight_shape, device, dtype=dtype
                ),
            }
        return {}

    def _forward(
        self,
        inputs,
        training: bool = False,
    ):
        """
        Perform forward pass of the BatchNorm layer.

        Parameters
        ----------
        inputs
            Inputs to process of shape N,C,*.
        training
            Determine the current phase (training/inference)
        Returns
        -------
        ret
            The outputs following the batch normalization operation.
        """
        # TODO: Assert dimension == 1
        normalized, running_mean, running_var = ivy.batch_norm(
            inputs,
            self.v.running_mean,
            self.v.running_var,
            eps=self._epsilon,
            momentum=self._momentum,
            data_format=self.data_format,
            training=training,
            scale=self.v.w if self._affine else None,
            offset=self.v.b if self._affine else None,
        )
        if self._track_running_stats and training:
            self.v.running_mean = running_mean
            self.v.running_var = running_var

        return normalized


class BatchNorm2D(Module):
    def __init__(
        self,
        num_features,
        /,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        data_format: str = "NSC",
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        v=None,
        dtype=None,
        training=True,
    ):
        """
        Class for applying Layer Normalization over a mini-batch of inputs.

        Parameters
        ----------
        num_features
            Trailing shape to applying the normalization to.
        epsilon
            small constant to add to the denominator,
            use global ivy.min_base by default.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
        affine
            Whether to include learnable affine parameters, default is ``True``.
        track_running_stats
            is a boolean flag that determines whether
            the running statistics should be updated
            during training in batch normalization.
        momentum
             The value used for the running_mean and running_var computation.
              Default is 0.1.
        device
            device on which to create the layer's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. (Default value = None)
        v
            the variables for each submodule in the sequence,
            constructed internally by default.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            internal `mean` and `variance` when affine is True.
        """
        self.num_features = num_features
        self._affine = affine
        self.data_format = data_format
        self._epsilon = eps
        self._momentum = momentum
        self._track_running_stats = track_running_stats
        self._weight_shape = num_features
        self._bias_shape = num_features
        self._running_mean_shape = num_features
        self._running_var_shape = num_features
        self._weight_init = Ones()
        self._bias_init = Zeros()
        self._running_mean_init = Zeros()
        self._running_var_init = Ones()
        Module.__init__(self, device=device, v=v, dtype=dtype, training=training)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer."""
        if self._affine:
            return {
                "b": self._bias_init.create_variables(
                    self._bias_shape, device, dtype=dtype
                ),
                "running_mean": self._running_mean_init.create_variables(
                    self._running_mean_shape, device, dtype=dtype
                ),
                "running_var": self._running_var_init.create_variables(
                    self._running_var_shape, device, dtype=dtype
                ),
                "w": self._weight_init.create_variables(
                    self._weight_shape, device, dtype=dtype
                ),
            }
        return {}

    def _forward(self, inputs):
        """
        Perform forward pass of the BatchNorm layer.

        Parameters
        ----------
        inputs
            Inputs to process of shape N,C,*.

        Returns
        -------
        ret
            The outputs following the batch normalization operation.
        """
        normalized, running_mean, running_var = ivy.batch_norm(
            inputs,
            self.v.running_mean,
            self.v.running_var,
            eps=self._epsilon,
            momentum=self._momentum,
            data_format=self.data_format,
            training=self.training,
            scale=self.v.w if self._affine else None,
            offset=self.v.b if self._affine else None,
        )
        if self._track_running_stats and self.training:
            self.v.running_mean = running_mean
            self.v.running_var = running_var

        return normalized


class WeightNorm(Module):
    def __init__(self, layer, data_init=True, **kwargs):
        self.layer = layer
        self._weight_init = Ones()
        self._bias_init = Zeros()
        self._v_init = RandomNormal()
        self._g_init = Zeros()

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer."""
        v = {
            "w": self._weight_init.create_variables(self._w_shape, device, dtype),
        }
        v = dict(
            **v,
            b=self._b_init.create_variables(
                device=device,
                dtype=dtype,
            ),
        )
        dict(
            **v,
            v=self._v_init.create_variables(
                device=device,
                dtype=dtype,
            ),
        )
        dict(
            **v,
            g=self._g_init.create_variables(
                device=device,
                dtype=dtype,
            ),
        )
        return v

    def data_dep_init(self, x):
        # TODO: check expand dims works well with all types of layers
        self.v.w = ivy.l2_normalize(
            self.v.v, axis=self.kernel_norm_axes
        ) * ivy.expand_dims(self.v.g, -4)
        act = self.layer.activation
        self.layer.activation = None

        x_init = self.layer(x)

        norm_axes_out = list(range(x_init.shape.rank - 1))

        mean_t, std_t = ivy.mean(x_init, norm_axes_out), ivy.std(x_init, norm_axes_out)

        self.v.g = 1.0 / (std_t + 1e-10)

        self.v.b = -mean_t / std_t

        self.layer.activation = act

    def _call(self, x):
        self.data_dep_init(x)
        self.v.v, self.v.g = ivy.weight_norm(self.v.v, self.c.g)
        output = self.layer(x)
        return output
