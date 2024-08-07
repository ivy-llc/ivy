import tensorflow


from .tensorflow__NormBase import tensorflow__NormBase
from .tensorflow__helpers import tensorflow_add__frnt_
from .tensorflow__helpers import tensorflow_batch_norm_frnt
from .tensorflow__helpers import tensorflow_handle_transpose_in_input_and_output
from .tensorflow__helpers import tensorflow_store_config_info


class tensorflow__BatchNorm(tensorflow__NormBase):
    @tensorflow_store_config_info
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    @tensorflow_handle_transpose_in_input_and_output
    def call(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                tensorflow_add__frnt_(self.num_batches_tracked, 1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        """
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        """
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        with tensorflow.name_scope("running_mean, selfrunning_var)"):
            normalized, self.running_mean, self.running_var = (
                tensorflow_batch_norm_frnt(
                    input,
                    self.running_mean
                    if not self.training or self.track_running_stats
                    else None,
                    self.running_var
                    if not self.training or self.track_running_stats
                    else None,
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps,
                )
            )
        return normalized
