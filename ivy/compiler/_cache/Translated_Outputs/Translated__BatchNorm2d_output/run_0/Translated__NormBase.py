import ivy.functional.frontends.torch as torch
import ivy.functional.frontends.torch.nn as nn

import typing

from .helpers import Translated_ones_
from .helpers import Translated_zeros_


class Translated__NormBase(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: typing.Any
    eps: typing.Any
    momentum: typing.Any
    affine: typing.Any
    track_running_stats: typing.Any

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
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.parameter.Parameter(
                torch.empty(num_features, **factory_kwargs)
            )
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(num_features, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, **factory_kwargs)
            )
            self.register_buffer(
                "running_var", torch.ones(num_features, **factory_kwargs)
            )
            self.running_mean: typing.Any
            self.running_var: typing.Any
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                ),
            )
            self.num_batches_tracked: typing.Any
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            Translated_ones_(self.weight)
            Translated_zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = (
                    self.num_batches_tracked
                    if self.num_batches_tracked is not None
                    and self.num_batches_tracked.device != torch.device("meta")
                    else torch.tensor(0, dtype=torch.long)
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
