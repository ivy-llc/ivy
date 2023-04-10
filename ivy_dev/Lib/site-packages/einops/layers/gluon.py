from typing import Optional, Dict

import mxnet

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = 'Alex Rogozhnikov'


class Rearrange(RearrangeMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


class Reduce(ReduceMixin, mxnet.gluon.HybridBlock):
    def hybrid_forward(self, F, x):
        return self._apply_recipe(x)


class EinMix(_EinmixMixin, mxnet.gluon.HybridBlock):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        with self.name_scope():

            self.weight = self.params.get(name='weight', shape=weight_shape,
                                          init=mxnet.initializer.Uniform(weight_bound),
                                          )
            if bias_shape is not None:
                self.bias = self.params.get(name='bias', shape=bias_shape,
                                            init=mxnet.initializer.Uniform(bias_bound),
                                            )
            else:
                self.bias = None

    def _create_rearrange_layers(self,
                                 pre_reshape_pattern: Optional[str],
                                 pre_reshape_lengths: Optional[Dict],
                                 post_reshape_pattern: Optional[str],
                                 post_reshape_lengths: Optional[Dict]):
        if (pre_reshape_pattern is not None) or (post_reshape_pattern is not None):
            raise NotImplementedError("EinMix in mxnet/gluon doesn't support axis group/ungroup "
                                      "because einsum in gluon defined only for mx.np.ndarrays")

    def hybrid_forward(self, F, x, *args, **kwargs):
        # mxnet.np can't work with 'usual' ndarrays; .data() is a standard way to get within in gluon
        # .as_np_mndarray makes the necessary conversion
        result = mxnet.np.einsum(self.einsum_pattern, x.as_np_ndarray(), self.weight.data())
        if self.bias is not None:
            result += self.bias.data()
        return result
