"""Converter from JAX Haiku Modules to Ivy Modules."""

# global
import haiku as hk
import jax
# noinspection PyProtectedMember
from haiku._src.data_structures import FlatMapping
import jax.numpy as jnp
# local
import ivy

RNG = jax.random.PRNGKey(0)


def _hk_flat_map_to_dict(hk_flat_map):
    ret_dict = dict()
    for k, v in hk_flat_map.items():
        new_k = k.replace("/", "|")
        if isinstance(v, FlatMapping):
            ret_dict[new_k] = _hk_flat_map_to_dict(v)
        else:
            ret_dict[new_k] = v
    return ret_dict


def _dict_to_hk_flat_map(dict_in):
    ret_flat_map = dict()
    for k, v in dict_in.items():
        new_k = k.replace("|", "/")
        if isinstance(v, dict):
            ret_flat_map[new_k] = _dict_to_hk_flat_map(v)
        else:
            ret_flat_map[new_k] = v
    return FlatMapping(ret_flat_map)


class IvyModule(ivy.Module):
    def __init__(
            self,
            *args,
            native_module,
            native_module_class,
            device,
            devices,
            **kwargs
    ):
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        ivy.Module.__init__(self, build_mode="on_init", device=device, devices=devices, *args, **kwargs)

    def _create_variables(self, device, dtype):
        return self._hk_params

    def _build(self, *args, **kwargs):
        args, kwargs = ivy.args_to_native(*args, **kwargs)
        # noinspection PyUnresolvedReferences
        params_hk = self._native_module.init(RNG, *args, **kwargs)
        params_dict = _hk_flat_map_to_dict(params_hk)
        self._hk_params = ivy.Container(params_dict)
        param_iterator = self._hk_params.to_iterator()
        _, param0 = next(param_iterator)
        self._dev = ivy.as_ivy_dev(param0.device())

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        params_hk = _dict_to_hk_flat_map(self.v.to_dict())
        ret = self._native_module.apply(params_hk, None, *a, **kw)
        if isinstance(ret, tuple):
            return ivy.args_to_native(*ret)
        return ivy.to_native(ret)


def to_ivy_module(
    native_module=None,
    native_module_class=None,
    args=None,
    kwargs=None,
    device=None,
    devices=None,
):

    args = ivy.default(args, [])
    kwargs = ivy.default(kwargs, {})

    if not ivy.exists(native_module):
        ivy.assertions.check_exists(
            native_module_class,
            message="native_module_class must be specified if native_module is None",
        )

        def forward_fn(*a, **kw):
            model = native_module_class(**kw)
            return model(*a)

        native_module = hk.transform(forward_fn)

    return IvyModule(
        native_module=native_module,
        native_module_class=None,
        device=device,
        devices=devices,
        *args,
        **kwargs
    )


def to_haiku_module(ivy_module, args=None, kwargs=None):

    class HaikuModule(hk.Module, ivy_module):
        def __init__(self):
            super(HaikuModule, self).__init__()
            ivy_module.__init__(self, **kwargs)

        def __call__(self, *args, **kwargs):
            self.v = self.v.map(
                lambda x, kc: hk.get_parameter
                (
                name=kc, shape=x.shape, dtype=x.dtype, init=lambda shape, dtype: ivy.to_native(self.v[kc]))
                )

            a, kw = ivy.args_to_native(*args, **kwargs)
            ret = self._forward(*a, **kw)
            if isinstance(ret, tuple):
                return ivy.args_to_native(*ret)
            return ivy.to_native(ret)

    def forward_fn(*a, **kw):
        model = HaikuModule()
        return model(*a, **kw)

    return HaikuModule
