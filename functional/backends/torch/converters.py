"""Converter from PyTorch Modules to Ivy Modules."""

# global
import torch
from collections import OrderedDict

# local
import ivy


class IvyModule(ivy.Module):
    def __init__(
        self,
        native_module_class,
        native_module,
        device,
        devices,
        inplace_update,
        *args,
        **kwargs
    ):
        self._native_module_class = native_module_class
        self._native_module = native_module
        self._args = args
        self._kwargs = kwargs
        self._update_v = (
            self._inplace_update_v if inplace_update else self._replace_update_v
        )
        ivy.Module.__init__(self, device=device, devices=devices)

    def _create_variables(self, device):
        return self._native_params

    def _build(self):
        self._native_module = ivy.default(
            lambda: self._native_module,
            lambda: self._native_module_class(*self._args, **self._kwargs),
            with_callable=True,
        )
        self._native_params = ivy.Container(
            OrderedDict(
                sorted(
                    [
                        (k.replace(".", "/"), v)
                        for k, v in dict(self._native_module.named_parameters()).items()
                    ]
                )
            )
        ).map(lambda x, kc: torch.nn.Parameter(x))

    @staticmethod
    def _inplace_update(p, v):
        p.data = v.data

    def _inplace_update_v(self, new_v):
        ivy.Container.multi_map(
            lambda xs, kc: self._inplace_update(xs[0], xs[1]),
            [self._native_params, new_v],
        )

    def _replace_update_v(self, new_v, native=None):
        native = ivy.default(native, self._native_module)
        for k, v in new_v.items():
            if isinstance(v, ivy.Container):
                # noinspection PyProtectedMember
                native._modules[k] = self._replace_update_v(v, native._modules[k])
            elif ivy.is_variable(v):
                if isinstance(v, torch.nn.Parameter):
                    # noinspection PyProtectedMember
                    native.__setattr__(k, v)
                else:
                    # noinspection PyProtectedMember
                    native.__setattr__(k, v.data)
            else:
                raise Exception(
                    "found item in variable container {} which was neither a "
                    "sub ivy.Container nor a variable.".format(v)
                )
        return native

    def _forward(self, *a, **kw):
        a, kw = ivy.args_to_native(*a, **kw)
        self._update_v(self.v)
        ret = self._native_module(*a, **kw)
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
    inplace_update=False,
):

    args = ivy.default(args, [])
    kwargs = ivy.default(kwargs, {})

    if not ivy.exists(native_module):
        if not ivy.exists(native_module_class):
            raise Exception(
                "native_module_class must be specified if native_module is not given"
            )

    return IvyModule(
        native_module_class,
        native_module,
        device,
        devices,
        inplace_update,
        *args,
        **kwargs
    )
