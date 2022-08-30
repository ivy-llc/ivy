# flake8: noqa
# global
import functools
import numpy as np
from operator import mul

# local
import ivy
from ivy.array.conversions import *


def _native_wrapper(f):
    @functools.wraps(f)
    def decor(self, *args, **kwargs):
        if isinstance(self, frontend_array):
            return f(self, *args, **kwargs)
        return getattr(self, f.__name__)(*args, **kwargs)

    return decor


class frontend_array:
    def _init(self, data):
        if ivy.is_ivy_array(data):
            self._data = data.data
        else:
            assert ivy.is_native_array(data)
            self._data = data
        self._shape = self._data.shape
        self._size = (
            functools.reduce(mul, self._data.shape) if len(self._data.shape) > 0 else 0
        )
        self._dtype = ivy.dtype(self._data)
        self._device = ivy.dev(self._data)
        self._dev_str = ivy.as_ivy_dev(self._device)
        self._pre_repr = "ivy."
        if "gpu" in self._dev_str:
            self._post_repr = ", dev={})".format(self._dev_str)
        else:
            self._post_repr = ")"
        self.framework_str = ivy.current_backend_str()
        self._is_variable = ivy.is_variable(self._data)

    # Properties #
    # -----------#

    # noinspection PyPep8Naming

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # Setters #
    # --------#

    @data.setter
    def data(self, data):
        assert ivy.is_native_array(data)
        self._init(data)

    # Built-ins #
    # ----------#

    @_native_wrapper
    def __array__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array__(*args, **kwargs)

    @_native_wrapper
    def __array_prepare__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_prepare__(*args, **kwargs)

    @_native_wrapper
    def __array_ufunc__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_ufunc__(*args, **kwargs)

    @_native_wrapper
    def __array_wrap__(self, *args, **kwargs):
        args, kwargs = args_to_native(*args, **kwargs)
        return self._data.__array_wrap__(*args, **kwargs)

    @_native_wrapper
    def __repr__(self):
        sig_fig = ivy.array_significant_figures()
        dec_vals = ivy.array_decimal_values()
        rep = (
            ivy.vec_sig_fig(ivy.to_numpy(self._data), sig_fig)
            if self._size > 0
            else ivy.to_numpy(self._data)
        )
        with np.printoptions(precision=dec_vals):
            return (
                self._pre_repr
                + rep.__repr__()[:-1].partition(", dtype")[0].partition(", dev")[0]
                + self._post_repr.format(ivy.current_backend_str())
            )

    @_native_wrapper
    def __dir__(self):
        return self._data.__dir__()

    @_native_wrapper
    def __getattr__(self, item):
        try:
            attr = self._data.__getattribute__(item)
        except AttributeError:
            attr = self._data.__getattr__(item)
        return to_ivy(attr)

    @_native_wrapper
    def __getitem__(self, query):
        query = to_native(query)
        return to_ivy(self._data.__getitem__(query))

    @_native_wrapper
    def __setitem__(self, query, val):
        try:
            self._data.__setitem__(query, val)
        except (AttributeError, TypeError):
            self._data = ivy.scatter_nd(
                query, val, tensor=self._data, reduction="replace"
            )._data
            self._dtype = ivy.dtype(self._data)

    @_native_wrapper
    def __contains__(self, key):
        return self._data.__contains__(key)

    @_native_wrapper
    def __getstate__(self):
        data_dict = dict()

        # only pickle the native array
        data_dict["data"] = self.data

        # also store the local ivy framework that created this array
        data_dict["framework_str"] = self.framework_str
        data_dict["device_str"] = ivy.as_ivy_dev(self.device)

        return data_dict

    @_native_wrapper
    def __setstate__(self, state):
        # we can construct other details of ivy.Array
        # just by re-creating the ivy.Array using the native array

        # get the required backend
        ivy.set_backend(state["framework_str"])
        ivy_array = ivy.array(state["data"])
        ivy.unset_backend()

        self.__dict__ = ivy_array.__dict__

        # TODO: what about placement of the array on the right device ?
        # device = backend.as_native_dev(state["device_str"])
        # backend.to_device(self, device)
