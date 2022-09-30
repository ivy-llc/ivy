# local
import ivy
import ivy.functional.frontends.jax as jax_frontend
from ivy.array.array import _native_wrapper


class DeviceArray:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        self.data = data

    # Instance Methoods #
    # -------------------#
    def reshape(self, new_sizes, dimensions=None):
        return jax_frontend.reshape(self.data, new_sizes, dimensions)

    def add(self, other):
        return jax_frontend.add(self.data, other)

    @_native_wrapper
    def __add__(self, other):
        return jax_frontend.add(self.data, other)

    @_native_wrapper
    def __radd__(self, other):
        return jax_frontend.add(other, self.data)

    @_native_wrapper
    def __sub__(self, other):
        return jax_frontend.sub(self.data, other)

    @_native_wrapper
    def __rsub__(self, other):
        return jax_frontend.sub(other, self.data)

    @_native_wrapper
    def __mul__(self, other):
        return jax_frontend.mul(self.data, other)

    @_native_wrapper
    def __rmul__(self, other):
        return jax_frontend.mul(other, self.data)
