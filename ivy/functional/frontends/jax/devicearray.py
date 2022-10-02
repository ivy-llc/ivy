# global
import functools

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend


def _frontend_wrapper(f):
    @functools.wraps(f)
    def decor(self, *args, **kwargs):
        args = list(args)
        if isinstance(self, jax_frontend.DeviceArray):
            i = 0
            for arg in args:
                if isinstance(arg, jax_frontend.DeviceArray):
                    args[i] = arg.data
                i += 1
            for u, v in kwargs.items():
                if isinstance(v, jax_frontend.DeviceArray):
                    kwargs[u] = v.data
            return f(self, *args, **kwargs)
        return getattr(self, f.__name__)(*args, **kwargs)

    return decor


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

    def radd(self, other):
        return jax_frontend.add(self.data)

    @_frontend_wrapper
    def __add__(self, other):
        return jax_frontend.add(self.data, other)

    @_frontend_wrapper
    def __radd__(self, other):
        return jax_frontend.add(other, self.data)

    @_frontend_wrapper
    def __sub__(self, other):
        return jax_frontend.sub(self.data, other)

    @_frontend_wrapper
    def __rsub__(self, other):
        return jax_frontend.sub(other, self.data)

    @_frontend_wrapper
    def __mul__(self, other):
        return jax_frontend.mul(self.data, other)

    @_frontend_wrapper
    def __rmul__(self, other):
        return jax_frontend.mul(other, self.data)
