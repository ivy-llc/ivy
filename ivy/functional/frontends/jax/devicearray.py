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
        self.data = ivy.array(data)

    # Instance Methods #
    # ---------------- #

    def reshape(self, new_sizes, dimensions=None):
        return jax_frontend.reshape(self.data, new_sizes, dimensions)

    def add(self, other):
        return jax_frontend.add(self.data, other)

    # Special Methods #
    # --------------- #

    @_frontend_wrapper
    def __pos__(self):
        return ivy.positive(self.data)

    @_frontend_wrapper
    def __neg__(self):
        return jax_frontend.neg(self.data)

    @_frontend_wrapper
    def __eq__(self, other):
        return jax_frontend.eq(self.data, other)

    @_frontend_wrapper
    def __ne__(self, other):
        return jax_frontend.ne(self.data, other)

    @_frontend_wrapper
    def __lt__(self, other):
        return jax_frontend.lt(self.data, other)

    @_frontend_wrapper
    def __le__(self, other):
        return jax_frontend.le(self.data, other)

    @_frontend_wrapper
    def __gt__(self, other):
        return jax_frontend.gt(self.data, other)

    @_frontend_wrapper
    def __ge__(self, other):
        return jax_frontend.ge(self.data, other)

    @_frontend_wrapper
    def __abs__(self):
        return jax_frontend.abs(self.data)
