# global
import functools

# local
import ivy
import ivy.functional.frontends.jax as jax_frontend
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


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

    def __repr__(self):
        return (
            "ivy.functional.frontends.jax.DeviceArray("
            + str(ivy.to_list(self.data))
            + ")"
        )

    # Instance Methods #
    # ---------------- #

    def reshape(self, new_sizes, dimensions=None):
        return jax_frontend.reshape(self.data, new_sizes, dimensions)

    def add(self, other):
        return jax_frontend.add(self.data, other)

    # Special Methods #
    # --------------- #

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

    @_frontend_wrapper
    def __div__(self, other):
        return jax_frontend.div(self.data, other)

    @_frontend_wrapper
    def __rdiv__(self, other):
        return jax_frontend.div(other, self.data)

    @_frontend_wrapper
    def __mod__(self, other):
        return jax_frontend.rem(self.data, other)

    @_frontend_wrapper
    def __rmod__(self, other):
        return jax_frontend.rem(other, self.data)

    @_frontend_wrapper
    def __divmod__(self, other):
        return divmod(self.data, other)

    @_frontend_wrapper
    def __rdivmod__(self, other):
        return divmod(self.data, other)

    @_frontend_wrapper
    def __truediv__(self, other):
        return jax_frontend.div(self.data, other)

    @_frontend_wrapper
    def __rtruediv__(self, other):
        return jax_frontend.div(other, self.data)

    @_frontend_wrapper
    def __floordiv__(self, other):
        return jax_frontend.floor_divide(self.data, other)

    @_frontend_wrapper
    def __rfloordiv__(self, other):
        return jax_frontend.floor_divide(other, self.data)

    @_frontend_wrapper
    def __matmul__(self, other):
        return jax_frontend.dot(self.data, other)

    @_frontend_wrapper
    def __rmatmul__(self, other):
        return jax_frontend.dot(other, self.data)

    @to_ivy_arrays_and_back
    def __pos__(self):
        return ivy.positive(self.data)

    @to_ivy_arrays_and_back
    def __neg__(self):
        return jax_frontend.neg(self.data)

    @to_ivy_arrays_and_back
    def __eq__(self, other):
        return jax_frontend.eq(self.data, other)

    @to_ivy_arrays_and_back
    def __ne__(self, other):
        return jax_frontend.ne(self.data, other)

    @to_ivy_arrays_and_back
    def __lt__(self, other):
        return jax_frontend.lt(self.data, other)

    @to_ivy_arrays_and_back
    def __le__(self, other):
        return jax_frontend.le(self.data, other)

    @to_ivy_arrays_and_back
    def __gt__(self, other):
        return jax_frontend.gt(self.data, other)

    @to_ivy_arrays_and_back
    def __ge__(self, other):
        return jax_frontend.ge(self.data, other)

    @to_ivy_arrays_and_back
    def __abs__(self):
        return jax_frontend.abs(self.data)

    @to_ivy_arrays_and_back
    def __pow__(self, other):
        return jax_frontend.pow(self.data, other)

    @to_ivy_arrays_and_back
    def __rpow__(self, other):
        return jax_frontend.pow(other, self.data)

    @to_ivy_arrays_and_back
    def __and__(self, other):
        return jax_frontend.bitwise_and(self.data, other)

    @to_ivy_arrays_and_back
    def __rand__(self, other):
        return jax_frontend.bitwise_and(other, self.data)

    @to_ivy_arrays_and_back
    def __or__(self, other):
        return jax_frontend.bitwise_or(self.data, other)

    @to_ivy_arrays_and_back
    def __ror__(self, other):
        return jax_frontend.bitwise_or(other, self.data)

    @to_ivy_arrays_and_back
    def __xor__(self, other):
        return jax_frontend.bitwise_xor(self.data, other)

    @to_ivy_arrays_and_back
    def __rxor__(self, other):
        return jax_frontend.bitwise_xor(other, self.data)

    @to_ivy_arrays_and_back
    def __invert__(self):
        return jax_frontend.bitwise_not(self.data)

    @to_ivy_arrays_and_back
    def __lshift__(self, other):
        return jax_frontend.shift_left(self.data, other)

    @to_ivy_arrays_and_back
    def __rlshift__(self, other):
        return jax_frontend.shift_left(other, self.data)

    @to_ivy_arrays_and_back
    def __rshift__(self, other):
        return jax_frontend.shift_right_logical(self.data, other)

    @to_ivy_arrays_and_back
    def __rrshift__(self, other):
        return jax_frontend.shift_right_logical(other, self.data)
